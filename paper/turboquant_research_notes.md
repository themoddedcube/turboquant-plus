# TurboQuant+ research notes

Lens-specific notes on the **TurboQuant+ codec itself** — what
assumptions it bakes in, what knobs exist, what's open. Companion to:

- `paper/research_brief.md` — the paper-side framing (C1–C3) for the
  group-size and 3-bit value-quant contributions.
- `docs/03_experiment_value_quantization.md` — per-coord cosine sweeps
  across (bits, group_size) on captured K/V.

This doc is the codec-internals view of those: what does TurboQuant+
*actually* assume, where does it break, and what design space opens up
once you stop treating the QJL scale as a universal constant.

---

## 1. Codec recap (so this doc is self-contained)

TurboQuant+ key path (`turboquant/quantizer.py:189`, `TurboQuantProd`):

```
K cache:
  x → norm → normalize → rotate (Π via QR)
                              ↓
                       quantize coords (Lloyd-Max centroids, b-1 bits)
                              ↓
                       dequantize → residual = x − x̃_mse
                              ↓
                       QJL: signs = sign(S · residual), res_norm = ‖r‖
  decode:
                       x̃_mse + (α/d) · ‖r‖ · Sᵀ · signs
                              ↓
                       inverse_rotate → rescale by norm

V cache:
  group-quant (per-group min-max symmetric) — no QJL, no rotation.
```

The "+" in TurboQuant+ is the 1-bit QJL residual sketch on K. It's what
makes inner-product preservation work at low centroid resolution
without rate-blowup. Variants:

- `key_bits=3` is the default (4 MSE centroids + 1 sign bit).
- `key_bits=4` extends MSE to 8 centroids; QJL component is unchanged
  in structure.

Key fact: **QJL is K-only**. V quality is independent of α — V is
group-quantized at `turboquant/kv_cache.py:59` (`quantize_values`).
Mixed-precision per-layer work inherits this: α calibration is a
K-side concern only.

---

## 2. The α constant is an assumption, not a universal

The QJL decode applies the scale α/d, where α defaults to `sqrt(π/2)`
(`turboquant/quantizer.py:237-240`):

```
r̂ = (α/d) · ‖r‖ · Sᵀ · sign(S · r)
```

`sqrt(π/2)` is the **unbiased-estimator coefficient** for the 1-bit
sign-sketch under one specific input distribution: iid isotropic
Gaussian. Derivation: for `z ~ N(0,1)`, `E[|z|] = sqrt(2/π)`; the
inverse `sqrt(π/2)` converts the mean-absolute estimator into an
unbiased reconstruction in expectation. The closer the actual residual
is to isotropic Gaussian, the smaller the bias.

This is an **engineering choice baked into the codec**, not a
mathematical universal. Two consequences:

1. **Any centroid-table change shifts the residual distribution and
   miscalibrates α.** Retuning centroids per-layer (e.g. Lloyd-Max on
   captured residuals) can improve per-coord MSE while regressing
   end-to-end quality, because the residuals after retuning are no
   longer shaped like the residuals the fixed α assumes.

2. **The bias scales with how far the residual is from Gaussian.**
   Heavy-tailed layers get hit harder than near-Gaussian ones. The
   penalty shows up as inflated reconstruction norms and Δcos on the
   most outlier-heavy heads.

Fix shipped on this branch: expose α as `qjl_scale: Optional[float]`
on both `TurboQuantProd` (`turboquant/quantizer.py:197`) and
`TurboQuantKVCache` (`turboquant/kv_cache.py:176`, as `key_qjl_scale`).
Default `None` preserves the historical `sqrt(π/2)/d`; any retuned
centroid table can now ship with a co-tuned α as a coupled artefact.

---

## 3. MSE-min α ≠ unbiased α (finite-d distinction)

Earlier framing of α calibration mixed up two distinct optima. The
clean version:

- **MSE is quadratic in α** — fitting the codec's decoder
  `r̂ = (α/d)·‖r‖·g` gives the closed form
  `α*_MSE = d · Σ[‖r_i‖·⟨r_i, g_i⟩] / Σ[‖r_i‖²·‖g_i‖²]`,
  aggregable additively across samples. For unit-normalized residuals
  this simplifies to `d · Σ⟨u, g⟩ / Σ‖g‖²`.
- **Cos error is rational in α** — `cos(x, x̃_mse + (α/d)·g)` is
  `(linear in α) / sqrt(quadratic in α)`. Per-sample optimum is a
  rational function in measured quantities; no analytic aggregation
  across samples. Cos-max requires a 1D search (golden section
  converges in ~10 evals; α is bounded, function is unimodal).

The two optima diverge at finite d. For iid Gaussian residuals in
dimension d:

| α type     | Analytic value                   | At d=64    | At d=∞     |
|------------|----------------------------------|------------|------------|
| Unbiased   | sqrt(π/2)                        | 1.2533     | 1.2533     |
| MSE-min    | sqrt(2/π) / (1 + 2/π)            | 0.4875     | 1.2533     |

Derivation of the MSE-min asymptote. Let `r ∈ R^d`, `S ∈ R^{d×d}` with
iid `N(0,1)` entries, `z = S·r`, `s = sign(z)`, `g = Sᵀs`. Then:

- `⟨r, g⟩ = ⟨Sr, s⟩ = Σ|z_i|`, with `E[|z_i|] = ‖r‖·sqrt(2/π)`, so
  `E[⟨r, g⟩] = d · ‖r‖ · sqrt(2/π)`.
- `E[‖g‖²]` has a systematic part from `Sᵀs` aligned with `r`
  (`2/π`) plus the JL sketch's per-coordinate noise (`+1` at finite
  d). The noise term vanishes as `d → ∞`, so MSE-min α converges to
  the unbiased α in the limit.
- The MSE-optimal `α/d` is `E[⟨r,g⟩] / E[‖g‖²]`, giving
  `α = sqrt(2/π) / (1 + 2/π)`.

This is implemented in `turboquant/codebook.py:203`
(`calibrate_qjl_scale`) as the closed-form
`α* = d · Σ⟨r,g⟩ / Σ‖g‖²`. Test gates live in
`tests/test_qjl_calibration.py`:
`test_calibrate_qjl_scale_mse_min_on_gaussian` asserts agreement with
the analytic 0.4875 within 0.02 on 2000 d=64 Gaussian samples;
`test_calibrate_reduces_l2_error_vs_default_alpha` confirms the
calibrated α beats the default α on Laplace-shaped residuals (a
deliberate violation of the iid-Gaussian assumption).

**Practical implication for TurboQuant+ design:** the choice of which α
target to ship is a *design decision*, not a derivation. Three options:

| Choice | What it optimizes | Plumbing cost | When to prefer |
|---|---|---|---|
| Frozen sqrt(π/2) | Unbiased recovery under Gaussian residuals | Zero (default) | Centroids exactly match a Gaussian-residual table |
| Per-table MSE-min α | Per-vector L2 reconstruction error | Closed form, fast | Any centroid retune; conservative shrinkage estimator |
| Per-table cos-max α | Whole-vector cos similarity | 1D search per table | Cos / attention-quality is the downstream metric |

For attention-driven workloads (the actual deployment target), cos-max
α is the principled choice — but no closed form, so it's not in this
branch. MSE-min is the analytic placeholder. Whether MSE-min α wins on
whole-vector cos / PPL on real per-layer captures (vs only on per-coord
L2) is an open empirical question — see §6 Q1.

---

## 4. Implementation traps worth remembering

### 4.1 Codec sign convention vs `torch.sign`

The codec packs sign as `projected > 0 → 1` at
`turboquant/quantizer.py:244` and decodes via `2·signs − 1` at
`turboquant/quantizer.py:257`, so a zero projection is encoded as `0`
and decoded as `−1`. `torch.sign(0) == 0` would diverge at the zero
tie. Any external reimplementation of the QJL math (the calibrator,
an alternative decoder, a downstream verification model) must use
`torch.where(proj > 0, 1.0, -1.0)`, not `torch.sign(proj)`.

This is a silent correctness trap: code using `torch.sign` passes
every sanity check on continuous samples (ties have measure zero) and
only fails on synthetic zero-residual edge cases or fixed-point ties.
`codebook.calibrate_qjl_scale` uses the explicit
`torch.where(proj > 0, 1.0, -1.0)` form to stay bit-exact with the
codec.

### 4.2 Override is independent of centroids

`qjl_scale`, `centroids`, and `boundaries` are independent constructor
args on `TurboQuantProd` (`turboquant/quantizer.py:195-199`). This is
deliberate — α-only, centroid-only, and joint tuning are all
expressible. Centroid-only is exactly the failure mode where retuned
centroids regress end-to-end quality under a stale α; making it
explicit-but-distinct surfaces the choice rather than hiding it.

The validator at `turboquant/quantizer.py:122-130` requires centroids
and boundaries to be supplied together (or neither) and checks their
sizes against `2**bits` / `2**bits + 1`.

### 4.3 `qjl_scale` semantics: bare α, not α/d

`TurboQuantProd.qjl_scale` (the attribute) stores `α/d`, since that's
what the dequantize path multiplies by. The *constructor argument*
`qjl_scale` accepts the bare `α` and the constructor divides by `d`.
This matches the calibrator output convention: `calibrate_qjl_scale`
returns bare α (~1.2533 default, ~0.4875 MSE-min on Gaussian), and
callers pass that value straight in.

---

## 5. What this opens up in the design space

### 5.1 Per-layer centroid retune (immediate)

`(centroids, boundaries, qjl_scale)` is now a coupled per-table
artefact. An offline retune pipeline can emit all three together; the
KV cache consumes them as a unit via `TurboQuantKVCache`'s
`key_centroids` / `key_boundaries` / `key_qjl_scale` args
(`turboquant/kv_cache.py:174-176`). The "retuned centroids regress
end-to-end quality" failure mode is now re-runnable with co-tuned α
on the next PPL sweep.

### 5.2 4-bit MSE inherits cleanly

`key_bits=4` (8 MSE centroids + 1 sign bit) keeps the same QJL
component on a different-sized residual. Two reasons it should compose
better with default α than `key_bits=3`:

1. **Smaller residuals.** At 8 centroids the per-coord granular noise
   shrinks by ~4× (variance scales as bin-width² ∝ 1/N²_centroids for
   matched distributions). The Gaussian-residual assumption holds
   tighter at higher bit depths because the residual variance is
   smaller relative to its shape.
2. **More homogeneous bin structure.** 8 bins span the distribution
   more uniformly, so per-bin residuals are closer to "the same shape
   everywhere" than 4-bin's coarse outermost-vs-inner asymmetry.

Both effects mean `key_bits=4` should need *less* α retuning than
`key_bits=3` under retuned centroids. Worth measuring α drift on
captured `key_bits=4` residuals before assuming default α is fine.

### 5.3 Per-layer mode selection

With α as a per-layer knob (via `layer_idx` propagating into a lookup
of `(centroids, boundaries, qjl_scale)`), the hybrid recommendation
becomes:

- Outlier-heavy layers (typically layer 0, last layer, and a few
  middle layers) ship with `(centroids, boundaries, qjl_scale)`
  calibrated per layer.
- The remaining layers ship with the default Lloyd-Max table and
  `qjl_scale=None`.

Bus contract: the same `layer_idx` channel used for rotation seeding
indexes into the (centroid, α) table. One table per layer; α is a
single float alongside.

Storage cost: at `head_dim=128`, a per-layer centroid table is
`2**(b-1)` floats; an α is one float. At Qwen2-0.5B's 24 layers,
`key_bits=3` is `24 × (4 + 1) = 120` floats = 480 bytes. Negligible.

### 5.4 Higher-bit QJL?

Currently the QJL component is 1-bit (sign of `S·r`). Higher-bit QJL
(e.g. 2-bit with two sign-sketches against orthogonal projection sets,
or a multi-level sketch) could in principle reduce the residual
reconstruction noise further. Open question for future codec
versions: at what bit budget does multi-bit QJL beat "spend the bits
on more MSE centroids instead"? Cross-over likely depends on how
heavy-tailed the per-layer residuals are.

This is NOT in scope for the current α / centroid override work.
Flag for later codec generations.

---

## 6. Open questions (research log, not commitments)

1. **MSE-min vs cos-max α empirically.** Does the closed-form MSE-min
   α actually lift whole-vector cos / PPL on real per-layer captures,
   or does it only win on per-coord MSE (the metric retuning also won
   on while losing end-to-end)? Resolvable only by running a PPL
   sweep with `qjl_scale=calibrate_qjl_scale(...)` and comparing
   against:
   - default α, default centroids (current baseline)
   - default α, retuned centroids (the regression case)
   - MSE-min α, retuned centroids (this hypothesis)
   - cos-max α (1D search), retuned centroids (the principled
     alternative if MSE-min disappoints)

2. **Does α calibration generalize across input contexts?** Calibrated
   α is fit on residuals from a specific corpus. Does the same α work
   on out-of-distribution text? On longer-context generation? On other
   models (Qwen2-7B, Llama-3)? Answerable by capturing residuals on a
   held-out corpus and computing α independently.

3. **Per-layer α drift over training.** If the model is fine-tuned, do
   per-layer residual distributions shift enough to require α
   re-calibration? If so, α should be a per-deployment artefact, not a
   per-model-weight one.

4. **Norm inflation under default α.** Even with default centroids,
   median `‖K̂‖ / ‖K‖` tends to drift above 1 (typically 1.02–1.06).
   Likely a second-order interaction between α and the per-vector
   normalize/rescale round-trip. Does MSE-min α (which shrinks toward
   zero) reduce this, or shift it the other way? Quantifiable on the
   same captures the calibrator uses.

5. **Group-size interaction with QJL on K.** C1 (paper) shows V
   group-size is the dominant lever for V quality at zero cost. K is
   group-free (per-vector rotation + per-coord centroid), but the same
   "smaller groups = better" intuition might motivate per-head or
   per-coordinate-block α. Out of scope for this branch; flag for
   later.

---

## 7. Pointers

**This branch (codec side):**
- `turboquant/quantizer.py`
  - `TurboQuantMSE.__init__` — `centroids`/`boundaries` override at
    lines 101-130
  - `TurboQuantProd.__init__` — `qjl_scale` override at lines 191-240
  - `dequantize` (QJL decode path) — lines 287-300
  - QJL sign convention — line 244 (pack), 255-257 (unpack)
- `turboquant/codebook.py`
  - `calibrate_qjl_scale(residuals, qjl_matrix)` — line 203
  - Doc block above it explaining MSE-min vs cos-max α
- `turboquant/kv_cache.py`
  - `TurboQuantKVCache.__init__` — `key_centroids`,
    `key_boundaries`, `key_qjl_scale` plumb-through, lines 174-194
- `tests/test_qjl_calibration.py` — 7 test gates covering default
  parity, override-takes-effect, codebook override, validator
  size-check, MSE-min calibration on Gaussian, L2-error reduction on
  Laplace residuals

**Related docs:**
- `paper/research_brief.md` — paper-side framing (C1 group-size, C2
  3-bit values, C3 Triton kernels)
- `docs/03_experiment_value_quantization.md` — per-coord cosine sweeps
  that motivated C1/C2
- `docs/05_experiment_fused_decode.md` — fused decode kernel that
  currently consumes `qjl_scale` as a `tl.constexpr` scalar
- `paper/turboquant_plus_v2.tex` — published draft
