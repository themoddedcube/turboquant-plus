# 09 — TurboQuant+ under per-token adaptive precision on GQA: a negative result (and the fix)

> Intended location in this repo: `docs/09_gqa_per_token_limitation.md`.
> Authored 2026-06-18 from an integration study combining TurboQuant+ (this
> codec) with **Don't Waste Bits!** (DWB, arXiv:2604.04722) per-token adaptive
> precision, evaluated on **task accuracy** (HellaSwag), on a **GQA** model
> (Qwen2-0.5B). Companion repos: the DWB controller + verification
> (`dont-waste-bits`), the chip-side codec (`kv-cache-engine`), and the eval
> harness (`adaptive-precision-attention/analysis/c13–c15`).

## TL;DR

TurboQuant+ is validated in this repo on **reconstruction quality** (per-token
cosine ≈ 0.99) and **uniform** low-bit value quantization, primarily on MHA.
This study asked a different question: *does TurboQuant+ preserve **task
accuracy** when paired with a per-token importance controller (DWB) on a **GQA**
model?* The answer is **no** — and the reason is the rotation.

| setting (Qwen2-0.5B, HellaSwag acc_norm, n=250, CPU/fp32) | acc_norm | avg bits |
|---|---:|---:|
| FP16 (no KV quant) | **0.420** | 16 |
| TurboQuant+ codec, **uniform** | 0.352 | ~4 |
| TurboQuant+ codec + **DWB routing** (best variant) | 0.360 | ~7.6 |
| TurboQuant+ codec + **oracle** routing (50% lossless) | 0.348 | ~10 |
| **scalar INT4** + oracle routing (50% lossless) | **0.412** | ~10 |
| **scalar {4,8,16} + DWB learned controller** | **0.416** | **7.6** |

**The codec is the limiter, not GQA and not DWB.** Same routing, same protected
set, same model — swapping TurboQuant+'s rotated vector codec for a plain scalar
quantizer moves the result from 0.348 (no recovery) to 0.412–0.416 (≈ FP16).

## Context — what was tested

DWB assigns a per-token bit-width ∈ {2,4,8,16} via a tiny learned controller
(important tokens kept high-precision, the rest compressed). The hope: use
TurboQuant+ as the compressor for the low tiers, getting its strong low-bit
reconstruction *and* DWB's selective protection.

On **SmolLM-360M (MHA)** this works — reproduced here: DWB routing through the
vector codec ("DWB-TurboQuant") reaches **42.0%** HellaSwag vs **42.6%** FP16
(−0.6pp), beating scalar INT by +2pp at the 2-bit tier. So the codec + routing
combination is sound *on MHA*.

On **Qwen2-0.5B (GQA: 2 KV heads shared across 14 query heads, 7:1)** it fails.

## Results — the elimination

All HellaSwag, Qwen2-0.5B, n=250, CPU/fp32, length-normalized `acc_norm`.

**1. Uniform codec is lossy (expected):** TurboQuant+ uniform = 0.352 vs FP16
0.420 (−0.068). Scalar INT4 uniform = 0.360. Comparable — uniform 4-bit KV
quant costs ~0.06 on this GQA model either way.

**2. DWB routing through the codec does not recover:**
- {2,4,8,16} tiers: 0.316 · drop 2-bit {4,8,16}: 0.360 · drop 2-bit + route
  important→FP16: 0.344. All ≈ or below uniform turbo4, none near FP16.

**3. Even an oracle through the codec fails:** route a sink+recent importance
heuristic — 50% of tokens at lossless FP16, rest through the codec — and it
still scores **0.348** (≈ uniform). Protecting 50% of tokens losslessly buys
*nothing*. Since the oracle is the ceiling of any controller, **no controller
can fix this through the codec.**

**4. The same oracle through scalar INT4 recovers:** identical routing, identical
protected 50%, only the compressor for the other half changed (scalar INT4
instead of the rotated codec) → **0.412 ≈ FP16.**

**5. The deployable result — learned controller + scalar tiers:** the *trained*
Qwen DWB controller driving scalar tiers, with the 2-bit tier dropped
(`{4→INT4, 8→INT8, 16→FP16}`) → **0.416 at 7.6 bits/token** (−0.004 vs FP16).
The raw controller including a scalar INT2 tier drops to 0.336 — the 2-bit tier
is catastrophic and must be dropped (consistent with the FPGA-side BRAM finding:
2-bit gives no bandwidth benefit *and* wrecks accuracy).

## Mechanism — why the rotation backfires on GQA

TurboQuant+'s strength is its rotation (RHT/WHT) + QJL residual: it decorrelates
the head-dim coordinates so quantization error spreads *uniformly* across the
vector. That is exactly what makes it a great **uniform** low-bit codec.

But "spread the error uniformly across the head dimension" is the **opposite** of
what per-token protection needs. Two compounding effects on GQA:

1. **Rotation delocalizes the error.** After the WHT, a token's quantization
   error is smeared across all D coordinates of its KV vector. So even when the
   controller marks a token "important" and the *other* tokens are compressed,
   the smeared error from those other tokens still corrupts the shared
   representation. Per-token protection can't "catch" an error that isn't
   localized to the tokens being compressed. Scalar quantization keeps each
   token's error token-local, so protecting the important tokens actually works.

2. **GQA amplifies it.** With 2 KV heads shared 7:1, every quantized KV vector
   feeds 7 query heads, so any residual error is amplified across the attention
   computation. MHA contains the error to one head.

Together: the codec's cleverness (rotation) and GQA's sharing combine so that the
loss becomes **diffuse and unprotectable**. A "dumber" scalar quantizer, whose
error stays token-local, is *better* in this regime — the oracle/controller can
protect the tokens that carry the error.

## Methodological note — cosine similarity ≠ task accuracy here

This repo's headline metric is per-token reconstruction **cosine similarity**
(0.986 for 3-bit gs=32). That metric stays high while **task accuracy collapses**
under per-token routing on GQA. Reconstruction fidelity of an *individual* KV
vector does not capture the *delocalized, attention-amplified* error that
actually degrades the task. Any adaptive-precision claim should be validated on
task accuracy, not cosine (or perplexity) alone.

## Recommendation / scope

- **TurboQuant+ as shipped is sound for what it claims:** uniform low-bit value
  quantization with strong reconstruction and fast kernels, especially on MHA.
  Nothing here contradicts the repo's cosine/throughput results.
- **Do not pair the rotated vector codec with per-token adaptive precision on
  GQA.** It does not recover task accuracy; a scalar-INT tiering does.
- **For adaptive-precision KV compression on GQA**, the working recipe is:
  DWB-style learned controller → **scalar-INT tiers `{4,8,16}` (drop 2-bit)**.
  On Qwen2-0.5B this reaches **0.416 ≈ FP16 at 7.6 bits/token**.
- **Open question for a future TurboQuant+ revision:** can the rotation be made
  per-token-protection-friendly on GQA (e.g. localized/blockwise rotation that
  doesn't smear error across the shared head dim, or skip rotation on the
  high-importance tiers)? That would be the way to keep the codec's low-bit edge
  *and* DWB's selectivity.

## Reproduction

Harnesses (in `adaptive-precision-attention/analysis/`, CPU/fp32, no GPU):
- `c13_dwb_routed_hellaswag.py` — DWB routing through the KVCE/TurboQuant codec
  (`--floor-tier`, `--bypass-above`, `--oracle-frac` flags). Results: rows 2–3.
- `c14_scalar_oracle_qwen.py` — scalar-INT isolation (oracle). Row 4.
- `c15_dwb_scalar_qwen.py` — learned controller + scalar tiers. Row 5.

DWB controller: `dont-waste-bits/research/data/dwb_controller_qwen2-0.5b.pt`
(trained via `research/src/run_qwen_controller.py`). SmolLM reproduction:
`dont-waste-bits/research/src/run_turboquant_h2.py`.

## Caveats

- Single model (Qwen2-0.5B) and one benchmark (HellaSwag); the GQA mechanism
  should generalize but is not yet multi-model-confirmed.
- 0.5B is a small, quantization-fragile scale; published low-bit KV results
  (KIVI, KVQuant) are mostly ≥7B, where margins are larger.
- The deployable 0.416 result is at 7.6 bits/token (~2.1× vs FP16) — near-
  lossless but a modest compression ratio; pushing avg-bits down (β-tuning the
  controller) without losing accuracy is future work.
- CIs are ~±0.057 at n=250; the codec-vs-scalar *dissociation* is the robust
  signal (0.348 vs 0.412 at identical routing), not any single cell.
