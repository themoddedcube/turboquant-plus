# TurboQuant+

Three targeted improvements to the TurboQuant KV cache compression algorithm
([Zandieh et al., arXiv:2504.19874](https://arxiv.org/abs/2504.19874)), with
end-to-end GPU validation and a paper draft. Every number in this README is
reproducible from the scripts in this repository on an NVIDIA RTX A4000.

**Paper draft**: [`paper/turboquant_plus_v2.tex`](paper/turboquant_plus_v2.tex)
· **Reference implementation**: [upstream TurboQuant](https://github.com/0xSero/turboquant)
· **License**: GPL-3.0

---

## TL;DR

| | TurboQuant (upstream, 2-bit gs=32) | **TurboQuant+ (3-bit gs=32)** | Delta |
|---|---|---|---|
| Per-token cosine similarity | 0.932 | **0.986** | **+0.054** |
| Bytes per token (d=128, values only) | 48 B | **64 B** | +16 B |
| Bytes per token vs 2-bit gs=16 (same quality tier) | 64 B | **64 B** | same |
| Rotation memory (d=128, O(d²) vs O(d)) | 64 KB (QR) | **640 B (RHT)** | **102× less** |
| WHT kernel speed vs cuBLAS dense matmul (d=128) | — | **7.67×** | — |
| Fused decode correctness (2-bit and 3-bit values) | 2-bit only | **all 9 configs pass** (max_err = 0, cos = 1.0) | + 3-bit path |
| Context extension vs FP16 (3-bit keys, 2-bit values) | 3.88× | **3.88×** | inherited |

The core result: **at the same 64 B/token budget, 3-bit gs=32 reaches cos=0.986
vs 0.952 for 2-bit gs=16** — a +0.034 quality gain with zero storage cost.

---

## What's new

### 1. True 3-bit value quantization

Upstream ships 2-bit and 4-bit value quantization. The natural middle — 3-bit —
wasn't packed efficiently (naive 1-byte-per-value storage wastes 5 bits). This
branch implements lossless 8-values-per-3-bytes packing in
[`turboquant/kv_cache.py`](turboquant/kv_cache.py).

| bits | group size | cos_sim | bytes / token (d=128) | compression vs FP16 |
|:---:|:---:|:---:|:---:|:---:|
| 2 | 32 | 0.9329 | 48 | 5.33× |
| 2 | 16 | 0.9521 | 64 | 4.00× |
| **3** | **32** | **0.9864** | **64** | **4.00×** |
| 3 | 16 | 0.9905 | 80 | 3.20× |
| 4 | 16 | 0.9979 | 96 | 2.67× |

Measured by `exp_a_values.py`, d=128, N=256 Gaussian samples. At the 64 B/token
tier, 3-bit gs=32 is Pareto-optimal.

### 2. Randomized Hadamard Transform (RHT) replaces QR rotation

Upstream stores a dense `d × d` orthogonal matrix for rotation. Replacing it
with an RHT (diagonal random signs + random permutation + Walsh-Hadamard
Transform) gives identical quantization quality at O(d) memory.

| Test | QR rotation | RHT | Difference |
|---|---|---|---|
| 2-bit cosine similarity | 0.9521 | 0.9523 | +0.0002 |
| 3-bit cosine similarity | 0.9881 | 0.9881 | <0.0001 |
| 4-bit cosine similarity | 0.9979 | 0.9979 | <0.0001 |
| Memory (d=128) | 64 KB | 640 B | **102× less** |
| Memory (d=256) | 256 KB | 1,280 B | **205× less** |

Measured by `exp_b_hadamard.py`. Full derivation and correctness proof in
[`turboquant/wht_kernel.py`](turboquant/wht_kernel.py).

### 3. Triton WHT kernel

The reference Python butterfly WHT is O(d log d) in FLOPs but dominated by
Python interpreter overhead. This branch ships a Triton JIT kernel that runs
the butterfly in-register for d ∈ {64, 128, 256, 512}.

| d | Triton WHT | cuBLAS dense matmul | Speedup |
|:---:|:---:|:---:|:---:|
| 64 | 0.070 ms | 0.470 ms | **6.71×** |
| 128 | 0.070 ms | 0.535 ms | **7.67×** |
| 256 | 0.071 ms | 0.637 ms | **9.00×** |

Benchmark: 4096-row batch, float32, RTX A4000, 200 repeats after warmup. The
Python butterfly takes 62.8 ms for a single 256-row call — a ~900× speedup.

Correctness validated against a Sylvester-constructed Hadamard matrix:
max reconstruction error ~1e-7 across d ∈ {64, 128, 256} (float32 noise floor).

### 4. Overridable QJL scale + per-table calibration

The decoder constant α (default `sqrt(π/2)`, baked into upstream as the
unbiased coefficient under iid-Gaussian residuals) is now a constructor
knob on both `TurboQuantProd` and `TurboQuantKVCache`. Centroid tables
are likewise overridable, so a per-layer Lloyd-Max retune can ship as a
coupled `(centroids, boundaries, qjl_scale)` artefact.

```python
from turboquant.codebook import calibrate_qjl_scale
from turboquant.kv_cache import TurboQuantKVCache

# Calibrate MSE-min α on captured rotated residuals.
alpha = calibrate_qjl_scale(residuals, S)  # bare α; ~0.4875 on Gaussian at d=64

cache = TurboQuantKVCache(
    head_dim=128, key_bits=3,
    key_centroids=retuned_centroids,
    key_boundaries=retuned_boundaries,
    key_qjl_scale=alpha,
)
```

`qjl_scale=None` (default) preserves bit-exact prior behaviour. Closed
form: `α* = d · Σ[‖r‖·⟨r,g⟩] / Σ[‖r‖²·‖g‖²]` where `g = Sᵀ·sign(S·r)`.
Design rationale and the MSE-min vs cos-max α discussion live in
[`paper/turboquant_research_notes.md`](paper/turboquant_research_notes.md);
verification gates in `tests/test_qjl_calibration.py` (8/8 passing,
including a Gaussian-residual analytic check at d=64).

---

## Quickstart

```bash
pip install -e .
```

Requirements: Python 3.10+, PyTorch 2.10+, Triton 3.6+, CUDA 12.x.

### Reproduce the paper

```bash
# Value quantization quality sweep (Table 1 in paper)
python exp_a_values.py

# RHT vs QR rotation equivalence (Table 4 in paper)
python exp_b_hadamard.py

# Fused decode correctness + throughput (Tables 2, 3 in paper)
python exp_c_fused.py

# Triton WHT kernel correctness test
python -c "from turboquant.wht_kernel import test_rht_correctness; \
           print('PASS' if test_rht_correctness() else 'FAIL')"
```

Each script is self-contained and runs in under 5 minutes on an RTX A4000.

---

## Results detail

### Fused decode correctness (`exp_c_fused.py`)

All 9 configurations pass with max_err=0 and cosine=1.0 against a hybrid
reference (Triton attention scoring + PyTorch softmax + dequantized values):

```
  BH= 8 N=  256 D=128 vbits=2 gs=32  max_err=0.00000  cos=1.000000  PASS
  BH=16 N= 1024 D=128 vbits=2 gs=32  max_err=0.00000  cos=1.000000  PASS
  BH=16 N= 1024 D=256 vbits=2 gs=32  max_err=0.00000  cos=1.000000  PASS
  BH=32 N= 4096 D=128 vbits=2 gs=32  max_err=0.00000  cos=1.000000  PASS
  BH= 8 N=  256 D=128 vbits=3 gs=32  max_err=0.00000  cos=1.000000  PASS
  BH=16 N= 1024 D=128 vbits=3 gs=32  max_err=0.00000  cos=1.000000  PASS
  BH=16 N= 1024 D=128 vbits=3 gs=16  max_err=0.00000  cos=1.000000  PASS
  BH=16 N= 1024 D=256 vbits=3 gs=32  max_err=0.00000  cos=1.000000  PASS
  BH=32 N= 4096 D=128 vbits=3 gs=32  max_err=0.00000  cos=1.000000  PASS
```

The 3-bit value configurations are new in this branch and validate the
`unpack_values` bits=3 branch added in
[`docs/07_kernel3_3bit_fix.md`](docs/07_kernel3_3bit_fix.md).

### Fused decode throughput (BH=32, D=128, 100 repeats)

| Context (N) | FP16 baseline | Hybrid (Triton scores + PyTorch) | Fused (Triton) |
|:---:|:---:|:---:|:---:|
| 256 | 0.133 ms | 0.484 ms | 0.340 ms |
| 1,024 | 0.254 ms | 0.457 ms | 0.345 ms |
| 4,096 | 0.896 ms | 1.616 ms | 1.081 ms |
| 16,384 | 3.453 ms | 6.401 ms | 4.334 ms |

See the [honest caveats](#limitations-and-honest-caveats) section for how to
read these numbers — on the A4000 the fused kernel is slower than FP16 in wall
time. TurboQuant+'s win on this hardware is **memory extension**, not raw
decode latency.

### Context capacity (RTX A4000, 16.8 GB, 32 heads)

| Config | Bytes/token | Max context | Extension vs FP16 |
|---|:---:|:---:|:---:|
| FP16 baseline | 512 | 615 k | 1.00× |
| 3-bit keys + 2-bit values (gs=32) | 132 | **2,386 k** | **3.88×** |
| 3-bit keys + 3-bit values (gs=32) | 148 | 2,130 k | 3.46× |
| 3-bit keys + 4-bit values (gs=32) | 164 | 1,920 k | 3.12× |

From `exp_c_fused.py`, assuming 60% of VRAM allocated to KV cache.

---

## Comparison to upstream TurboQuant

The upstream README (on the `main` branch of
[0xSero/turboquant](https://github.com/0xSero/turboquant)) documents a vLLM
integration with end-to-end inference benchmarks:

- **RTX 5090 (32 GB)** — Qwen3.5-27B-AWQ, TP=1, vLLM 0.18.0: +5.7 % prefill,
  +3.1 % decode, 30 GB KV freed, 2.0× token capacity.
- **8× RTX 3090 (24 GB)** — Qwen3.5-35B-A3B MoE, TP=8: 30.9 % KV savings
  across the 10 full-attention layers, 1.45× context extension.

Those benchmarks require the upstream integration scripts
(`validate_paper.py`, `validate_moe*.py`, `profile_100k.py`, etc.) and models
that live on the upstream main branch. **They are not reproducible on this
branch** and are not the focus of TurboQuant+. This branch validates the three
algorithmic improvements in isolation on commodity hardware; production
integration work is upstream.

Specific upstream claims, revisited:

| Upstream statement | TurboQuant+ |
|---|---|
| *"Value quantization is the bottleneck: 2-bit values cause cos_sim=0.94 degradation."* | Confirmed — and addressed. At the same 64 B/token budget, 3-bit gs=32 reaches cos=0.986. |
| *"Hybrid decode dequantizes all history per decode step."* | Kernel 3 (`turboquant_fused_decode`) now validated end-to-end with both 2-bit and 3-bit values; no full-history dequant on the fast path. |
| *"200–400× faster than Python butterfly"* (WHT docstring) | Measured on A4000: ~900× vs Python, 6.71–9.00× vs cuBLAS dense matmul. |
| *"Value quant 2-bit: cos_sim 0.940"* (head_dim=256) | Consistent with our d=128 measurement of 0.9329 (2-bit gs=32) — upstream uses a different metric. TurboQuant+ raises this to 0.986 at the same budget. |

---

## Architecture

```
turboquant/
  codebook.py        Lloyd-Max optimal scalar quantizer + calibrate_qjl_scale
  codebooks/         Pre-generated codebooks (d=128/256, bits 1..4)
  rotation.py        Legacy QR rotation + QJL projection matrices
  wht_kernel.py      Triton WHT + RHTRotation (102× less memory than QR)
  quantizer.py       TurboQuantMSE + TurboQuantProd (overridable centroids/α)
  kv_cache.py        Value bit-packing (2/3/4/8-bit, group-asymmetric)
  triton_kernels.py  Three fused Triton kernels for decode attention

tests/
  test_qjl_calibration.py   Override + MSE-min calibration gates

docs/
  00_codebase_overview.md       Repo tour
  01..05_experiment_*.md        Per-experiment design notes
  06_improvements_shipped.md    Quality deltas and diffs
  07_kernel3_3bit_fix.md        3-bit fused-kernel enablement
  08_wht_kernel.md              Triton WHT design and benchmarks

paper/
  turboquant_plus_v2.tex        Paper draft
  research_brief.md             Single-page summary
  neurips.sty, extra_pkgs.tex   Style

exp_a_values.py       Value quantization quality sweep
exp_b_hadamard.py     RHT vs QR rotation equivalence
exp_c_fused.py        Fused decode correctness + throughput
```

---

## Limitations and honest caveats

1. **All quality numbers are on synthetic Gaussian inputs.** `exp_a_values.py`
   samples from `torch.randn`. Real KV tensors from trained models have
   channel-wise outlier structure ([SmoothQuant](https://arxiv.org/abs/2211.10438))
   that can degrade group-quantization quality beyond what Gaussian inputs
   suggest. The paper acknowledges this explicitly (Section 6, Limitations).
   End-to-end PPL on a real model is the next validation milestone.

2. **The 7.67× WHT speedup is vs cuBLAS dense matmul, not a hand-tuned QR
   implementation.** cuBLAS is the natural comparison for "how QR would be
   implemented in practice," but a specialized QR kernel would close part of
   the gap. The claim is that WHT is competitive with the best available
   dense-matmul alternative, not maximally faster than any possible baseline.

3. **Fused decode on RTX A4000 is slower than FP16 baseline** (0.3–0.8× of
   FP16 wall time across N=256..16384). The A4000 has strong FP16 tensor cores
   that dominate the comparison on this hardware. TurboQuant+'s value on the
   A4000 is the 3.88× memory extension, not raw decode speed. Upstream's 5.7 %
   prefill / 3.1 % decode wins are on RTX 5090 under a different workload.

4. **No inference-quality numbers yet** (PPL, MMLU, HumanEval). The
   contribution is algorithmic: we show cosine-similarity quality improvements
   on the quantization path itself. Demonstrating that these translate into
   end-task quality wins requires integration with a real inference stack and
   is left to upstream or future work.

5. **RHT requires d to be a power of 2**. The kernel supports d ∈ {16, 32, 64,
   128, 256, 512}. Non-power-of-2 head dimensions require zero-padding, which
   the RHTRotation class handles but at non-lossless inverse accuracy.

6. **Correctness claims apply to the shipped code as of commit `2c976f6`.**
   The Triton WHT kernel has three bugs fixed on this branch (butterfly sign,
   cross-warp barrier, constexpr sqrt — see commit `4c8d571`); earlier commits
   of this branch do not pass `test_rht_correctness` for d ≥ 128.

---

## Citation

Upstream TurboQuant paper:

```bibtex
@misc{zandieh2025turboquant,
  title         = {TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author        = {Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  year          = {2025},
  eprint        = {2504.19874},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2504.19874}
}
```

TurboQuant+ (this repository) — citation pending paper submission; in the
meantime, see [`paper/turboquant_plus_v2.tex`](paper/turboquant_plus_v2.tex).

---

## Environment

Validation hardware:

- NVIDIA RTX A4000, 16 GB, compute capability 8.6, CUDA 12.8
- Ubuntu 22.04 (Shadeform / Brev cloud GPU instance)
- Python 3.10, PyTorch 2.11.0+cu128, Triton 3.6.0
