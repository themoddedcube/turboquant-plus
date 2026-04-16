# Experiment 04 — Randomized Hadamard Transform vs QR Rotation

**Date**: 2026-04-16  
**Hardware**: NVIDIA RTX A4000 (16.8 GB), CUDA 12.8  
**Goal**: Replace the O(d²) stored QR rotation matrix with O(d log d) RHT

---

## Summary

RHT produces **identical quality** to QR (< 0.001 cosine difference at all dims/bits). Memory savings are massive (100-200x smaller). However, the Python butterfly implementation is **400x slower** on GPU vs cuBLAS matmul. A production RHT requires a custom Triton WHT kernel — the algorithm is validated, just needs a fast kernel.

---

## Correctness

RHT roundtrip error (forward ∘ inverse = identity):

| d   | max error |
|-----|-----------|
| 64  | 3.58e-07  |
| 128 | 4.77e-07  |
| 256 | 4.77e-07  |

Numerically perfect (floating point rounding only).

---

## Quality: QR vs RHT

| d   | bits | QR cos  | RHT cos | diff    | verdict |
|-----|------|---------|---------|---------|---------|
| 64  | 2    | 0.94240 | 0.94171 | −0.00069 | same |
| 64  | 3    | 0.98375 | 0.98355 | −0.00021 | same |
| 64  | 4    | 0.99552 | 0.99564 | +0.00013 | same |
| 128 | 2    | 0.94051 | 0.94064 | +0.00012 | same |
| 128 | 3    | 0.98307 | 0.98309 | +0.00002 | same |
| 128 | 4    | 0.99544 | 0.99539 | −0.00005 | same |
| 256 | 2    | 0.93976 | 0.94046 | +0.00069 | same |
| 256 | 3    | 0.98273 | 0.98280 | +0.00006 | same |
| 256 | 4    | 0.99536 | 0.99532 | −0.00004 | same |

**All differences are within statistical noise (< 0.001).** RHT is a theoretically sound drop-in replacement — the quality is equivalent because both are random orthogonal transforms; the specific realization doesn't matter for quantization quality.

---

## Speed: QR (cuBLAS matmul) vs RHT (Python butterfly)

| d   | N      | QR ms  | RHT ms  | speedup |
|-----|--------|--------|---------|---------|
| 64  | 1024   | 0.023  | 5.49    | 0.004x (QR is 240x faster) |
| 64  | 16384  | 0.030  | 5.58    | 0.005x |
| 128 | 1024   | 0.028  | 12.73   | 0.002x |
| 128 | 4096   | 0.060  | 10.87   | 0.006x |
| 128 | 16384  | 0.065  | 10.76   | 0.006x |
| 256 | 1024   | 0.023  | 21.44   | 0.001x |
| 256 | 16384  | 0.214  | 21.60   | 0.010x |

The Python butterfly loop is ~200-400x slower than cuBLAS. This is expected — cuBLAS uses tensor cores for the d×d matmul, while our butterfly runs as a Python-level loop that launches many small CUDA kernels per step.

**This is a Python implementation problem, not an algorithm problem.** A Triton WHT kernel would be competitive. For d=128: WHT is O(128 × 7) = 896 multiplications vs matmul O(128²) = 16384. Proper WHT should be ~18x fewer FLOPs.

---

## Memory Footprint

| d   | QR matrix | RHT (float signs) | RHT (int8 signs) | ratio (QR/RHT_opt) |
|-----|-----------|------------------|------------------|--------------------|
| 64  | 16 KB     | 512 B            | 320 B            | 51x smaller |
| 128 | 64 KB     | 1024 B           | 640 B            | 102x smaller |
| 256 | 256 KB    | 2048 B           | 1280 B           | 205x smaller |

For a 32-layer model with heads_per_layer=32 and d=128:
- QR: 32 × 64KB = 2 MB (negligible individually, but multiplied across tensor-parallel ranks)
- RHT: 32 × 640B = 20 KB — essentially free

In a 4-GPU tensor parallel setup: each GPU stores its own set of rotation matrices. RHT reduces rotation matrix overhead from 8 MB to 80 KB total.

---

## How RHT Works

```
forward(x):
  1. pad x to d_pad = next_power_of_2(d)
  2. multiply by random ±1 signs: x ← x ⊙ r
  3. apply Hadamard transform: x ← H(x) / sqrt(d_pad)
  4. permute coordinates: x ← x[perm][:d]

inverse(y):
  1. reverse permute: y ← y[inv_perm]
  2. apply Hadamard (self-inverse up to scale): y ← H(y)
  3. remove signs: y ← y ⊙ r
  4. take [:d]
```

The random signs `r` and permutation `perm` are deterministic from a seed — no need to store them as full matrices. `r` can be stored as int8 (1 byte per element), `perm` as int32 (4 bytes per element).

---

## Recommendation

1. **Implement a Triton WHT kernel** (`triton_kernels.py`) — this is the key enabling step
2. Replace `generate_rotation_matrix` call in `TurboQuantMSE.__init__` with `RHTRotation`
3. Store seeds (not matrices) for layer-wise rotation — saves 2 MB per model
4. The quality is proven equivalent — only the speed optimization remains

**Estimated speedup of proper Triton WHT vs cuBLAS**: ~5-15x (due to ~18x fewer FLOPs + reduced memory bandwidth from no d×d matrix load). This would reduce the per-decode rotation cost from ~0.06ms to ~0.005ms per layer.
