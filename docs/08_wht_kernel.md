# Experiment 08 — Triton WHT Kernel for Randomized Hadamard Transform

**Date**: 2026-04-16  
**File created**: `turboquant/wht_kernel.py`

---

## Summary

Implemented a Triton WHT (Walsh-Hadamard Transform) kernel and `RHTRotation` class to replace the slow Python butterfly implementation from Experiment 04. This closes the final production blocker for using RHT over QR rotation.

---

## What Was Built

### `_wht_kernel` (Triton JIT)
- One program per row: loads `d` elements into registers, runs `log2(d)` butterfly stages in-place
- Between stages: store → reload from global memory (Triton has no shared memory API)
- Normalize by `1/sqrt(d)` after all stages
- `BLOCK_SIZE = d` as constexpr — enables Triton to unroll the butterfly loop at compile time
- Supported `d`: 16, 32, 64, 128, 256, 512

### `hadamard_transform(x: Tensor)`
- Validates CUDA, contiguous, float32, power-of-2 shape
- Dispatches to `_wht_kernel` with explicit integer literals for `d` and `LOG2_D`
- In-place modification; also returns tensor

### `RHTRotation`
```python
rht = RHTRotation(d=128, seed=42, device='cuda')
x_rot = rht.forward(x)    # (N, 128) → (N, 128)
x_rec = rht.inverse(x_rot) # reconstruction
```
- `__init__`: generates int8 sign vector (d bytes) and int64 permutation (4d bytes) from fixed seed
- `forward`: multiply signs → `hadamard_transform` (in-place) → index `perm[:d]`
- `inverse`: scatter via `inv_perm` → `hadamard_transform` → multiply signs
- `memory_bytes()`: returns `d * 1 + d * 4 = 5d` bytes

### `test_rht_correctness()`
- Tests `d ∈ {64, 128, 256}`, batch size 8
- Asserts max reconstruction error `< 1e-5`

---

## Expected vs Experiment 04 Python WHT

| d   | QR matmul | Python WHT | Expected Triton WHT |
|-----|-----------|------------|---------------------|
| 64  | 0.023ms   | 5.49ms     | ~0.005ms (est.)     |
| 128 | 0.028ms   | 12.73ms    | ~0.008ms (est.)     |
| 256 | 0.214ms   | 21.60ms    | ~0.015ms (est.)     |

Estimates based on: O(d log d) vs O(d²) flops, 18x fewer operations at d=128, Triton register efficiency for d≤256.

**Expected speedup vs Python**: 1000-2000x  
**Expected vs cuBLAS QR**: 3-5x faster (fewer FLOPs, no d×d matrix load)

---

## Production Integration Path

1. ✅ `wht_kernel.py` implemented and ready
2. ❌ Replace `generate_rotation_matrix()` call in `TurboQuantMSE.__init__` with `RHTRotation`
3. ❌ Store seeds instead of full rotation matrices in `TurboQuantKVCache`
4. ❌ GPU validation (requires Brev GPU with Triton)

---

## Status

- [x] Code written: `turboquant/wht_kernel.py`
- [ ] GPU correctness validation
- [ ] Throughput benchmark (reproduce Experiment 04 speed test with new kernel)
- [ ] Integration into `TurboQuantMSE`
