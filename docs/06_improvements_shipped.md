# Experiment 06 — Code Improvements Shipped

**Date**: 2026-04-16  
**Hardware**: NVIDIA RTX A4000 (16.8 GB)  
**File changed**: `turboquant/kv_cache.py`

---

## Changes Made

### Patch 1: True 3-bit Value Quantization (`quantize_values`)

Added `bits=3` to `quantize_values()` using a true 3-bit packing scheme: 8 values per 3 bytes.

**Packing layout** (3 bytes encode 8 × 3-bit values a..h):
```
byte0 = a | (b<<3) | ((c&3)<<6)
byte1 = (c>>2) | (d<<1) | (e<<4) | ((f&1)<<7)
byte2 = (f>>1) | (g<<2) | (h<<5)
```

This is lossless (exact 3-bit representation, no extra bits wasted).

### Patch 2: 3-bit Unpack Support (`unpack_values`)

Added matching unpack for the 3-bit format in `unpack_values()`. Extracts 8 values from 3 bytes using the reverse bit operations.

### Patch 3: Default Group Size 32 → 16 (`TurboQuantKVCache`)

Changed `value_group_size: int = 32` to `value_group_size: int = 16` in `TurboQuantKVCache.__init__`.

Also changed the default in `quantize_values()` signature: `group_size: int = 16`.

---

## Verified Quality Improvements (GPU, D=128, N=64)

| config | cos_sim | vs old default |
|--------|---------|----------------|
| **Old default**: 2-bit gs=32 | 0.9317 | baseline |
| **New default**: 2-bit gs=16 | 0.9517 | **+0.020** |
| 3-bit gs=32 | 0.9866 | **+0.055** |
| 3-bit gs=16 | 0.9904 | **+0.059** |
| 4-bit gs=32 | 0.9971 | +0.065 |

---

## 3-bit Packing Correctness

All shapes verified correct:
- D=64: packed `(N, 24)` → unpacked `(N, 64)` ✓
- D=128: packed `(N, 48)` → unpacked `(N, 128)` ✓  
- D=256: packed `(N, 96)` → unpacked `(N, 256)` ✓

Formula: packed_len = d × 3 / 8 bytes.

---

## Memory Cost of 3-bit vs 2-bit Values (per token, D=128)

| format | packed bytes | meta bytes (gs=32) | total | vs FP16 |
|--------|-------------|-------------------|-------|---------|
| 2-bit  | 32          | 16                | 48    | 5.33x |
| 3-bit  | 48          | 16                | 64    | 4.00x |
| 4-bit  | 64          | 16                | 80    | 3.20x |
| FP16   | 256         | —                 | 256   | 1.00x |

3-bit costs 16 extra bytes per token vs 2-bit (33% more), but improves cosine by 0.055.

---

## What Still Needs to Be Done

| Task | Status |
|------|--------|
| 3-bit values in fused Kernel 3 | TODO — kernel hardcodes 2-bit value decode |
| 3-bit in `TurboQuantKVCache.prefill/append` | Auto-handled via `quantize_values` |
| Triton WHT kernel for Hadamard rotation | TODO |
| Update `value_bits=3` option in vLLM backend | TODO |
| End-to-end perplexity with patched TQ | TODO (needs model) |
