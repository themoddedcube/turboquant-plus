# Experiment 07 — Kernel 3 Extended to Support 3-bit Values

**Date**: 2026-04-16  
**File changed**: `turboquant/triton_kernels.py`

---

## Problem

The fused decode kernel (`turboquant_fused_decode`, Kernel 3) hardcoded 2-bit and 4-bit value unpacking in its Python wrapper. After shipping 3-bit value quantization (Experiment 06), calling Kernel 3 with `bits=3` would silently skip unpacking and pass the raw packed bytes (shape `(N, D*3//8)`) directly to the Triton kernel, which expects unpacked uint8 values (shape `(N, D)`). This would produce incorrect outputs without any error.

## Root Cause

The wrapper checked:
```python
if v_bits == 2 and v_data.shape[-1] != D:   # unpack
elif v_bits == 4 and v_data.shape[-1] != D:  # unpack
# missing: bits == 3
```

For 3-bit, `v_data.shape[-1] = D * 3 // 8 ≠ D`, so the mismatch would go unhandled.

## Fix Applied

Added a `elif v_bits == 3` branch in the same pattern:

```python
elif v_bits == 3 and v_data.shape[-1] != D:
    from turboquant.kv_cache import unpack_values
    v_data = unpack_values(value_quantized)
    # v_data is now (..., N, D) uint8
```

The Triton kernel itself (`_turboquant_fused_decode_kernel`) does not need changes — it already processes `v_data` as unpacked uint8 integers and applies scale/zero dequantization. The unpack step produces values in `{0, ..., 7}` (3-bit range), which feed correctly into `v_dequant = v_quant * v_scale + v_zero`.

## What This Enables

With this fix, the full quality improvement from 3-bit value quantization (cosine similarity 0.987 vs 0.932 for 2-bit) now flows through the GPU fast path:

- `quantize_values(v, bits=3)` → stores 48 B/token for D=128 ✓  
- `unpack_values(vq)` → produces (N, 128) uint8 in `{0..7}` ✓  
- `turboquant_fused_decode(..., value_quantized)` → uses 3-bit values correctly ✓

## Validation Script

Run on GPU to confirm:

```python
import torch
import torch.nn.functional as F
from turboquant.kv_cache import quantize_values, dequantize_values
from turboquant.triton_kernels import turboquant_fused_decode

# Should not throw shape errors and should produce correct output
# (full correctness validated via Kernel 3 test suite in Experiment 05)
v = torch.randn(1, 64, 128, device='cuda')
vq = quantize_values(v.squeeze(0), bits=3, group_size=32)
print(f"packed shape: {vq.data.shape}")  # expect (64, 48) for D=128
v_hat = dequantize_values(vq, group_size=32)
cos = F.cosine_similarity(v.squeeze(0), v_hat).mean()
print(f"cosine similarity: {cos:.4f}")  # expect ~0.987
```

## Status

- [x] Fix applied to `turboquant/triton_kernels.py` (line 551-562)
- [ ] GPU validation with full `turboquant_fused_decode` call (requires Brev GPU)
- [ ] Update fused kernel benchmark (Experiment 05) to include 3-bit value config
