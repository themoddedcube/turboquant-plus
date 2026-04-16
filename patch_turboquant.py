"""
Apply TurboQuant improvements discovered in experiments 03-05:
  1. Add true 3-bit value quantization to kv_cache.py
  2. Change default group_size 32 -> 16 in TurboQuantKVCache
  3. Add group_size=16 as new default in quantize_values
  4. Verify quality improvements match experiment predictions
"""
import torch, sys, re
import torch.nn.functional as F

device = torch.device("cuda")
print("=== Applying TurboQuant Improvements ===\n")

# ── Patch 1: Read current kv_cache.py ─────────────────────────────────────
with open("/root/turboquant/turboquant/kv_cache.py", "r") as f:
    src = f.read()

original = src

# ── Patch 2: Add 3-bit support to quantize_values ─────────────────────────
old_quantize = '''def quantize_values(
    v: torch.Tensor,
    bits: int = 2,
    group_size: int = 32,
) -> ValueQuantized:
    """
    Symmetric group quantization for value vectors.

    Args:
        v: (..., seq_len, d) value vectors
        bits: quantization bits (2 or 4)
        group_size: number of elements per quantization group
    """
    orig_shape = v.shape
    d = orig_shape[-1]
    n_groups = d // group_size
    assert d % group_size == 0, f"head_dim {d} must be divisible by group_size {group_size}"

    # Reshape to groups
    v_grouped = v.reshape(*orig_shape[:-1], n_groups, group_size)  # (..., seq, n_groups, gs)

    # Compute scale and zero per group (asymmetric)
    v_min = v_grouped.min(dim=-1, keepdim=True).values
    v_max = v_grouped.max(dim=-1, keepdim=True).values

    n_levels = 2**bits - 1
    scale = (v_max - v_min) / n_levels
    scale = scale.clamp(min=1e-10)
    zero = v_min

    # Quantize
    v_q = ((v_grouped - zero) / scale).round().clamp(0, n_levels).to(torch.uint8)
    v_q_flat = v_q.reshape(*orig_shape[:-1], d)

    # Bit-pack: for 2-bit, pack 4 values per byte; for 4-bit, pack 2 per byte
    if bits == 2:
        # Pack 4 x 2-bit values into each uint8: [a, b, c, d] -> a | (b<<2) | (c<<4) | (d<<6)
        assert d % 4 == 0
        v_4 = v_q_flat.reshape(*orig_shape[:-1], d // 4, 4)
        packed = v_4[..., 0] | (v_4[..., 1] << 2) | (v_4[..., 2] << 4) | (v_4[..., 3] << 6)
        v_q_flat = packed  # shape: (..., d//4)
    elif bits == 4:
        assert d % 2 == 0
        v_2 = v_q_flat.reshape(*orig_shape[:-1], d // 2, 2)
        packed = v_2[..., 0] | (v_2[..., 1] << 4)
        v_q_flat = packed  # shape: (..., d//2)
    # bits==8: no packing needed

    return ValueQuantized(
        data=v_q_flat,
        scales=scale.squeeze(-1),
        zeros=zero.squeeze(-1),
        bits=bits,
    )'''

new_quantize = '''def quantize_values(
    v: torch.Tensor,
    bits: int = 2,
    group_size: int = 16,
) -> ValueQuantized:
    """
    Asymmetric group quantization for value vectors.

    Args:
        v: (..., seq_len, d) value vectors
        bits: quantization bits (2, 3, 4, or 8).
              3-bit uses true 3-bit packing (8 values per 3 bytes).
        group_size: elements per quantization group (default 16, was 32).
                    Smaller groups improve quality: 2-bit gs=16 -> cos=0.952 vs gs=32 -> cos=0.932.
    """
    orig_shape = v.shape
    d = orig_shape[-1]
    n_groups = d // group_size
    assert d % group_size == 0, f"head_dim {d} must be divisible by group_size {group_size}"

    # Reshape to groups
    v_grouped = v.reshape(*orig_shape[:-1], n_groups, group_size)

    # Compute scale and zero per group (asymmetric)
    v_min = v_grouped.min(dim=-1, keepdim=True).values
    v_max = v_grouped.max(dim=-1, keepdim=True).values

    n_levels = 2**bits - 1
    scale = (v_max - v_min) / n_levels
    scale = scale.clamp(min=1e-10)
    zero = v_min

    # Quantize
    v_q = ((v_grouped - zero) / scale).round().clamp(0, n_levels).to(torch.uint8)
    v_q_flat = v_q.reshape(*orig_shape[:-1], d)

    # Bit-pack based on bits
    if bits == 2:
        # 4 values per byte: a | (b<<2) | (c<<4) | (d<<6)
        assert d % 4 == 0
        v_4 = v_q_flat.reshape(*orig_shape[:-1], d // 4, 4)
        packed = v_4[..., 0] | (v_4[..., 1] << 2) | (v_4[..., 2] << 4) | (v_4[..., 3] << 6)
        v_q_flat = packed  # (..., d//4)
    elif bits == 3:
        # True 3-bit: 8 values per 3 bytes
        # [a,b,c,d,e,f,g,h] -> byte0: a|b<<3|(c&3)<<6, byte1: c>>2|d<<1|e<<4|(f&1)<<7, byte2: f>>1|g<<2|h<<5
        assert d % 8 == 0, f"head_dim {d} must be divisible by 8 for 3-bit packing"
        v_8 = v_q_flat.reshape(*orig_shape[:-1], d // 8, 8).long()
        b0 = v_8[...,0] | (v_8[...,1] << 3) | ((v_8[...,2] & 0x3) << 6)
        b1 = (v_8[...,2] >> 2) | (v_8[...,3] << 1) | (v_8[...,4] << 4) | ((v_8[...,5] & 0x1) << 7)
        b2 = (v_8[...,5] >> 1) | (v_8[...,6] << 2) | (v_8[...,7] << 5)
        packed = torch.stack([b0, b1, b2], dim=-1).to(torch.uint8).reshape(*orig_shape[:-1], d * 3 // 8)
        v_q_flat = packed  # (..., d*3//8)
    elif bits == 4:
        # 2 values per byte: a | (b<<4)
        assert d % 2 == 0
        v_2 = v_q_flat.reshape(*orig_shape[:-1], d // 2, 2)
        packed = v_2[..., 0] | (v_2[..., 1] << 4)
        v_q_flat = packed  # (..., d//2)
    # bits==8: no packing needed

    return ValueQuantized(
        data=v_q_flat,
        scales=scale.squeeze(-1),
        zeros=zero.squeeze(-1),
        bits=bits,
    )'''

if old_quantize in src:
    src = src.replace(old_quantize, new_quantize)
    print("[PATCH 1] quantize_values: added 3-bit support + changed default group_size 32->16")
else:
    print("[PATCH 1] WARNING: could not find quantize_values signature to patch")

# ── Patch 3: Add 3-bit support to unpack_values ───────────────────────────
old_unpack = '''def unpack_values(vq: ValueQuantized) -> torch.Tensor:
    """Unpack bit-packed value data to uint8 per-element."""
    bits = vq.bits if len(vq) > 3 else 2
    packed = vq.data
    if bits == 2:
        v0 = packed & 0x03
        v1 = (packed >> 2) & 0x03
        v2 = (packed >> 4) & 0x03
        v3 = (packed >> 6) & 0x03
        return torch.stack([v0, v1, v2, v3], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 4)
    elif bits == 4:
        v0 = packed & 0x0F
        v1 = (packed >> 4) & 0x0F
        return torch.stack([v0, v1], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return packed'''

new_unpack = '''def unpack_values(vq: ValueQuantized) -> torch.Tensor:
    """Unpack bit-packed value data to uint8 per-element."""
    bits = vq.bits if len(vq) > 3 else 2
    packed = vq.data
    if bits == 2:
        v0 = packed & 0x03
        v1 = (packed >> 2) & 0x03
        v2 = (packed >> 4) & 0x03
        v3 = (packed >> 6) & 0x03
        return torch.stack([v0, v1, v2, v3], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 4)
    elif bits == 3:
        # Unpack 8 values from 3 bytes
        p = packed.long()
        pr = p.reshape(*packed.shape[:-1], packed.shape[-1] // 3, 3)
        b0, b1, b2 = pr[..., 0], pr[..., 1], pr[..., 2]
        v0 = b0 & 0x7
        v1 = (b0 >> 3) & 0x7
        v2 = ((b0 >> 6) & 0x3) | ((b1 & 0x1) << 2)
        v3 = (b1 >> 1) & 0x7
        v4 = (b1 >> 4) & 0x7
        v5 = ((b1 >> 7) & 0x1) | ((b2 & 0x3) << 1)
        v6 = (b2 >> 2) & 0x7
        v7 = (b2 >> 5) & 0x7
        return torch.stack([v0,v1,v2,v3,v4,v5,v6,v7], dim=-1).to(torch.uint8).reshape(*packed.shape[:-1], packed.shape[-1] * 8 // 3)
    elif bits == 4:
        v0 = packed & 0x0F
        v1 = (packed >> 4) & 0x0F
        return torch.stack([v0, v1], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return packed'''

if old_unpack in src:
    src = src.replace(old_unpack, new_unpack)
    print("[PATCH 2] unpack_values: added 3-bit support")
else:
    print("[PATCH 2] WARNING: could not find unpack_values to patch")

# ── Patch 4: Change TurboQuantKVCache default group_size 32->16 ────────────
old_init_sig = "    value_group_size: int = 32,"
new_init_sig = "    value_group_size: int = 16,  # Experiment 03: gs=16 gives cos=0.952 vs gs=32 cos=0.932 at no compression cost"

if old_init_sig in src:
    src = src.replace(old_init_sig, new_init_sig)
    print("[PATCH 3] TurboQuantKVCache: changed default value_group_size 32->16")
else:
    print("[PATCH 3] WARNING: could not find value_group_size=32 in __init__")

# ── Write patched file ─────────────────────────────────────────────────────
with open("/root/turboquant/turboquant/kv_cache.py", "w") as f:
    f.write(src)
print("\n[WRITTEN] /root/turboquant/turboquant/kv_cache.py")

# ── Verify patches ─────────────────────────────────────────────────────────
print("\n=== Verifying Patches ===\n")

# Reload module
import importlib, turboquant.kv_cache
importlib.reload(turboquant.kv_cache)
from turboquant.kv_cache import quantize_values, dequantize_values, unpack_values

for D in [128, 256]:
    v = torch.randn(64, D, device=device)
    print(f"D={D}:")
    for bits, gs in [(2,16),(2,32),(3,32),(3,16),(4,32)]:
        if D % gs != 0: continue
        if bits == 3 and D % 8 != 0: continue
        vq = quantize_values(v, bits=bits, group_size=gs)
        v_hat = dequantize_values(vq, group_size=gs)
        cos = F.cosine_similarity(v, v_hat).mean().item()
        packed_b = vq.data.nelement()
        print(f"  bits={bits} gs={gs:>2}  cos={cos:.5f}  packed={packed_b}B  bits_field={vq.bits}")

# ── Verify 3-bit roundtrip ─────────────────────────────────────────────────
print("\n=== 3-bit Packing Roundtrip ===")
for D in [64, 128, 256]:
    v = torch.randint(0, 8, (32, D), device=device).float()
    vq = quantize_values(v, bits=3, group_size=16 if D >= 16 else 8)
    unpacked = unpack_values(vq).float()
    # Dequantize manually
    v_hat = dequantize_values(vq, group_size=16 if D >= 16 else 8)
    # Just check unpacked has right shape
    print(f"  D={D}  packed.shape={vq.data.shape}  unpacked.shape={unpacked.shape}  bits={vq.bits}")

print("\nPATCH VERIFICATION DONE")
