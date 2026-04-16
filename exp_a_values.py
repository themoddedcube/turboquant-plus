"""
Experiment A: Better value quantization.
Current default: 2-bit, group_size=32 -> cos=0.93
Test: 3-bit packing, smaller groups, SmoothQuant-style per-channel scaling.
"""
import torch, time, json
import torch.nn.functional as F
device = torch.device("cuda")

results = {}

from turboquant.kv_cache import quantize_values, dequantize_values

print("=== Value Quantization Quality Survey ===\n")
print(f"{'config':<25} {'cos':>7} {'B/tok':>6} {'ratio':>8}")

for D in [128, 256]:
    v = torch.randn(256, D, device=device)
    fp16_bytes_total = D * 2 * 256

    for bits in [2, 3, 4, 8]:
        for gs in [8, 16, 32, 64]:
            if gs > D or D % gs != 0: continue
            try:
                vq = quantize_values(v, bits=bits, group_size=gs)
                v_hat = dequantize_values(vq, group_size=gs)
                cos = F.cosine_similarity(v, v_hat).mean().item()
                packed_bytes = vq.data.nelement()
                scale_bytes = vq.scales.nelement() * 2
                zero_bytes = vq.zeros.nelement() * 2
                total = packed_bytes + scale_bytes + zero_bytes
                ratio = fp16_bytes_total / total
                label = f"D={D} b={bits} gs={gs}"
                print(f"  {label:<23} {cos:>7.5f} {total//256:>6} {ratio:>7.2f}x")
                results[f"D{D}_b{bits}_g{gs}"] = {"cos": cos, "bytes_per_tok": total//256, "ratio": ratio}
            except Exception as e:
                pass

# ── SmoothQuant-style per-channel normalization ──────────────────────────
print("\n=== SmoothQuant Channel Equalization + 2-bit ===\n")

for D in [128, 256]:
    v = torch.randn(256, D, device=device)
    # Calibration: per-channel max abs
    channel_scales = v.abs().max(dim=0).values.clamp(min=1e-6)
    v_smooth = v / channel_scales.unsqueeze(0)

    for bits in [2, 3, 4]:
        for gs in [16, 32]:
            if D % gs != 0: continue
            try:
                vq = quantize_values(v_smooth, bits=bits, group_size=gs)
                v_hat_smooth = dequantize_values(vq, group_size=gs)
                v_hat = v_hat_smooth * channel_scales.unsqueeze(0)
                cos_smooth = F.cosine_similarity(v, v_hat).mean().item()

                vq_base = quantize_values(v, bits=bits, group_size=gs)
                v_hat_base = dequantize_values(vq_base, group_size=gs)
                cos_base = F.cosine_similarity(v, v_hat_base).mean().item()

                diff = cos_smooth - cos_base
                label = f"D={D} b={bits} gs={gs}"
                print(f"  {label:<23}  base={cos_base:.5f}  smooth={cos_smooth:.5f}  diff={diff:+.5f}")
            except Exception as e:
                print(f"  D={D} b={bits} gs={gs}: error {e}")

# ── True 3-bit pack: 8 values in 3 bytes ─────────────────────────────────
print("\n=== True 3-bit Packing (8 values per 3 bytes) ===\n")

def quantize_3bit(v, group_size=32):
    d = v.shape[-1]
    batch = v.shape[:-1]
    n_groups = d // group_size
    vg = v.reshape(*batch, n_groups, group_size)
    vmin = vg.min(-1, keepdim=True).values
    vmax = vg.max(-1, keepdim=True).values
    scale = (vmax - vmin) / 7.0
    scale = scale.clamp(min=1e-10)
    vq = ((vg - vmin) / scale).round().clamp(0, 7).to(torch.uint8)
    vq_flat = vq.reshape(*batch, d)
    # Pack 8x 3-bit values into 3 bytes
    # [a b c d e f g h] -> byte0: a|b<<3|c[:2]<<6, byte1: c[2]|d<<1|e<<4|f[:1]<<7, byte2: f[1:]|g<<2|h<<5
    assert d % 8 == 0
    vq_r = vq_flat.reshape(*batch, d // 8, 8).long()
    b0 = (vq_r[...,0]) | (vq_r[...,1] << 3) | ((vq_r[...,2] & 0x3) << 6)
    b1 = (vq_r[...,2] >> 2) | (vq_r[...,3] << 1) | (vq_r[...,4] << 4) | ((vq_r[...,5] & 0x1) << 7)
    b2 = (vq_r[...,5] >> 1) | (vq_r[...,6] << 2) | (vq_r[...,7] << 5)
    packed = torch.stack([b0, b1, b2], dim=-1).to(torch.uint8).reshape(*batch, d * 3 // 8)
    return packed, scale.squeeze(-1), vmin.squeeze(-1)

def dequantize_3bit(packed, scale, vmin, group_size=32):
    batch = packed.shape[:-1]
    n_packed = packed.shape[-1]
    d = n_packed * 8 // 3
    pr = packed.long().reshape(*batch, n_packed // 3, 3)
    b0, b1, b2 = pr[...,0], pr[...,1], pr[...,2]
    v0 = b0 & 7
    v1 = (b0 >> 3) & 7
    v2 = ((b0 >> 6) & 3) | ((b1 & 1) << 2)
    v3 = (b1 >> 1) & 7
    v4 = (b1 >> 4) & 7
    v5 = ((b1 >> 7) & 1) | ((b2 & 3) << 1)
    v6 = (b2 >> 2) & 7
    v7 = (b2 >> 5) & 7
    vq = torch.stack([v0,v1,v2,v3,v4,v5,v6,v7], dim=-1).reshape(*batch, d).float()
    n_groups = d // group_size
    vq = vq.reshape(*batch, n_groups, group_size)
    v = vq * scale.unsqueeze(-1) + vmin.unsqueeze(-1)
    return v.reshape(*batch, d)

for D in [128, 256]:
    v = torch.randn(256, D, device=device)
    fp16_b = D * 2
    for gs in [16, 32, 64]:
        if D % gs != 0 or D % 8 != 0: continue
        try:
            packed, scale, vmin = quantize_3bit(v, gs)
            v_hat = dequantize_3bit(packed, scale, vmin, gs)
            cos = F.cosine_similarity(v, v_hat).mean().item()
            # Check roundtrip
            n_groups = D // gs
            bytes_tok = packed.shape[-1] + n_groups * 2 * 4  # packed + scale+zero float32
            ratio = fp16_b / (bytes_tok / 256 * 256 / 256)
            print(f"  True-3bit D={D} gs={gs}  cos={cos:.5f}  {packed.shape[-1]//256}B packed + {n_groups*8}B meta = ~{bytes_tok//256}B/tok  ratio~{fp16_b*256/bytes_tok:.2f}x")
        except Exception as e:
            print(f"  D={D} gs={gs}: {e}")

print("\nEXP-A DONE")
with open("/tmp/exp_a_results.json", "w") as f:
    json.dump(results, f)
