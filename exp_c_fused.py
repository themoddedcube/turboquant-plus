"""
Experiment C: Kernel 3 (turboquant_fused_decode) benchmark + correctness.
Three-way comparison:
  1. Full FP16 PyTorch decode (baseline)
  2. TQ scores (Triton) + PyTorch softmax + dequant values (hybrid)
  3. Fully fused: TQ keys + softmax + values in single kernel pass
"""
import torch, time, math
import torch.nn.functional as F
device = torch.device("cuda")
from turboquant.quantizer import TurboQuantProd
from turboquant.kv_cache import quantize_values, dequantize_values
from turboquant.triton_kernels import turboquant_attention_score, turboquant_fused_decode


def pytorch_decode(query, keys_fp, values_fp, sm_scale):
    scores = torch.matmul(query.float(), keys_fp.float().transpose(-2,-1)) * sm_scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, values_fp.float())


def hybrid_decode(query, pq_r, vq, q_prod, sm_scale, gs=32):
    scores = turboquant_attention_score(
        query, pq_r,
        q_prod.mse_quantizer.Pi, q_prod.S,
        q_prod.mse_quantizer.centroids, pq_r.mse_bits, q_prod.qjl_scale)
    weights = torch.softmax(scores * sm_scale, dim=-1)
    v_hat = dequantize_values(vq, group_size=gs)
    return torch.bmm(weights.unsqueeze(1), v_hat)


print("=== Kernel 3: turboquant_fused_decode ===\n")
print("── Correctness ──\n")

for (BH, N, D) in [(8, 256, 128), (16, 1024, 128), (16, 1024, 256), (32, 4096, 128)]:
    q_prod = TurboQuantProd(dim=D, bits=3, device=device)
    keys   = torch.randn(BH * N, D, device=device)
    values = torch.randn(BH, N, D, device=device, dtype=torch.float16)
    query  = torch.randn(BH, D, device=device)
    sm_scale = 1.0 / math.sqrt(D)

    pq = q_prod.quantize(keys)
    pq_r = type('PQ',(),{
        'mse_indices': pq.mse_indices.reshape(BH,N,-1),
        'qjl_signs':   pq.qjl_signs.reshape(BH,N,-1),
        'residual_norms': pq.residual_norms.reshape(BH,N),
        'norms':       pq.norms.reshape(BH,N),
        'mse_bits':    pq.mse_bits})()
    vq = quantize_values(values, bits=2, group_size=32)

    # Reference: hybrid
    out_ref = hybrid_decode(query.unsqueeze(1), pq_r, vq, q_prod, sm_scale).squeeze(1)

    try:
        out_fused = turboquant_fused_decode(
            query, pq_r, vq,
            q_prod.mse_quantizer.Pi, q_prod.S,
            q_prod.mse_quantizer.centroids,
            pq.mse_bits, q_prod.qjl_scale, sm_scale, group_size=32)
        max_err = (out_fused.float() - out_ref.float()).abs().max().item()
        cos = F.cosine_similarity(out_fused.float(), out_ref.float(), dim=-1).mean().item()
        status = "PASS" if max_err < 0.05 else ("~OK" if max_err < 0.5 else "FAIL")
        print(f"  BH={BH:2d} N={N:5d} D={D}  max_err={max_err:.5f}  cos={cos:.6f}  {status}")
    except Exception as e:
        print(f"  BH={BH:2d} N={N:5d} D={D}  FUSED FAILED: {e}")

print("\n── Throughput: 3-way comparison (BH=32, D=128) ──\n")
print(f"  {'N':>6}  {'FP16_ms':>8}  {'Hybrid_ms':>10}  {'Fused_ms':>9}  {'Hybrid/FP16':>12}  {'Fused/FP16':>11}")

REPS = 100
for N in [256, 1024, 4096, 16384]:
    BH, D, GS = 32, 128, 32
    q_prod = TurboQuantProd(dim=D, bits=3, device=device)
    keys_fp = torch.randn(BH, N, D, device=device, dtype=torch.float16)
    vals_fp = torch.randn(BH, N, D, device=device, dtype=torch.float16)
    query   = torch.randn(BH, D, device=device, dtype=torch.float16)
    sm_scale = 1.0 / math.sqrt(D)

    pq = q_prod.quantize(keys_fp.reshape(-1,D).float())
    pq_r = type('PQ',(),{
        'mse_indices': pq.mse_indices.reshape(BH,N,-1),
        'qjl_signs':   pq.qjl_signs.reshape(BH,N,-1),
        'residual_norms': pq.residual_norms.reshape(BH,N),
        'norms':       pq.norms.reshape(BH,N),
        'mse_bits':    pq.mse_bits})()
    vq = quantize_values(vals_fp, bits=2, group_size=GS)

    def bench(fn):
        for _ in range(5): fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / REPS * 1000

    t_fp = bench(lambda: pytorch_decode(query.unsqueeze(1), keys_fp, vals_fp, sm_scale))
    t_hy = bench(lambda: hybrid_decode(query.unsqueeze(1), pq_r, vq, q_prod, sm_scale, GS))

    try:
        t_fu = bench(lambda: turboquant_fused_decode(
            query, pq_r, vq,
            q_prod.mse_quantizer.Pi, q_prod.S,
            q_prod.mse_quantizer.centroids,
            pq.mse_bits, q_prod.qjl_scale, sm_scale, group_size=GS))
        fused_col = f"{t_fu:>9.3f}  {t_fp/t_fu:>11.1f}x"
    except Exception as e:
        fused_col = f"  FAILED ({e})"

    print(f"  {N:>6}  {t_fp:>8.3f}  {t_hy:>10.3f}  {fused_col}  {t_fp/t_hy:>11.1f}x")

print("\n── Context capacity on RTX A4000 (16.8 GB) ──\n")
print(f"  {'D':>4} {'k_bits':>6} {'v_bits':>6}  {'TQ_B/tok':>9}  {'FP16_B/tok':>11}  {'TQ_max_ctx':>11}  {'FP16_max_ctx':>13}  {'extension':>10}")
for D in [128, 256]:
    for kb, vb in [(3,2),(3,4),(4,2),(4,4)]:
        n_heads = 32
        # TQ compressed bytes per token per head
        mse_packed_b = D * 4 // 8   # 4-bit packed (3-bit rounds to 4)
        qjl_b = D // 8
        norm_b = 2 * 2               # 2x float16
        val_packed_b = D * vb // 8
        n_groups = D // 32
        val_meta_b = n_groups * 2 * 2  # scale+zero float16
        tq_per_tok_head = mse_packed_b + qjl_b + norm_b + val_packed_b + val_meta_b
        fp16_per_tok_head = D * 2 * 2  # K+V float16

        vram_bytes = 16.8e9 * 0.6  # 60% for KV cache
        tq_max = int(vram_bytes / (tq_per_tok_head * n_heads))
        fp16_max = int(vram_bytes / (fp16_per_tok_head * n_heads))
        ext = tq_max / fp16_max

        print(f"  {D:>4} {kb:>6} {vb:>6}  {tq_per_tok_head:>9}  {fp16_per_tok_head:>11}  {tq_max//1000:>9}k  {fp16_max//1000:>11}k  {ext:>9.2f}x")

print("\nEXP-C DONE")
