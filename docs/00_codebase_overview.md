# TurboQuant — Codebase Overview

**Date**: 2026-04-16  
**Repo**: https://github.com/0xSero/turboquant  
**Purpose**: Near-optimal KV cache compression for LLM inference (~4.4x compression, enabling 2x+ longer contexts on consumer GPUs)

---

## What It Does

TurboQuant compresses transformer KV caches with two goals:

1. **Keys**: Minimize per-score error using unbiased attention score estimation
2. **Values**: Minimize reconstruction error using group quantization

The result: at 3-bit keys + 2-bit values, VRAM usage drops ~4-5x while cosine similarity stays ≥ 0.93.

---

## Architecture

```
turboquant/
├── quantizer.py          Algorithm 1 (TurboQuantMSE) and Algorithm 2 (TurboQuantProd)
├── codebook.py           Lloyd-Max optimal scalar quantizer (Beta-distributed coordinates)
├── rotation.py           Random orthogonal matrix (QR) + QJL projection matrix
├── kv_cache.py           Drop-in KV cache: prefill/decode buffer, key+value quantization
├── store.py              Chunked compressed KV store with lazy flattening
├── capture.py            Modular KV capture hooks for attention layers
├── score.py              Hybrid attention: compressed history + exact recent buffer
├── triton_kernels.py     3 fused Triton kernels for GPU decode (no materialization)
├── vllm_attn_backend.py  vLLM shim (delegates to integration/vllm.py)
├── integration/
│   └── vllm.py           Full vLLM integration: modes, layer state, monkey-patches
└── codebooks/            Pre-computed codebooks (d64/128, b1-4; d576, b3)
```

---

## Core Algorithms

### Algorithm 1 — TurboQuantMSE (`quantizer.py:TurboQuantMSE`)

Minimizes reconstruction error for keys:

```
quantize:
  1. Compute ||x||₂ (store for rescaling)
  2. Normalize: x_unit = x / ||x||
  3. Rotate: y = x_unit @ Π^T  (Π = random orthogonal matrix)
  4. Quantize each coordinate via Lloyd-Max codebook (searchsorted)
  5. Bit-pack indices (1-4 bits/coord)

dequantize:
  1. Look up centroids for each index
  2. Rotate back: x_hat = y_hat @ Π
  3. Rescale: x_hat * ||x||
```

After rotation, each coordinate follows a Beta distribution on [-1,1]. The Lloyd-Max codebook is optimal for this distribution.

### Algorithm 2 — TurboQuantProd (`quantizer.py:TurboQuantProd`)

Minimizes inner-product estimation error (for attention scores). Two-stage:

```
Stage 1: MSE quantize at (b-1) bits → get residual r = x - x̃_mse
Stage 2: QJL on residual: sign(S·r) → 1 bit/coord, store ||r||₂

Score estimation:
  <q, x> ≈ <q, x̃_mse> + (√(π/2)/d) * ||r|| * <S^T·q, sign(S·r)>
  This is mathematically unbiased: E[estimate] = <q, x>
```

The QJL term uses the Johnson-Lindenstrauss lemma to encode the residual direction in 1 bit per dimension.

### Value Quantization (`kv_cache.py:quantize_values`)

Standard asymmetric group quantization:
- Group size: 32 (configurable)
- Per-group min-max scale + zero
- 2-bit (default) or 4-bit
- Bit-packed: 4 values/byte at 2-bit, 2 values/byte at 4-bit

---

## Memory Layout

For a single token with head_dim=128, key_bits=3, value_bits=2:

| Component | Storage | Bytes |
|-----------|---------|-------|
| MSE indices | 128 × 4-bit packed | 64 bytes |
| QJL signs | 128 × 1-bit packed | 16 bytes |
| Key norms (||x||, ||r||) | 2 × float16 | 4 bytes |
| Values | 128 × 2-bit packed | 32 bytes |
| Value scales+zeros | (128/32) × 2 × float16 | 16 bytes |
| **Total (TQ)** | | **132 bytes** |
| **Original (FP16)** | 2 × 128 × 2 | **512 bytes** |
| **Compression ratio** | | **3.88x** |

At head_dim=256 (Qwen MoE models): ~4.4x compression ratio.

---

## Fused Triton Kernels (`triton_kernels.py`)

Three kernels avoid materializing full FP16 key vectors during decode:

| Kernel | What it fuses | Input | Output |
|--------|--------------|-------|--------|
| `_turboquant_mse_score_kernel` | unpack + centroid lookup + rotate + dot | packed MSE indices | (BH, N) scores |
| `_turboquant_qjl_score_kernel` | unpack signs + dot with sketched query | packed QJL signs | (BH, N) QJL scores |
| `_turboquant_fused_decode_kernel` | full decode: keys + softmax + values | all compressed data | (BH, D) output |

Key insight for Kernel 1: instead of rotating key back (`y @ Π`), rotate query forward once (`q @ Π^T`), then score = `norms * Σ q_rot[j] * centroid[idx[j]]`. This avoids materializing the D-dim dequantized key vector entirely.

Kernel 3 implements flash-attention style online softmax over compressed KV — single pass, no intermediate full-precision materialization.

---

## vLLM Integration

**Modes**:
- `capture_only`: Capture KV into compressed store, always use flash output (safe baseline)
- `hybrid`: Use compressed history + exact recent for decode (production mode)
- `full_tq`: Future — TQ handles everything including prefill

**Hook installation** (`integration/vllm.py:install_hooks`):
- Detects flash vs MLA/GDN layer types
- Monkey-patches attention `forward()` methods
- Each layer gets a `LayerState` (owns `CompressedKVStore` + `KVCaptureEngine`)

**Buffer design** (`kv_cache.py:TurboQuantKVCache`):
- Recent 128 tokens kept unquantized for quality
- Older tokens compressed and stored in `CompressedKVStore`
- During flush: quantize oldest chunk, concatenate to compressed store

---

## Known Bottlenecks & Optimization Targets

| # | Issue | Impact | Where |
|---|-------|--------|-------|
| 1 | **2-bit value quantization** | cos_sim = 0.93 vs 0.997 at 4-bit | `kv_cache.py:quantize_values` |
| 2 | **Hybrid decode path** | All compressed tokens expanded to FP32 per step | `score.py`, `kv_cache.py:attend` |
| 3 | **Linear-attention layers (Mamba)** | Not compressed at all | `integration/vllm.py` |
| 4 | **Per-token norm storage** | norms stored FP32; could be FP16 or block-scaled | `quantizer.py`, `store.py` |
| 5 | **Prefill memory** | KV allocated at engine init, not zero-alloc | `vllm_attn_backend.py` |
| 6 | **QJL high per-sample variance** | Individual score estimates noisy despite unbiasedness | `quantizer.py:TurboQuantProd` |
| 7 | **Codebook is 1D scalar** | Independent per-coordinate quantization; could use vector quantization | `codebook.py`, `quantizer.py` |

---

## Environment Requirements

| Tier | Hardware | What runs |
|------|----------|-----------|
| CPU only | Any machine | Quantizer, codebook, rotation, packing, value quant — all unit logic |
| Single GPU | Any CUDA GPU | Triton kernels, quantizer on GPU, memory benchmarks |
| Multi-GPU (4x 3090+) | 4x RTX 3090 or better | Full vLLM integration, end-to-end proof benchmark |
| Optimal | 4x RTX 5090 or A100 cluster | Full Qwen3.5-27B benchmarks as in README |

**Current environment**: CPU only (Windows, Python 3.13, PyTorch 2.6 CPU build). Can run all CPU logic and design/validate optimization algorithms. Cannot run Triton kernels, vLLM integration, or VRAM benchmarks.

---

## Stack

- Python 3.10+, PyTorch 2.1+
- Triton 3.0+ (GPU kernels)
- vLLM 0.16+ (production integration)
- scipy (Lloyd-Max codebook computation)
- License: GPL-3.0
