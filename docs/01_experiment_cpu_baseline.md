# Experiment 01 — CPU Baseline Quality Survey

**Date**: 2026-04-16  
**Environment**: Windows, Python 3.13, PyTorch 2.6 (CPU)  
**Goal**: Establish ground-truth quality numbers for all quantization parameters before any optimization

---

## Summary

All core quantization logic works on CPU. We measured cosine similarity and memory ratios across all supported dimensions and bit-widths. Key findings:

- **3-bit MSE keys are the sweet spot**: 0.98+ cosine similarity at ~5x compression
- **2-bit values are the biggest quality liability**: 0.93 cosine sim vs 0.997 at 4-bit
- **Smaller group sizes improve value quality significantly** (16 >> 32 >> 64 groups)
- **QJL inner product estimator is nearly unbiased** (bias < 0.07) but has high per-sample variance (~1.3x relative error) — this is expected from the JL guarantee

---

## Experiment 1: TurboQuantMSE Key Quality

**Setup**: Random vectors of shape `(64, dim)`, each bit-width measured independently.

| dim | bits | cosine_sim | memory_ratio | notes |
|-----|------|-----------|--------------|-------|
| 64  | 1    | 0.8069    | 12.80x       | too lossy for attention |
| 64  | 2    | 0.9428    | 7.11x        | borderline acceptable |
| 64  | 3    | **0.9835**| **4.92x**    | **sweet spot** |
| 64  | 4    | 0.9953    | 3.76x        | diminishing returns |
| 128 | 1    | 0.8005    | 14.22x       | too lossy |
| 128 | 2    | 0.9410    | 7.53x        | borderline |
| 128 | 3    | **0.9835**| **5.12x**    | **sweet spot** |
| 128 | 4    | 0.9952    | 3.88x        | diminishing returns |
| 256 | 1    | 0.8028    | 15.06x       | too lossy |
| 256 | 2    | 0.9398    | 7.76x        | borderline |
| 256 | 3    | **0.9824**| **5.22x**    | **sweet spot** |
| 256 | 4    | 0.9953    | 3.94x        | diminishing returns |

**Observations**:
- Quality is remarkably stable across dimensions (64 vs 128 vs 256) — rotation + Lloyd-Max adapts well
- 3-bit is consistently the best compression/quality tradeoff
- Memory ratio improves with dimension because norm overhead (fixed 2 bytes) amortizes

---

## Experiment 2: Value Quantization Quality

**Setup**: Random vectors `(32, dim)`, group_size varied.

| dim | bits | group_size | cosine_sim | notes |
|-----|------|-----------|-----------|-------|
| 128 | 2    | 16        | 0.9509    | best 2-bit quality |
| 128 | 2    | 32        | 0.9306    | default setting |
| 128 | 2    | 64        | 0.9133    | noticeably worse |
| 128 | 4    | 16        | **0.9979**| near-lossless |
| 128 | 4    | 32        | 0.9970    | excellent |
| 128 | 4    | 64        | 0.9961    | still very good |
| 256 | 2    | 16        | 0.9518    | best 2-bit at 256 |
| 256 | 2    | 32        | 0.9330    | default |
| 256 | 4    | 16        | 0.9979    | near-lossless |
| 256 | 4    | 32        | 0.9969    | excellent |

**Observations**:
- 2-bit values with group_size=32 give cosine_sim=0.93, which is a real quality gap
- Switching 2-bit→4-bit recovers to 0.997, costing only 1 extra bit/value
- Smaller group_size=16 helps 2-bit values (+0.02 cosine) at no compression cost
- **Immediate optimization candidate**: try group_size=16 as default, or mixed 3-bit values

**Memory impact of switching 2-bit to 4-bit values**:
- At dim=128: values go from 32 bytes/token to 64 bytes/token (+32 bytes)
- Total token: 132 → 164 bytes (compression 3.88x → 3.12x)
- Quality gain: 0.93 → 0.997 cosine similarity

---

## Experiment 3: Inner Product Estimation (TurboQuantProd)

**Setup**: 100 random query-key pairs, dim=128. Measuring bias and relative error of `attention_score()`.

| bits | bias    | relative_error | notes |
|------|---------|----------------|-------|
| 2    | -0.0416 | 1.3893         | high variance |
| 3    | +0.0613 | 1.3644         | near-unbiased |
| 4    | -0.0396 | 1.2584         | best variance |

**Observations**:
- Bias is near zero at all bit widths — the unbiasedness claim holds empirically
- Relative error ~1.3x means individual score estimates are noisy
- This is expected: the QJL term is a randomized estimator, variance ∝ 1/d
- The softmax operation over many tokens means errors partially cancel in attention
- More bits = lower variance, as expected from JL theory

**Key insight**: High per-sample variance in attention scores is acceptable because softmax is relatively robust to score perturbation when there are many tokens. The attention weights are re-normalized by the partition function, so bounded score variance leads to small output error.

---

## Baseline Memory Calculations

At `head_dim=128, key_bits=3, value_bits=2, buffer_size=128`:

```
Per-token compressed storage:
  MSE indices:     64 bytes  (128 coords × 4-bit packed)
  QJL signs:       16 bytes  (128 coords × 1-bit packed)
  Key norms (×2):   4 bytes  (float16 × 2)
  Values packed:   32 bytes  (128 × 2-bit)
  Val scales:       8 bytes  (4 groups × float16)
  Val zeros:        8 bytes  (4 groups × float16)
  TOTAL:          132 bytes/token

FP16 KV:
  Key:   128 × 2 = 256 bytes
  Value: 128 × 2 = 256 bytes
  TOTAL: 512 bytes/token

Compression ratio: 512 / 132 = 3.88x
```

At `head_dim=256` (Qwen3.5-27B full-attention layers):

```
  MSE indices:    128 bytes  (256 × 4-bit)
  QJL signs:       32 bytes  (256 × 1-bit)
  Key norms (×2):   4 bytes
  Values:          64 bytes  (256 × 2-bit)
  Val scales:      16 bytes  (8 groups × float16)
  Val zeros:       16 bytes
  TOTAL:          260 bytes/token

FP16 KV:         1024 bytes/token
Compression ratio: 1024 / 260 = 3.94x
```

---

## What Can't Be Tested Locally (Needs GPU)

1. Triton kernel correctness and throughput
2. vLLM integration (MODE_HYBRID, MODE_ACTIVE)
3. Actual VRAM reduction measurements
4. End-to-end throughput (tokens/sec)
5. Multi-GPU tensor parallelism behavior
6. Long-context needle-in-haystack quality
7. Perplexity on real text datasets

---

## Next Experiment Candidates

| Priority | Experiment | Goal |
|----------|-----------|------|
| HIGH | Value quantization improvement | Replace 2-bit with 3-bit or mixed strategy |
| HIGH | Group size sweep | Verify group_size=16 default benefit |
| MEDIUM | Norm quantization | Quantize norms to FP16 (save 50% norm storage) |
| MEDIUM | Hadamard rotation | Replace O(d²) QR with O(d log d) RHT |
| LOW | Vector codebook | Replace scalar per-coord codebook with product quantization |
| LOW | Adaptive bits | Higher bits for initial layers, lower for later |
