"""
Experiment B: Randomized Hadamard Transform (RHT) vs QR rotation.
Goal: replace the O(d^2) stored matrix with O(d log d) butterfly RHT.
Benefits: no d*d matrix stored, faster rotation, same quality.
"""
import torch, time, math
import torch.nn.functional as F
device = torch.device("cuda")
from turboquant.quantizer import TurboQuantMSE
from turboquant.rotation import generate_rotation_matrix
from turboquant.codebook import get_codebook_tensors

def next_pow2(n):
    return 1 << (n - 1).bit_length()

def hadamard_transform(x):
    """Fast Walsh-Hadamard Transform. x: (..., d) where d is power of 2."""
    d = x.shape[-1]
    result = x.clone().float()
    step = 1
    while step < d:
        for i in range(0, d, step * 2):
            a = result[..., i:i+step].clone()
            b = result[..., i+step:i+2*step].clone()
            result[..., i:i+step] = a + b
            result[..., i+step:i+2*step] = a - b
        step *= 2
    return result / math.sqrt(d)


class RHTRotation(torch.nn.Module):
    """Randomized Hadamard Transform. Stores only d signs + d ints (vs d*d floats)."""
    def __init__(self, d, device, seed=42):
        super().__init__()
        self.d = d
        self.d_pad = next_pow2(d)
        rng = torch.Generator()
        rng.manual_seed(seed)
        signs = (torch.randint(0, 2, (self.d_pad,), generator=rng).float() * 2 - 1)
        perm = torch.randperm(self.d_pad, generator=rng)
        self.register_buffer("signs", signs.to(device))
        self.register_buffer("perm", perm.to(device))
        self.register_buffer("inv_perm", torch.argsort(perm).to(device))

    def forward(self, x):
        x_f = x.float()
        if self.d < self.d_pad:
            x_f = F.pad(x_f, (0, self.d_pad - self.d))
        x_f = x_f * self.signs
        x_f = hadamard_transform(x_f)
        return x_f[..., self.perm][..., :self.d]

    def inverse(self, y):
        y_f = y.float()
        padded = torch.zeros(*y_f.shape[:-1], self.d_pad, device=y.device, dtype=torch.float32)
        padded[..., :self.d] = y_f
        y_perm = padded[..., self.inv_perm]
        y_had = hadamard_transform(y_perm)
        return (y_had * self.signs)[..., :self.d]


print("=== RHT Rotation: Correctness ===\n")
for d in [64, 128, 256]:
    rht = RHTRotation(d, device)
    x = torch.randn(64, d, device=device)
    y = rht(x)
    x_rec = rht.inverse(y)
    err = (x - x_rec).abs().max().item()
    print(f"  d={d:3d}  roundtrip_max_err={err:.2e}  {'PASS' if err < 1e-4 else 'FAIL'}")

print("\n=== Quality: QR vs RHT (cosine similarity after quantize+dequantize) ===\n")
print(f"  {'d':>4} {'bits':>4}  {'QR_cos':>8}  {'RHT_cos':>9}  {'diff':>7}")

for d in [64, 128, 256]:
    for bits in [2, 3, 4]:
        tq = TurboQuantMSE(dim=d, bits=bits, device=device)
        x = torch.randn(512, d, device=device)

        # QR path
        x_hat_qr = tq(x)
        cos_qr = F.cosine_similarity(x, x_hat_qr).mean().item()

        # RHT path
        rht = RHTRotation(d, device, seed=42)
        centroids, boundaries = get_codebook_tensors(d, bits, device)
        decision = boundaries[1:-1].contiguous()
        norms = x.norm(dim=-1, keepdim=True)
        x_unit = x / (norms + 1e-10)
        y = rht(x_unit)
        indices = torch.searchsorted(decision, y.contiguous())
        y_hat = centroids[indices]
        x_hat_rht = rht.inverse(y_hat) * norms
        cos_rht = F.cosine_similarity(x, x_hat_rht).mean().item()

        diff = cos_rht - cos_qr
        tag = "better" if diff > 0.001 else ("same" if abs(diff) < 0.003 else "worse")
        print(f"  {d:>4} {bits:>4}  {cos_qr:>8.5f}  {cos_rht:>9.5f}  {diff:>+7.5f}  {tag}")

print("\n=== Speed: QR matmul vs RHT butterfly ===\n")
print(f"  {'d':>4} {'N':>6}  {'QR_ms':>8}  {'RHT_ms':>9}  {'speedup':>8}")

for d in [64, 128, 256]:
    Pi = generate_rotation_matrix(d, device)
    rht = RHTRotation(d, device)
    for N in [1024, 4096, 16384]:
        x = torch.randn(N, d, device=device)
        REPS = 200

        for _ in range(5): torch.matmul(x, Pi.T)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS): torch.matmul(x, Pi.T)
        torch.cuda.synchronize()
        t_qr = (time.perf_counter() - t0) / REPS * 1000

        for _ in range(5): rht(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS): rht(x)
        torch.cuda.synchronize()
        t_rht = (time.perf_counter() - t0) / REPS * 1000

        print(f"  {d:>4} {N:>6}  {t_qr:>8.4f}  {t_rht:>9.4f}  {t_qr/t_rht:>7.2f}x")

print("\n=== Memory Footprint ===\n")
for d in [64, 128, 256]:
    d_pad = next_pow2(d)
    qr_kb = d * d * 4 / 1024
    rht_b = d_pad * 4 + d_pad * 4   # signs float32 + perm int32
    rht_opt_b = d_pad * 1 + d_pad * 4  # signs int8 + perm int32
    print(f"  d={d:3d}: QR={qr_kb:.1f}KB  RHT_float={rht_b}B  RHT_int8signs={rht_opt_b}B  ratio={d*d*4/rht_opt_b:.1f}x smaller")

print("\nEXP-B DONE")
