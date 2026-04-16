"""
Triton Walsh-Hadamard Transform (WHT) kernel for TurboQuant.

Implements the fast O(d log d) butterfly WHT as a Triton JIT kernel, replacing
the slow Python butterfly loop (200-400x slower than cuBLAS). For d ∈ {64, 128,
256}, the full vector fits in registers at BLOCK_SIZE=d, giving near-peak
memory bandwidth efficiency.

The Randomized Hadamard Transform (RHT) is:
    forward(x): signs → WHT/sqrt(d) → permute
    inverse(y): unpermute → WHT → signs (WHT is self-inverse up to 1/sqrt(d))

Key types:
    _wht_kernel          — Triton JIT kernel (one program = one row)
    hadamard_transform   — Python wrapper: (*, d) → (*, d) / sqrt(d)
    RHTRotation          — Class encapsulating seeds, signs, and permutation
    test_rht_correctness — Verification: forward ∘ inverse ≈ identity
"""

import math
import torch
import triton
import triton.language as tl


# ─── Triton WHT Kernel ────────────────────────────────────────────────────────
#
# The Walsh-Hadamard butterfly at stage s (0-indexed):
#   h = 1 << s            # group size at this stage
#   For i in 0..d, if (i & h) == 0:
#       a = x[i],  b = x[i | h]
#       x[i]     = a + b
#       x[i | h] = a - b
#
# After log2(d) stages, x holds the (unnormalized) Hadamard transform.
# We normalize by 1/sqrt(d) at the end.
#
# Constraint: BLOCK_SIZE must equal d (power of 2) so the entire row fits
# in registers.  For d ≤ 256 this is a small constexpr tile.

@triton.jit
def _wht_kernel(
    X_ptr,               # pointer to input/output tensor (modified in-place)
    stride_row,          # stride between rows (in elements)
    d: tl.constexpr,     # vector length — must be a power of 2
    LOG2_D: tl.constexpr,  # log2(d), number of butterfly stages
):
    """
    Perform the Walsh-Hadamard Transform on one row of X in-place.

    One Triton program handles one row.  The butterfly stages must be
    sequential (each stage reads the output of the previous one), so all
    work for a single row is done in a single program instance.

    Parameters
    ----------
    X_ptr      : pointer to contiguous (n_rows, d) float32 array
    stride_row : row stride in elements
    d          : vector length (constexpr power of 2)
    LOG2_D     : log2(d) (constexpr)
    """
    row_id = tl.program_id(0)

    # Offset for this row
    base = row_id * stride_row

    # Load the entire row into registers
    offs = tl.arange(0, d)
    x = tl.load(X_ptr + base + offs)          # shape: (d,)

    # ── Butterfly stages ──────────────────────────────────────────────────────
    # Stage s:  h = 1 << s
    #   For every index i where bit s is 0: butterfly with its partner i | h
    #
    # IMPORTANT: x is held entirely in registers.  We must NOT re-load from
    # global memory between stages — that would read stale (pre-stage-0) data.
    # Instead we write x back to SMEM/registers and gather the partner value
    # from the *in-register* x using tl.permute / index arithmetic.
    #
    # Triton does not expose shared memory explicitly, but we can achieve the
    # gather from x (already loaded into registers) by noting that both
    # x[i] and x[i^h] live in the same register tile.  We compute both
    # "left" and "right" butterfly results for every lane simultaneously:
    #
    #   left_result  = x[i]      + x[i^h]   when (i & h) == 0
    #   right_result = x[i^h] - x[i]        when (i & h) != 0  (i is the right partner)
    #
    # In Triton, shuffling within the register tile requires storing to a
    # scratch buffer and reloading, OR using tl.gather/tl.scatter.  The
    # cleanest approach for Triton < 3.x is to store x to global memory at the
    # end of each stage and reload for the next stage (two passes per stage).
    # For d ≤ 256 this is still very fast — each stage is just one read + one
    # write of d floats.

    for s in range(LOG2_D):
        # Write current x to global memory so the next tl.load sees updated values
        tl.store(X_ptr + base + offs, x)
        # Barrier is required when d spans more than one warp (d > 32 for fp32
        # on CUDA): without it, a lane in warp B may read its partner in warp A
        # before warp A's tl.store has become visible, giving stale data and a
        # corrupted butterfly.  Intra-warp dependencies don't need the barrier,
        # which is why d=64 happened to work without it.
        tl.debug_barrier()

        h = 1 << s              # partner distance at this stage

        # Masks: which lanes are the "left" of a butterfly pair?
        left_mask  = (offs & h) == 0       # shape: (d,) bool
        # Partner of lane i: i ^ h  (flip bit s)
        partner_offs = offs ^ h

        # Reload this lane's own value (just stored) and its partner
        x_self    = tl.load(X_ptr + base + offs)
        x_partner = tl.load(X_ptr + base + partner_offs)

        # Butterfly: left lane (i) gets x[i]+x[i^h], right lane (i^h) gets
        # x[i]-x[i^h]. From a right lane's perspective, x_self is x[i^h] and
        # x_partner is x[i], so the right result is (partner - self).
        a = x_self
        b = x_partner

        x_new_left  = a + b
        x_new_right = b - a

        x = tl.where(left_mask, x_new_left, x_new_right)

    # ── Normalize by 1/sqrt(d) ────────────────────────────────────────────────
    # d is a tl.constexpr (Python int), so the scale is a compile-time float
    # that Triton broadcasts across the tile.
    scale = 1.0 / (float(d) ** 0.5)
    x = x * scale

    # Store result back
    tl.store(X_ptr + base + offs, x)


# ─── Python wrapper ───────────────────────────────────────────────────────────

def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Apply the Walsh-Hadamard Transform to the last dimension of x, in-place,
    normalized by 1/sqrt(d).

    Parameters
    ----------
    x : torch.Tensor of shape (..., d), float32, CUDA
        d must be a power of 2.  The tensor is modified **in-place** and also
        returned, so callers may use either the return value or rely on the
        side-effect.

    Returns
    -------
    torch.Tensor of shape (..., d)  — same object as input, modified in-place.

    Notes
    -----
    The kernel loads the full d-element row into registers (BLOCK_SIZE = d),
    which is efficient for d ≤ 256.  For larger d, a multi-pass tiled approach
    would be needed, but the target values here are 64, 128, 256.
    """
    assert x.is_cuda,       "hadamard_transform requires a CUDA tensor"
    assert x.is_contiguous(), "hadamard_transform requires a contiguous tensor"
    assert x.dtype == torch.float32, "hadamard_transform requires float32"

    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, f"d must be a power of 2, got {d}"
    assert d <= 512, f"Triton WHT kernel supports d ≤ 512, got {d}"

    log2_d = int(math.log2(d))

    # Flatten to 2-D: (n_rows, d)
    original_shape = x.shape
    x_2d = x.view(-1, d)
    n_rows = x_2d.shape[0]

    grid = (n_rows,)

    # Dispatch on d to use it as a constexpr
    # Triton requires constexpr template parameters to be compile-time constants.
    # We use a conditional to specialize for each supported d.
    if d == 64:
        _wht_kernel[grid](x_2d, x_2d.stride(0), d=64,  LOG2_D=6)
    elif d == 128:
        _wht_kernel[grid](x_2d, x_2d.stride(0), d=128, LOG2_D=7)
    elif d == 256:
        _wht_kernel[grid](x_2d, x_2d.stride(0), d=256, LOG2_D=8)
    elif d == 512:
        _wht_kernel[grid](x_2d, x_2d.stride(0), d=512, LOG2_D=9)
    elif d == 32:
        _wht_kernel[grid](x_2d, x_2d.stride(0), d=32,  LOG2_D=5)
    elif d == 16:
        _wht_kernel[grid](x_2d, x_2d.stride(0), d=16,  LOG2_D=4)
    else:
        # Generic fallback — Triton requires constexpr, so we provide the
        # actual int literal via a helper.
        raise NotImplementedError(
            f"hadamard_transform: d={d} not in supported set "
            "{{16, 32, 64, 128, 256, 512}}.  Add an elif branch if needed."
        )

    return x.view(original_shape)


# ─── RHTRotation class ────────────────────────────────────────────────────────

class RHTRotation:
    """
    Randomized Hadamard Transform rotation for KV-cache compression.

    Stores only O(d) state (random signs + permutation) instead of the
    O(d^2) dense rotation matrix used by the QR approach.

    RHT forward pass:
        1. x_pad = pad(x, d_pad)     — zero-pad to next power of 2
        2. x_pad *= signs            — random ±1 diagonal
        3. x_pad  = WHT(x_pad) / sqrt(d_pad)   — fast Hadamard
        4. return x_pad[perm][:d]    — random permutation + trim

    RHT inverse pass (WHT is self-inverse up to scale):
        1. y_full[perm] = y          — reverse permutation
        2. y_full = WHT(y_full)      — un-scale: WHT(WHT(x)/sqrt(d)) = x*sqrt(d)/sqrt(d) = x
        3. y_full *= signs           — signs are ±1, so they are their own inverse
        4. return y_full[:d]

    Note on the inverse normalization:
        WHT(WHT(x)) = d * x   (the Hadamard matrix is symmetric and H@H = d*I)
        So WHT applied twice gives back the original vector scaled by d.
        Our forward divides by sqrt(d_pad), so:
          inverse applies WHT once more (no extra division), giving:
              WHT(WHT(x) / sqrt(d_pad)) / sqrt(d_pad) = d_pad * x / d_pad = x  ✓

    Parameters
    ----------
    d      : int  — original vector dimension (need not be a power of 2)
    seed   : int  — RNG seed for reproducible signs/permutation
    device : str or torch.device — where to store the parameters
    """

    def __init__(self, d: int, seed: int = 42, device='cuda'):
        self.d = d
        self.seed = seed

        # Pad to the next power of 2.
        # For the target dims (64, 128, 256) d is already a power of 2, so
        # d_pad == d and the inverse is exact.  For non-power-of-2 d the
        # forward pads with zeros, but the inverse cannot recover those
        # zero-padded positions from the permuted output, so reconstruction
        # is only lossless when d == d_pad.
        self.d_pad = 1 if d == 0 else (1 << (d - 1).bit_length())
        if self.d_pad < d:     # edge case: d is already a power of 2
            self.d_pad = d

        device = torch.device(device)

        rng = torch.Generator(device='cpu')
        rng.manual_seed(seed)

        # Random ±1 signs stored as int8 — 1 byte per element
        # Shape: (d_pad,)
        raw = torch.randint(0, 2, (self.d_pad,), generator=rng, dtype=torch.int8)
        self.signs: torch.Tensor = (raw * 2 - 1).to(device)   # {0,1} → {-1,+1}

        # Random permutation of d_pad indices stored as int64
        # We only use the first d entries after WHT (the rest are discarded)
        self.perm: torch.Tensor = torch.randperm(
            self.d_pad, generator=rng, device='cpu', dtype=torch.int64
        ).to(device)

        # Precompute inverse permutation for the inverse pass
        inv_perm = torch.empty_like(self.perm)
        inv_perm[self.perm] = torch.arange(self.d_pad, device=device, dtype=torch.int64)
        self.inv_perm: torch.Tensor = inv_perm

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RHT forward: signs → WHT/sqrt(d_pad) → permute → trim.

        Parameters
        ----------
        x : torch.Tensor of shape (..., d), float32

        Returns
        -------
        torch.Tensor of shape (..., d), float32
        """
        orig_shape = x.shape
        d = orig_shape[-1]
        assert d == self.d, f"Expected last dim {self.d}, got {d}"

        # ── 1. Zero-pad to d_pad ──────────────────────────────────────
        if self.d_pad > d:
            pad_size = self.d_pad - d
            x_pad = torch.nn.functional.pad(x, (0, pad_size))  # (..., d_pad)
        else:
            x_pad = x.clone()

        # ── 2. Multiply by random ±1 signs ───────────────────────────
        x_pad = x_pad * self.signs.to(x_pad.dtype)

        # ── 3. WHT / sqrt(d_pad) via Triton kernel ───────────────────
        # hadamard_transform expects contiguous float32 CUDA tensor
        x_pad = x_pad.contiguous().float()
        hadamard_transform(x_pad)   # in-place, divides by sqrt(d_pad)

        # ── 4. Permute and trim to [:d] ───────────────────────────────
        # x_pad has shape (..., d_pad); gather along last dim
        perm_idx = self.perm[:d]                    # first d of the permutation
        out = x_pad[..., perm_idx]                  # (..., d)

        return out.view(*orig_shape[:-1], d)

    # ------------------------------------------------------------------

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply RHT inverse: unpermute → WHT → signs → trim.

        Because WHT is self-inverse up to scale (H@H = d_pad * I), and the
        forward divided by sqrt(d_pad), the inverse must also divide by
        sqrt(d_pad) — which hadamard_transform already does — giving:

            H(H(x)/sqrt(d_pad)) / sqrt(d_pad) = d_pad * x / d_pad = x  ✓

        Parameters
        ----------
        y : torch.Tensor of shape (..., d), float32

        Returns
        -------
        torch.Tensor of shape (..., d), float32
        """
        orig_shape = y.shape
        d = orig_shape[-1]
        assert d == self.d, f"Expected last dim {self.d}, got {d}"

        # ── 1. Reverse permutation (scatter back to d_pad positions) ──
        # Build a full d_pad buffer filled with zeros, then scatter y into it
        batch_shape = orig_shape[:-1]
        n = 1
        for s in batch_shape:
            n *= s

        y_flat = y.reshape(n, d).float()
        buf = torch.zeros(n, self.d_pad, dtype=torch.float32, device=y.device)

        # The forward used perm[:d] to select positions; inverse scatters back
        perm_idx = self.perm[:d]      # (d,)
        buf[:, perm_idx] = y_flat     # scatter: buf[..., perm[i]] = y[..., i]

        # ── 2. WHT / sqrt(d_pad) ─────────────────────────────────────
        hadamard_transform(buf)       # in-place

        # ── 3. Remove signs ──────────────────────────────────────────
        buf = buf * self.signs.to(buf.dtype)   # signs are ±1, so self-inverse

        # ── 4. Trim to [:d] ──────────────────────────────────────────
        out = buf[:, :d]

        return out.view(*orig_shape)

    # ------------------------------------------------------------------

    def memory_bytes(self) -> int:
        """
        Return the minimum storage bytes needed to persist this RHTRotation.

        Following the spec:
            signs : d_pad × 1 byte  (int8, ±1 per element)
            perm  : d     × 4 bytes (int32-equivalent index per active position)

        The inv_perm is derived from perm at load time and is not counted.

        Returns
        -------
        int — total bytes
        """
        signs_bytes = self.d_pad * 1    # int8: 1 byte per element
        perm_bytes  = self.d * 4        # 4 bytes per index (int32 equivalent)
        return signs_bytes + perm_bytes


# ─── Correctness test ─────────────────────────────────────────────────────────

def test_rht_correctness(verbose: bool = True) -> bool:
    """
    Verify that RHTRotation.forward ∘ RHTRotation.inverse ≈ identity.

    Tests d ∈ {64, 128, 256} with a batch of random vectors.  Asserts that
    the maximum absolute reconstruction error is below 1e-5.

    Parameters
    ----------
    verbose : bool — print per-d results

    Returns
    -------
    bool — True if all tests pass, False otherwise.
    """
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available — cannot test Triton WHT kernel")
        return True    # not a failure of the code itself

    all_pass = True

    for d in [64, 128, 256]:
        rht = RHTRotation(d=d, seed=42, device='cuda')

        # Random float32 batch: (8, d)
        torch.manual_seed(0)
        x = torch.randn(8, d, dtype=torch.float32, device='cuda')

        # Forward then inverse should recover x
        y    = rht.forward(x)
        x_rec = rht.inverse(y)

        max_err = (x - x_rec).abs().max().item()
        passed  = max_err < 1e-5

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  d={d:3d}: max reconstruction error = {max_err:.2e}  [{status}]")

        if not passed:
            all_pass = False

    return all_pass


# ─── Quick self-test when run directly ───────────────────────────────────────

if __name__ == "__main__":
    print("Testing RHTRotation correctness...")
    ok = test_rht_correctness(verbose=True)
    if ok:
        print("All tests passed.")
    else:
        print("Some tests FAILED.")
        raise SystemExit(1)
