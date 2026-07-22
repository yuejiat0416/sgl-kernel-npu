"""apply_attn_res (K3 learned attn-residual) BF16 kernel for Ascend 910C."""

import torch
import triton
import triton.language as tl

from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _apply_attn_res_kernel(
    v_ptr,
    score_weight_ptr,
    out_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    B: tl.constexpr,
    EPS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    NB: tl.constexpr,
):
    block_size = (N - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    tok0 = pid * block_size
    if tok0 >= N:
        return
    tok1 = tl.minimum(tok0 + block_size, N)

    cols = tl.arange(0, H)                                   # full-row (non-pow2 OK on Ascend)
    s_idx = tl.arange(0, NB)                                 # padded stream-index block
    w = tl.load(score_weight_ptr + cols).to(tl.float32)      # [H] score_weight, resident

    for tok in range(tok0, tok1):
        # ---- pass 1: per stream, one full-row load; MS + vw; score = rstd * vw ----
        scores = tl.full([NB], -float("inf"), dtype=tl.float32)
        for s in range(B + 1):
            v = tl.load(v_ptr + tok * (B + 1) * H + s * H + cols).to(tl.float32)
            ms = tl.sum(v * v) / H
            rstd = tl.rsqrt(ms + EPS)
            k = v * rstd  # normalize first (matches reference FP32 path exactly)
            scores = tl.where(s_idx == s, tl.sum(k * w), scores)

        # ---- softmax (manual: tl.max + tl.exp + tl.sum; tl.softmax unusable) ----
        scores_max = tl.max(scores)
        exp_scores = tl.exp(scores - scores_max)
        weights = exp_scores / tl.sum(exp_scores)

        # ---- pass 2: weighted sum of raw streams (full-row reload) ----
        out = tl.zeros([H], dtype=tl.float32)
        for s in range(B + 1):
            v = tl.load(v_ptr + tok * (B + 1) * H + s * H + cols).to(tl.float32)
            w_s = tl.sum(tl.where(s_idx == s, weights, 0.0))
            out += w_s * v

        tl.store(out_ptr + tok * H + cols, out.to(out_ptr.dtype.element_ty))


def apply_attn_res(prefix_sum, block_residual, proj, norm):
    """K3 learned attn-residual: softmax-mix B+1 residual streams per token.

    Args:
        prefix_sum: [N, H] BF16 (current running sum).
        block_residual: [N, B, H] BF16 (B past block snapshots).
        proj: nn.Linear(H, 1) — learned per-channel scoring projection.
        norm: KimiRMSNorm-like — has .weight [H] and .variance_epsilon (float).

    Returns:
        [N, H] BF16 — the softmax-weighted mix of the B+1 streams.
    """
    N, H = prefix_sum.shape
    B = block_residual.shape[1]
    proj_w = proj.weight.squeeze(0)
    norm_w = norm.weight
    eps = norm.variance_epsilon
    score_weight = (norm_w * proj_w).float()

    # Cat into single [N, B+1, H] — kernel reads from one pointer
    # (Triton-Ascend can't select between two different-source pointers in-kernel).
    v = torch.cat([block_residual, prefix_sum.unsqueeze(1)], dim=1)

    out = torch.empty((N, H), dtype=prefix_sum.dtype, device=prefix_sum.device)
    NB = triton.next_power_of_2(B + 1)

    _, num_vectorcore = get_device_properties()
    _apply_attn_res_kernel[(num_vectorcore,)](
        v,
        score_weight,
        out,
        N=N,
        H=H,
        B=B,
        EPS=eps,
        NUM_CORES=num_vectorcore,
        NB=NB,
        multibuffer=True,
    )
    return out
