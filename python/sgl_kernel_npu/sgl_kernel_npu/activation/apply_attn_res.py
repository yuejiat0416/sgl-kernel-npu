"""apply_attn_res (K3 learned attn-residual) BF16 Triton-Ascend kernel for Ascend 910C.

Activation math adapted from the PyTorch ``_apply_attn_res`` in Kimi-K3's
HuggingFace model (``modeling_kimi.py:1119``); kernel structure mirrors
``sgl_kernel_npu.activation.swiglu_quant`` (vector-core grid, full-row load +
``tl.arange``, ``multibuffer``, FP32-internal) with the SituAndMul-style math
removed and the learned attn-residual math in its place.
"""

import torch
import triton
import triton.language as tl

from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _apply_attn_res_kernel(
    prefix_sum_ptr,
    block_residual_ptr,
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

    cols = tl.arange(0, H)                                   # full-row (Triton-Ascend allows non-pow2)
    s_idx = tl.arange(0, NB)                                 # padded stream-index block for softmax
    w = tl.load(score_weight_ptr + cols).to(tl.float32)      # [H] score_weight, resident

    for tok in range(tok0, tok1):
        # ---- pass 1: per stream, one full-row load; MS + vw; score = rstd * vw ----
        scores = tl.full([NB], -float("inf"), dtype=tl.float32)   # pad slots stay -inf -> softmax 0
        for s in range(B + 1):
            if s < B:
                base = block_residual_ptr + tok * B * H + s * H
            else:
                base = prefix_sum_ptr + tok * H
            v = tl.load(base + cols).to(tl.float32)
            ms = tl.sum(v * v) / H
            vw = tl.sum(v * w)
            rstd = tl.rsqrt(ms + EPS)
            scores = tl.where(s_idx == s, rstd * vw, scores)

        # ---- softmax over B+1 (pad slots are -inf -> weight 0) ----
        weights = tl.softmax(scores, axis=0)

        # ---- pass 2: weighted sum of raw streams (full-row reload) ----
        out = tl.zeros([H], dtype=tl.float32)
        for s in range(B + 1):
            if s < B:
                base = block_residual_ptr + tok * B * H + s * H
            else:
                base = prefix_sum_ptr + tok * H
            v = tl.load(base + cols).to(tl.float32)
            w_s = tl.sum(tl.where(s_idx == s, weights, 0.0))     # extract scalar weight_s
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
    proj_w = proj.weight.squeeze(0)                         # [H]
    norm_w = norm.weight                                     # [H]
    eps = norm.variance_epsilon
    score_weight = (norm_w * proj_w).float()                # [H] fp32 — host-precomputed constant
    out = torch.empty((N, H), dtype=prefix_sum.dtype, device=prefix_sum.device)
    NB = triton.next_power_of_2(B + 1)

    _, num_vectorcore = get_device_properties()
    _apply_attn_res_kernel[(num_vectorcore,)](
        prefix_sum,
        block_residual,
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
