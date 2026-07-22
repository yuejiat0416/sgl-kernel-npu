"""SituAndMul activation (BF16) for Ascend 910C, MoE group_list aware.

Activation math adapted from the PyTorch ``SituAndMul`` in sglang's Kimi-K3
model; kernel structure mirrors ``sgl_kernel_npu.activation.swiglu_quant``
(group_list handling, vector-core grid, full-row load + ``al.extract_slice``,
``multibuffer``, FP32-internal compute) with no quantization.

Per real row (the first ``sum(group_list)`` rows of ``x``), with
``d = x.shape[-1] // 2``::

    gate   = x[:, :d],  up = x[:, d:]
    situ_a = beta * tanh(gate / beta) * sigmoid(gate)
    up     = linear_beta * tanh(up / linear_beta)   # only when linear_beta is set
    out    = (situ_a * up).to(x.dtype)

``beta`` and ``linear_beta`` are model-config scalars specialized as
``tl.constexpr``. At the production ``d = moe_intermediate_size = 3072`` a
full-row FP32 load is ~24 KiB, well within the ~192 KiB Unified Buffer.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.language.extra.cann.libdevice as libdevice

from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _situ_and_mul_kernel(
    x_ptr,
    group_list_ptr,
    out_ptr,
    TOTAL_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_ALGIN: tl.constexpr,
    GROUP_LIST_TYPE: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BETA: tl.constexpr,
    INV_BETA: tl.constexpr,
    DO_LINEAR_BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    INV_LINEAR_BETA: tl.constexpr,
):
    # calc real total_rows (same as swiglu_quant)
    if GROUP_LIST_TYPE == 0:  # cusum
        total_rows = tl.load(group_list_ptr + NUM_EXPERTS).to(tl.int32)
    else:
        gl_offsets = tl.arange(0, NUM_EXPERTS_ALGIN)
        gl_mask = gl_offsets < NUM_EXPERTS
        group_list = tl.load(group_list_ptr + gl_offsets, gl_mask, other=0).to(tl.int32)
        total_rows = tl.sum(group_list)

    block_size = (total_rows - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= total_rows:
        return
    row_end = tl.minimum((pid + 1) * block_size, total_rows)

    for row_idx in range(row_begin, row_end):
        # situ_and_mul
        x_offsets = row_idx * TOTAL_COLS + tl.arange(0, TOTAL_COLS)
        cur_x = tl.load(x_ptr + x_offsets).to(tl.float32)
        gate = al.extract_slice(cur_x, offsets=(0,), sizes=(HALF_COLS,), strides=(1,))
        up = al.extract_slice(
            cur_x, offsets=(HALF_COLS,), sizes=(HALF_COLS,), strides=(1,)
        )

        # situ_a = beta * tanh(gate / beta) * sigmoid(gate). tl.tanh is
        # unavailable on Ascend, so tanh uses the CANN libdevice lowering.
        situ_a = BETA * libdevice.tanh(gate * INV_BETA) * tl.sigmoid(gate)
        if DO_LINEAR_BETA:
            up = LINEAR_BETA * libdevice.tanh(up * INV_LINEAR_BETA)
        out = situ_a * up

        # store out
        o_offsets = row_idx * HALF_COLS + tl.arange(0, HALF_COLS)
        tl.store(out_ptr + o_offsets, out.to(out_ptr.dtype.element_ty))


def situ_and_mul(
    x,
    group_list,
    group_list_type,
    beta: float = 4.0,
    linear_beta: Optional[float] = 25.0,
):
    """SituAndMul activation with MoE group_list.

    Args:
        x: ``[..., 2d]`` BF16 tensor (gate | up halves along the last dim).
        group_list: per-expert token counts (count) or cumulative sum (cusum).
        group_list_type: 0 = cusum, 1 = count.
        beta: SituAndMul beta (soft-saturation bound on the gate path).
        linear_beta: optional soft-saturation bound on the up path; ``None``
            leaves ``up`` unchanged.

    Returns:
        ``[..., d]`` BF16 tensor; only the first ``sum(group_list)`` rows are
        written (the rest are padding and left uninitialized).
    """
    # group_list_type must be 0 cusum or 1 count
    if group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, but got {group_list_type}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"x last dim must be even, but got {x.shape[-1]}")

    x_2d = x.reshape(-1, x.shape[-1])
    s, h = x_2d.shape
    out = torch.empty((s, h // 2), dtype=x.dtype, device=x.device)
    num_experts = group_list.shape[0]

    # ub must be 32-byte aligned on npu
    if group_list.dtype == torch.int64:
        num_experts_algin = (num_experts + 7) // 8 * 8
    elif group_list.dtype == torch.int32:
        num_experts_algin = (num_experts + 15) // 16 * 16
    else:
        raise ValueError(
            f"group_list dtype must be torch.int32 or torch.int64, "
            f"but got {group_list.dtype}"
        )

    do_linear_beta = linear_beta is not None
    linear_beta_v = linear_beta if do_linear_beta else 1.0

    _, num_vectorcore = get_device_properties()
    _situ_and_mul_kernel[(num_vectorcore,)](
        x_2d,
        group_list,
        out,
        TOTAL_COLS=h,
        HALF_COLS=h // 2,
        NUM_EXPERTS=num_experts,
        NUM_EXPERTS_ALGIN=num_experts_algin,
        GROUP_LIST_TYPE=group_list_type,
        NUM_CORES=num_vectorcore,
        BETA=beta,
        INV_BETA=1.0 / beta,
        DO_LINEAR_BETA=do_linear_beta,
        LINEAR_BETA=linear_beta_v,
        INV_LINEAR_BETA=(1.0 / linear_beta_v) if do_linear_beta else 1.0,
        multibuffer=True,
    )
    return out.reshape(*x.shape[:-1], h // 2)
