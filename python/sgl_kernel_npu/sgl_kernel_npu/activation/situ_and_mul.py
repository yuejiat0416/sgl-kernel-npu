from typing import Optional

import torch
import triton
import triton.language as tl
import triton.backends.ascend.runtime  # REQUIRED: activates the Triton-Ascend autotune path
import triton.language.extra.cann.libdevice as libdevice

from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.autotune(
    configs=[triton.Config({"BLOCK_H": b, "multibuffer": True}) for b in (1024, 2048, 4096, 8192)],
    key=["HALF_COLS", "HAS_GROUP_LIST"],
)
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
    N_ROWS,
    NUM_CORES: tl.constexpr,
    HAS_GROUP_LIST: tl.constexpr,
    BETA: tl.constexpr,
    INV_BETA: tl.constexpr,
    DO_LINEAR_BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    INV_LINEAR_BETA: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # total_rows: from group_list (routed MoE) or N_ROWS (dense / shared, no group_list).
    if HAS_GROUP_LIST:
        if GROUP_LIST_TYPE == 0:  # cusum
            total_rows = tl.load(group_list_ptr + (NUM_EXPERTS - 1)).to(tl.int32)
        else:  # count
            gl_offsets = tl.arange(0, NUM_EXPERTS_ALGIN)
            gl_mask = gl_offsets < NUM_EXPERTS
            group_list = tl.load(group_list_ptr + gl_offsets, gl_mask, other=0).to(tl.int32)
            total_rows = tl.sum(group_list)
    else:
        total_rows = N_ROWS

    # rows distributed across vector cores (manual split + early return for over-provision).
    block_size = (total_rows - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= total_rows:
        return
    row_end = tl.minimum((pid + 1) * block_size, total_rows)

    # H-tile over the OUTPUT dim (HALF_COLS): out[i] only needs gate[i]=x[i] and
    # up[i]=x[d+i], so every [h:h+BLOCK_H] tile is self-contained -- no full-row resident,
    # which is what keeps large d (e.g. 33792) within UB. gate = first half, up = second half.
    h_offs = tl.arange(0, BLOCK_H)
    for row_idx in range(row_begin, row_end):
        # int64 row offset: row_idx * stride stays int32 by default on triton-ascend
        # (no auto-promote), which overflows when N*d is large (e.g. N=32768, d=33792).
        row_off = row_idx.to(tl.int64) * TOTAL_COLS
        gate_base = x_ptr + row_off
        up_base = x_ptr + row_off + HALF_COLS
        out_base = out_ptr + row_idx.to(tl.int64) * HALF_COLS
        for h_start in range(0, HALF_COLS, BLOCK_H):
            h_idx = h_start + h_offs
            mask = h_idx < HALF_COLS
            gate = tl.load(gate_base + h_idx, mask=mask, other=0.0).to(tl.float32)
            up = tl.load(up_base + h_idx, mask=mask, other=0.0).to(tl.float32)
            situ_a = BETA * libdevice.tanh(gate * INV_BETA) * tl.sigmoid(gate)
            if DO_LINEAR_BETA:
                up = LINEAR_BETA * libdevice.tanh(up * INV_LINEAR_BETA)
            out = situ_a * up
            tl.store(out_base + h_idx, out.to(out_ptr.dtype.element_ty), mask=mask)


def situ_and_mul(
    x,
    group_list=None,
    group_list_type=None,
    beta: float = 4.0,
    linear_beta: Optional[float] = 25.0,
):
    """SituAndMul activation with optional MoE group_list.

    Args:
        x: ``[..., 2d]`` tensor (gate | up halves along the last dim).
        group_list: per-expert token counts (count) or cumulative sum (cusum).
            ``None`` = dense / shared-expert path: process ALL rows of ``x`` (no dispatch
            padding). Required only for the routed-MoE path.
        group_list_type: 0 = cusum, 1 = count. Ignored when ``group_list is None``.
        beta: SituAndMul beta (soft-saturation bound on the gate path).
        linear_beta: optional soft-saturation bound on the up path; ``None`` leaves ``up``.

    Returns:
        ``[..., d]`` tensor. With ``group_list``: only the first ``sum(group_list)`` rows
        are written (rest is padding). Without: all rows written.
    """
    has_group_list = group_list is not None
    if has_group_list and group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, but got {group_list_type}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"x last dim must be even, but got {x.shape[-1]}")

    x_2d = x.reshape(-1, x.shape[-1])
    s, h = x_2d.shape
    out = torch.empty((s, h // 2), dtype=x.dtype, device=x.device)

    if has_group_list:
        num_experts = group_list.shape[0]
        if group_list.dtype == torch.int64:
            num_experts_algin = (num_experts + 7) // 8 * 8
        elif group_list.dtype == torch.int32:
            num_experts_algin = (num_experts + 15) // 16 * 16
        else:
            raise ValueError(
                f"group_list dtype must be torch.int32 or torch.int64, "
                f"but got {group_list.dtype}"
            )
        group_list_arg = group_list
        num_experts_arg = num_experts
        num_experts_algin_arg = num_experts_algin
        gl_type_arg = group_list_type
    else:
        # dense / shared: kernel skips the group_list block (HAS_GROUP_LIST=False),
        # so these are never read -- pass harmless dummies.
        group_list_arg = x_2d
        num_experts_arg = 1
        num_experts_algin_arg = 1
        gl_type_arg = 0

    do_linear_beta = linear_beta is not None
    linear_beta_v = linear_beta if do_linear_beta else 1.0

    _, num_vectorcore = get_device_properties()
    _situ_and_mul_kernel[(num_vectorcore,)](
        x_2d,
        group_list_arg,
        out,
        TOTAL_COLS=h,
        HALF_COLS=h // 2,
        NUM_EXPERTS=num_experts_arg,
        NUM_EXPERTS_ALGIN=num_experts_algin_arg,
        GROUP_LIST_TYPE=gl_type_arg,
        N_ROWS=s,
        NUM_CORES=num_vectorcore,
        HAS_GROUP_LIST=has_group_list,
        BETA=beta,
        INV_BETA=1.0 / beta,
        DO_LINEAR_BETA=do_linear_beta,
        LINEAR_BETA=linear_beta_v,
        INV_LINEAR_BETA=(1.0 / linear_beta_v) if do_linear_beta else 1.0,
    )
    return out.reshape(*x.shape[:-1], h // 2)
