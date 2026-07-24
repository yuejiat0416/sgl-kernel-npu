from typing import Optional

import torch
import triton
import triton.language as tl
import triton.backends.ascend.runtime  # REQUIRED: activates the Triton-Ascend autotune path
import triton.language.extra.cann.libdevice as libdevice
import triton.language.extra.cann.extension as al

from sgl_kernel_npu.utils.triton_utils import get_device_properties

# quant supports d in {3072, 6144} (full-row load, fits UB); d=33792 falls back to the
# unquant kernel below (no quantization). See situ_and_mul_quant() dispatch.


@triton.jit
def _situ_and_mul_quant_kernel(
    x_ptr,
    group_list_ptr,
    out_ptr,
    scale_ptr,
    TOTAL_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
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
    SCALE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
):
    # total_rows: from group_list (routed MoE) or N_ROWS (dense / shared).
    if HAS_GROUP_LIST:
        if GROUP_LIST_TYPE == 0:  # cusum
            total_rows = tl.load(group_list_ptr + NUM_EXPERTS).to(tl.int32)
        else:  # count
            gl_offsets = tl.arange(0, NUM_EXPERTS_ALGIN)
            gl_mask = gl_offsets < NUM_EXPERTS
            group_list = tl.load(group_list_ptr + gl_offsets, gl_mask, other=0).to(tl.int32)
            total_rows = tl.sum(group_list)
    else:
        total_rows = N_ROWS

    block_size = (total_rows - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= total_rows:
        return
    row_end = tl.minimum((pid + 1) * block_size, total_rows)

    # full-row load (d<=6144 fits UB): situ computed once, single tl.max over the row.
    cols = tl.arange(0, HALF_COLS)
    for row_idx in range(row_begin, row_end):
        row_off = row_idx.to(tl.int64) * TOTAL_COLS
        gate = tl.load(x_ptr + row_off + cols).to(tl.float32)
        up = tl.load(x_ptr + row_off + HALF_COLS + cols).to(tl.float32)
        situ_a = BETA * libdevice.tanh(gate * INV_BETA) * tl.sigmoid(gate)
        if DO_LINEAR_BETA:
            up = LINEAR_BETA * libdevice.tanh(up * INV_LINEAR_BETA)
        out = situ_a * up

        if SCALE:
            scale = tl.maximum(tl.max(tl.abs(out)) / DTYPE_MAX, 1e-30)
            tl.store(scale_ptr + row_idx.to(tl.int64), scale.to(scale_ptr.dtype.element_ty))
            # quantize in COL_BLOCK_SIZE slices (a full-row rint overflows UB, cf. swiglu_quant).
            for cb in range(0, HALF_COLS, COL_BLOCK_SIZE):
                tmp = al.extract_slice(out, offsets=(cb,), sizes=(COL_BLOCK_SIZE,), strides=(1,))
                tmp = tmp.to(tl.float32) / scale
                tmp = tl.floor(tmp + 0.5)
                tmp = tl.clamp(tmp, -128, 127).to(tl.int8)
                c_idx = cb + tl.arange(0, COL_BLOCK_SIZE)
                mask = c_idx < HALF_COLS
                tl.store(out_ptr + row_idx.to(tl.int64) * HALF_COLS + c_idx,
                         tmp.to(out_ptr.dtype.element_ty), mask=mask)
        else:
            tl.store(out_ptr + row_idx.to(tl.int64) * HALF_COLS + cols,
                     out.to(out_ptr.dtype.element_ty))


@triton.autotune(
    configs=[triton.Config({"BLOCK_H": b, "multibuffer": True}) for b in (4096, 8192, 16384)],
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
    # unquant situ (H-tiled), same as situ_and_mul.py -- used for d=33792 (no quant).
    if HAS_GROUP_LIST:
        if GROUP_LIST_TYPE == 0:
            total_rows = tl.load(group_list_ptr + NUM_EXPERTS).to(tl.int32)
        else:
            gl_offsets = tl.arange(0, NUM_EXPERTS_ALGIN)
            gl_mask = gl_offsets < NUM_EXPERTS
            group_list = tl.load(group_list_ptr + gl_offsets, gl_mask, other=0).to(tl.int32)
            total_rows = tl.sum(group_list)
    else:
        total_rows = N_ROWS

    block_size = (total_rows - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= total_rows:
        return
    row_end = tl.minimum((pid + 1) * block_size, total_rows)

    h_offs = tl.arange(0, BLOCK_H)
    for row_idx in range(row_begin, row_end):
        row_off = row_idx.to(tl.int64) * TOTAL_COLS
        for h_start in range(0, HALF_COLS, BLOCK_H):
            h_idx = h_start + h_offs
            mask = h_idx < HALF_COLS
            gate = tl.load(x_ptr + row_off + h_idx, mask=mask, other=0.0).to(tl.float32)
            up = tl.load(x_ptr + row_off + HALF_COLS + h_idx, mask=mask, other=0.0).to(tl.float32)
            situ_a = BETA * libdevice.tanh(gate * INV_BETA) * tl.sigmoid(gate)
            if DO_LINEAR_BETA:
                up = LINEAR_BETA * libdevice.tanh(up * INV_LINEAR_BETA)
            out = situ_a * up
            tl.store(out_ptr + row_idx.to(tl.int64) * HALF_COLS + h_idx,
                     out.to(out_ptr.dtype.element_ty), mask=mask)


def situ_and_mul_quant(
    x,
    group_list=None,
    group_list_type=None,
    beta: float = 4.0,
    linear_beta: Optional[float] = 25.0,
    need_quant: bool = True,
    quant_type: int = 0,
):
    """SituAndMul activation + fused dynamic int8 quant (d<=6144); unquant fallback (d=33792).

    Args:
        x: ``[..., 2d]`` tensor (gate | up halves along the last dim).
        group_list: per-expert token counts (count) or cumulative sum (cusum).
            ``None`` = dense / shared path (all rows). Required for routed MoE.
        group_list_type: 0 = cusum, 1 = count. Ignored when ``group_list is None``.
        beta / linear_beta: SituAndMul soft-saturation bounds (``linear_beta=None`` leaves up).
        need_quant: True -> int8 out + per-token fp32 scale; False -> activation out (scale is
            uninitialised, caller must ignore).
        quant_type: 0 = int8 (default), 1 = fp8 (deferred -> NotImplementedError).

    Returns:
        ``(out, scale)``. For d<=6144 + quant: ``out`` int8, ``scale`` fp32. For d>6144 (e.g.
        33792) or need_quant=False: ``out`` is the BF16/FP32 activation (no quant), ``scale``
        uninitialised -- quant only supports d in {3072, 6144}.
    """
    if quant_type not in (0, 1):
        raise ValueError(f"quant_type must be 0 (int8) or 1 (fp8), but got {quant_type}")
    if need_quant and quant_type == 1:
        raise NotImplementedError(
            "fp8 (quant_type=1) is deferred: A5-only, uses npu_dynamic_mx_quant (not fusible "
            "into Triton); MoE MXFP8 downstream still WIP in sglang. Use quant_type=0 (int8)."
        )

    has_group_list = group_list is not None
    if has_group_list and group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, but got {group_list_type}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"x last dim must be even, but got {x.shape[-1]}")

    x_2d = x.reshape(-1, x.shape[-1])
    s, h = x_2d.shape
    half_cols = h // 2
    # quant only for small d (3072/6144); large d (33792) -> unquant fallback.
    do_quant = need_quant and (half_cols <= 6144)
    out_dtype = torch.int8 if do_quant else x.dtype
    out = torch.empty((s, half_cols), dtype=out_dtype, device=x.device)
    scale = torch.empty((s,), dtype=torch.float32, device=x.device)

    if has_group_list:
        num_experts = group_list.shape[0]
        if group_list.dtype == torch.int64:
            num_experts_algin = (num_experts + 7) // 8 * 8
        elif group_list.dtype == torch.int32:
            num_experts_algin = (num_experts + 15) // 16 * 16
        else:
            raise ValueError(
                f"group_list dtype must be torch.int32 or torch.int64, but got {group_list.dtype}"
            )
        group_list_arg = group_list
        num_experts_arg = num_experts
        num_experts_algin_arg = num_experts_algin
        gl_type_arg = group_list_type
    else:
        group_list_arg = x_2d
        num_experts_arg = 1
        num_experts_algin_arg = 1
        gl_type_arg = 0

    do_linear_beta = linear_beta is not None
    linear_beta_v = linear_beta if do_linear_beta else 1.0

    _, num_vectorcore = get_device_properties()
    if do_quant:
        _situ_and_mul_quant_kernel[(num_vectorcore,)](
            x_2d, group_list_arg, out, scale,
            TOTAL_COLS=h, HALF_COLS=half_cols, COL_BLOCK_SIZE=half_cols,
            NUM_EXPERTS=num_experts_arg, NUM_EXPERTS_ALGIN=num_experts_algin_arg,
            GROUP_LIST_TYPE=gl_type_arg, N_ROWS=s, NUM_CORES=num_vectorcore,
            HAS_GROUP_LIST=has_group_list, BETA=beta, INV_BETA=1.0 / beta,
            DO_LINEAR_BETA=do_linear_beta, LINEAR_BETA=linear_beta_v,
            INV_LINEAR_BETA=(1.0 / linear_beta_v) if do_linear_beta else 1.0,
            SCALE=need_quant, DTYPE_MAX=127, multibuffer=True,
        )
    else:
        _situ_and_mul_kernel[(num_vectorcore,)](
            x_2d, group_list_arg, out,
            TOTAL_COLS=h, HALF_COLS=half_cols,
            NUM_EXPERTS=num_experts_arg, NUM_EXPERTS_ALGIN=num_experts_algin_arg,
            GROUP_LIST_TYPE=gl_type_arg, N_ROWS=s, NUM_CORES=num_vectorcore,
            HAS_GROUP_LIST=has_group_list, BETA=beta, INV_BETA=1.0 / beta,
            DO_LINEAR_BETA=do_linear_beta, LINEAR_BETA=linear_beta_v,
            INV_LINEAR_BETA=(1.0 / linear_beta_v) if do_linear_beta else 1.0,
        )
    return out.reshape(*x.shape[:-1], half_cols), scale
