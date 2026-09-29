"""Fused historical readout; recurrence updates use upstream FLA kernels.

FP32 recurrent state is retained. TF32x3 dot products avoid reducing it to bf16.
One launch fuses query L2 normalization, state readout, gamma and local residual.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _read(
    Q,
    S,
    L,
    G,
    O,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    LS0: tl.constexpr,
    LS1: tl.constexpr,
    LS2: tl.constexpr,
    M: tl.constexpr = 16,
    N: tl.constexpr = 64,
):
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    rows = tl.program_id(1) * M + tl.arange(0, M)
    cols = tl.program_id(2) * N + tl.arange(0, N)
    ks = tl.arange(0, D)
    q = tl.load(
        Q + b * QS0 + rows[:, None] * QS1 + h * QS2 + ks[None, :],
        rows[:, None] < T,
        other=0,
    ).to(tl.float32)
    norm = tl.sqrt(tl.sum(q * q, axis=1))
    q = q / tl.maximum(norm[:, None], 1e-6)
    state = tl.load(
        S + bh * D * D + ks[:, None] * D + cols[None, :], cols[None, :] < D, other=0
    )
    hist = tl.dot(q, state, input_precision="tf32x3")
    local = tl.load(
        L + b * LS0 + rows[:, None] * LS1 + h * LS2 + cols[None, :],
        (rows[:, None] < T) & (cols[None, :] < D),
        other=0,
    ).to(tl.float32)
    gamma = tl.load(G + h).to(tl.float32)
    result = local + gamma * hist
    tl.store(
        O + ((b * T + rows[:, None]) * H + h) * D + cols[None, :],
        result,
        (rows[:, None] < T) & (cols[None, :] < D),
    )


def fused_history_read(q, state, local, gamma):
    b, t, h, d = q.shape
    if d != 128 or not state.is_contiguous():
        raise ValueError(
            "Fused LingBot readout requires contiguous [B,H,128,128] state"
        )
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    _read[(b * h, triton.cdiv(t, 16), triton.cdiv(d, 64))](
        q,
        state,
        local,
        gamma,
        out,
        t,
        h,
        d,
        *q.stride()[:3],
        *local.stride()[:3],
        num_warps=4,
    )
    return out
