"""Inference-only historical attention adapters; the dense path stays in model.py.

GDN compresses chronological, post-RoPE K/V into a decayed recurrent matrix.
It preserves commit/speculation transactions, NOT dense FIFO token eviction.
No model projections, RMSNorm, RoPE or output projections are replaced.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MemorySnapshot:
    committed: torch.Tensor | None = None
    speculative: torch.Tensor | None = None
    committed_tokens: int = 0
    speculative_tokens: int = 0

    @property
    def active(self):
        return self.speculative if self.speculative is not None else self.committed


def local_softmax(q, k, v):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)


def reference_update(k, v, g, beta, state=None):
    """Independent, slow FP32 delta recurrence for correctness tests only."""
    k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6)
    b, t, h, d = k.shape
    s = (
        torch.zeros(b, h, d, v.shape[-1], device=k.device)
        if state is None
        else state.float().clone()
    )
    for i in range(t):
        s = s * g[:, i, :, None, None].exp()
        residual = v[:, i].float() - torch.einsum("bhd,bhdv->bhv", k[:, i], s)
        s = s + torch.einsum("bhd,bhv->bhdv", k[:, i], residual * beta[:, i, :, None])
    return s


class GDNHistory(nn.Module):
    def __init__(self, heads, head_dim, kernel="auto", read_kernel="torch"):
        super().__init__()
        # Minimal learned per-head gates; no extra hidden-state projections.
        self.log_decay = nn.Parameter(
            torch.full((heads,), math.log(math.expm1(0.0005)))
        )
        self.beta_logit = nn.Parameter(torch.full((heads,), math.log(0.1 / 0.9)))
        self.gamma = nn.Parameter(torch.full((heads,), 0.1))
        self.head_dim = head_dim
        self.kernel = kernel
        self.read_kernel = read_kernel
        self.states = {}

    def clear(self, name):
        self.states.pop(name, None)

    def snapshot(self, name):
        return self.states.get(name, MemorySnapshot())

    def restore(self, name, snapshot):
        if not isinstance(snapshot, MemorySnapshot):
            raise TypeError("GDN rollback requires a MemorySnapshot")
        self.states[name] = snapshot

    def clear_pred(self, name):
        old = self.snapshot(name)
        self.states[name] = MemorySnapshot(old.committed, None, old.committed_tokens, 0)

    def update_state(self, k, v, state=None):
        b, t, h, _ = k.shape
        g = (
            (-F.softplus(self.log_decay.float()))
            .view(1, 1, h)
            .expand(b, t, h)
            .contiguous()
        )
        beta = (
            self.beta_logit.float().sigmoid().view(1, 1, h).expand(b, t, h).contiguous()
        )
        if self.kernel == "reference":
            return reference_update(k, v, g, beta, state)
        if not k.is_cuda:
            raise RuntimeError("FLA GDN requires CUDA; reference is for tests only")
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )

        differentiable = torch.is_grad_enabled() and (
            g.requires_grad or beta.requires_grad or k.requires_grad or v.requires_grad
        )
        kernel = self.kernel
        if kernel == "auto":
            # H100 B=2,H=24,D=128 measurements favor recurrent for both
            # 16-action and 128-video updates; chunk remains for long/backward.
            kernel = "chunk" if differentiable or t > 128 else "recurrent"
        if differentiable and kernel == "recurrent":
            raise RuntimeError("Use chunk GDN for alignment/backward")
        op = (
            chunk_gated_delta_rule
            if kernel == "chunk"
            else fused_recurrent_gated_delta_rule
        )
        _, final = op(
            q=k.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g,
            beta=beta,
            scale=1.0,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        return final

    def read(self, q, state, local):
        if state is None:
            return local
        if self.read_kernel == "triton" and not torch.is_grad_enabled():
            from .history_kernels import fused_history_read

            return fused_history_read(q, state, local, self.gamma)
        qn = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
        history = torch.matmul(qn.transpose(1, 2), state.float()).transpose(1, 2)
        return (local.float() + self.gamma.float()[None, None, :, None] * history).to(
            q.dtype
        )

    def forward(self, q, k, v, update_cache=0, cache_name="pos"):
        if update_cache not in (0, 1, 2):
            raise ValueError(
                "update_cache must be 0=temporary, 1=predicted, or 2=observed"
            )
        old = self.snapshot(cache_name)
        if update_cache == 2 and old.speculative is not None:
            raise RuntimeError("Clear predicted state before committing observations")
        output = self.read(q, old.active, local_softmax(q, k, v))
        if update_cache:
            new_state = self.update_state(k, v, old.active)
            if update_cache == 1:
                self.states[cache_name] = MemorySnapshot(
                    old.committed,
                    new_state,
                    old.committed_tokens,
                    old.speculative_tokens + k.shape[1],
                )
            else:
                self.states[cache_name] = MemorySnapshot(
                    new_state, None, old.committed_tokens + k.shape[1], 0
                )
        return output

    def state_bytes(self):
        tensors = {
            id(t): t
            for s in self.states.values()
            for t in (s.committed, s.speculative)
            if t is not None
        }
        return sum(t.numel() * t.element_size() for t in tensors.values())


def converted_layers(count, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0,1]")
    n = math.floor(count * fraction + 0.5)
    return (
        sorted({min(count - 1, int((i + 0.5) * count / n)) for i in range(n)})
        if n
        else []
    )


def configure_history_attention(
    model,
    backend="full",
    fraction=1.0,
    kernel="auto",
    read_kernel="torch",
    adapter_path=None,
):
    """Call after loading original weights and before FSDP wrapping."""
    if backend not in ("full", "local", "local_gdn"):
        raise ValueError(f"Unknown history backend {backend}")
    chosen = (
        set(converted_layers(len(model.blocks), fraction))
        if backend != "full"
        else set()
    )
    for i, block in enumerate(model.blocks):
        attn = block.attn1
        attn.attn_caches = {}
        attn.history_backend = backend if i in chosen else "full"
        attn.history_memory = None
        if attn.history_backend == "local_gdn":
            # Keep tiny adaptation parameters in FP32 even with bf16 base weights.
            attn.history_memory = GDNHistory(
                attn.heads, attn.inner_dim // attn.heads, kernel, read_kernel
            ).to(attn.to_q.weight.device)
    if adapter_path:
        state = torch.load(adapter_path, map_location="cpu", weights_only=True)
        for i in chosen:
            if model.blocks[i].attn1.history_memory is not None:
                model.blocks[i].attn1.history_memory.load_state_dict(
                    state[str(i)], strict=True
                )
    return sorted(chosen)


def cache_bytes(model):
    total = 0
    for block in model.blocks:
        a = block.attn1
        if a.history_memory is not None:
            total += a.history_memory.state_bytes()
        else:
            for cache in a.attn_caches.values():
                if cache:
                    total += sum(
                        t.numel() * t.element_size()
                        for t in cache.values()
                        if torch.is_tensor(t)
                    )
    return total
