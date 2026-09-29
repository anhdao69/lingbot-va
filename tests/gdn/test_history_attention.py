import importlib.util
import subprocess

import pytest
import torch

from wan_va.modules.history_attention import (
    GDNHistory,
    converted_layers,
    local_softmax,
)
from wan_va.modules.model import WanAttention


def inputs(device="cpu", dtype=torch.float32, tokens=4, heads=2, dim=8):
    torch.manual_seed(72)
    return tuple(
        torch.randn(1, tokens, heads, dim, device=device, dtype=dtype) for _ in range(3)
    )


def test_uniform_layer_counts():
    assert len(converted_layers(30, 0.5)) == 15
    assert len(converted_layers(30, 0.75)) == 23
    assert converted_layers(30, 1) == list(range(30))
    assert converted_layers(30, 0) == []


def test_temporary_denoising_cannot_mutate_committed():
    m = GDNHistory(2, 8, kernel="reference")
    q, k, v = inputs()
    m(q, k, v, 2)
    before = m.snapshot("pos")
    saved = before.committed.clone()
    for _ in range(4):
        m(q, k * 2, v * 3, 0)
    assert m.snapshot("pos") is before
    torch.testing.assert_close(before.committed, saved, rtol=0, atol=0)


def test_speculative_video_conditions_action_and_rollback():
    m = GDNHistory(2, 8, kernel="reference")
    q, k, v = inputs()
    m(q, k, v, 2)
    committed = m.snapshot("pos")
    base = m(q, k, v, 0)
    m(q, k, v * 4, 1)  # predicted video
    video = m.snapshot("pos")
    assert video.committed is committed.committed
    action = m(q, k, v, 0)
    assert not torch.allclose(base, action)
    assert m.snapshot("pos") is video
    m(q, k, v * 2, 1)  # final predicted actions build on predicted video
    assert m.snapshot("pos").speculative_tokens == 8
    m.clear_pred("pos")
    torch.testing.assert_close(m(q, k, v, 0), base, rtol=0, atol=0)
    m.restore("pos", video)
    torch.testing.assert_close(m(q, k, v, 0), action, rtol=0, atol=0)
    m.clear_pred("pos")
    m(q, k, v, 2)
    assert m.snapshot("pos").committed_tokens == 8
    assert committed.committed_tokens == 4


def test_no_episode_leak_and_cache_names_independent():
    m = GDNHistory(2, 8, kernel="reference")
    q, k, v = inputs()
    expected = local_softmax(q, k, v)
    m(q, k, v, 2, "first")
    torch.testing.assert_close(m(q, k, v, 0, "second"), expected, rtol=0, atol=0)
    m.clear("first")
    torch.testing.assert_close(m(q, k, v, 0, "first"), expected, rtol=0, atol=0)
    assert m.state_bytes() == 0


def test_commit_rejects_uncleared_prediction():
    m = GDNHistory(2, 8, kernel="reference")
    q, k, v = inputs()
    m(q, k, v, 1)
    with pytest.raises(RuntimeError, match="Clear predicted"):
        m(q, k, v, 2)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_dense_original_equivalence_including_eviction(tmp_path, device):
    # The official base commit is deliberately independent of the branch's code.
    source = subprocess.check_output(
        ["git", "show", "7c6ffa9:wan_va/modules/model.py"], text=True
    )
    source = source.replace(
        "except:\n    from flash_attn import flash_attn_func",
        "except ImportError:\n    flash_attn_func = None",
    )
    path = tmp_path / "original.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("original_lingbot_attention", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(3)
    old = module.WanAttention(32, heads=2, dim_head=16, attn_mode="torch").to(device)
    new = WanAttention(32, heads=2, dim_head=16, attn_mode="torch").to(device)
    new.load_state_dict(old.state_dict(), strict=True)
    for a in (old, new):
        a.init_kv_cache("pos", 12, 2, 16, device, torch.float32, 1)
    for step, flag in enumerate([0, 0, 1, 0, 1, 2, 2, 0, 0, 1, 0, 1, 2, 2] * 2):
        if flag == 2:
            old.clear_pred_cache("pos")
            new.clear_pred_cache("pos")
        x = torch.randn(1, 4, 32, device=device)
        rope = torch.polar(
            torch.ones(1, 4, 1, 8, device=device),
            torch.randn(1, 4, 1, 8, device=device),
        )
        a = old(x, x, x, rope, update_cache=flag)
        b = new(x, x, x, rope, update_cache=flag)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        for key in ("mask", "id", "is_pred"):
            torch.testing.assert_close(
                old.attn_caches["pos"][key], new.attn_caches["pos"][key], rtol=0, atol=0
            )
    new.clear_cache("pos")
    assert new.attn_caches["pos"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kernel", ["chunk", "recurrent"])
@pytest.mark.parametrize("tokens", [16, 128])
def test_fla_state_matches_reference_and_input_immutable(kernel, tokens):
    _q, k, v = inputs("cuda", torch.bfloat16, tokens, 2, 128)
    m = GDNHistory(2, 128, kernel=kernel).cuda()
    ref = GDNHistory(2, 128, kernel="reference").cuda()
    initial = torch.randn(1, 2, 128, 128, device="cuda") * 0.01
    saved = initial.clone()
    with torch.no_grad():
        state = m.update_state(k, v, initial)
        expected = ref.update_state(k, v, initial)
    torch.testing.assert_close(initial, saved, rtol=0, atol=0)
    torch.testing.assert_close(state, expected, rtol=0.03, atol=0.006)
    assert state.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [16, 128])
def test_fused_read_matches_fp32_reference(tokens):
    q, k, v = inputs("cuda", torch.bfloat16, tokens, 24, 128)
    m = GDNHistory(24, 128, read_kernel="torch").cuda()
    state = torch.randn(1, 24, 128, 128, device="cuda") * 0.1
    with torch.no_grad():
        local = local_softmax(q, k, v)
        ref = m.read(q, state, local)
        m.read_kernel = "triton"
        got = m.read(q, state, local)
    torch.testing.assert_close(got, ref, rtol=0.008, atol=0.008)


def test_local_matches_dense_without_history_and_allocates_no_cache():
    torch.manual_seed(17)
    dense = WanAttention(32, heads=2, dim_head=16, attn_mode="torch")
    local = WanAttention(32, heads=2, dim_head=16, attn_mode="torch")
    local.load_state_dict(dense.state_dict())
    local.history_backend = "local"
    local.init_kv_cache("pos", 12, 2, 16, "cpu", torch.float32, 1)
    x = torch.randn(1, 4, 32)
    rope = torch.polar(torch.ones(1, 4, 1, 8), torch.randn(1, 4, 1, 8))
    for flag in (0, 1, 2):
        torch.testing.assert_close(
            local(x, x, x, rope, update_cache=flag),
            dense(x, x, x, rope, update_cache=flag),
            rtol=0,
            atol=0,
        )
    assert local.attn_caches == {}
    local.clear_pred_cache("pos")
    local.clear_cache("pos")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fla_gate_gradients_match_reference():
    q, k, v = inputs("cuda", torch.bfloat16, 16, 2, 128)
    fla = GDNHistory(2, 128, kernel="chunk").cuda()
    reference = GDNHistory(2, 128, kernel="reference").cuda()
    target = torch.randn_like(v)
    for memory in (fla, reference):
        state = memory.update_state(k, v)
        prediction = memory.read(q, state, local_softmax(q, k, v))
        loss = (prediction.float() - target.float()).square().mean()
        loss.backward()
    for name, parameter in fla.named_parameters():
        expected = dict(reference.named_parameters())[name].grad
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
        torch.testing.assert_close(parameter.grad, expected, rtol=0.08, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_replicated_gates_match_fsdp_compute(tmp_path):
    import copy

    from wan_va.distributed.fsdp import shard_model

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = WanAttention(256, heads=2, dim_head=128).cuda().bfloat16()
            self.attn1.history_backend = "local_gdn"
            self.attn1.history_memory = GDNHistory(2, 128, read_kernel="triton").cuda()
            self.attn2 = torch.nn.Linear(256, 256).cuda().bfloat16()
            self.ffn = torch.nn.Linear(256, 256).cuda().bfloat16()

        def forward(self, x, flag):
            return self.ffn(self.attn2(self.attn1(x, x, x, None, update_cache=flag)))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block(), Block()])

        def forward(self, x, flag):
            for block in self.blocks:
                x = block(x, flag)
            return x

    torch.distributed.init_process_group(
        "nccl", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1
    )
    try:
        torch.manual_seed(31)
        original = Model().requires_grad_(False)
        optimized = copy.deepcopy(original)
        ignored = {
            p for b in optimized.blocks for p in b.attn1.history_memory.parameters()
        }
        for parameter in ignored:
            parameter.data = parameter.data.bfloat16()
        shard_model(original)
        shard_model(optimized, ignored_params=ignored)
        x = torch.randn(2, 16, 256, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            for flag in (2, 0, 1, 0, 1):
                expected = original(x, flag)
                actual = optimized(x, flag)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for b1, b2 in zip(original.blocks, optimized.blocks):
                    a = b1.attn1.history_memory.snapshot("pos").active
                    b = b2.attn1.history_memory.snapshot("pos").active
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()
