"""Catch changed eviction sets, speculative visibility, and temporary mutations."""

import pytest
import torch

from wan_va.modules.history_attention import configure_history_attention
from wan_va.modules.model import WanAttention


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
@pytest.mark.parametrize("capacity", [13, 32, 61])
def test_dense_fast_matches_original_through_partial_eviction(device, capacity):
    torch.manual_seed(41)
    old = WanAttention(32, heads=2, dim_head=16, attn_mode="torch").to(device)
    new = WanAttention(32, heads=2, dim_head=16, attn_mode="torch").to(device)
    new.load_state_dict(old.state_dict())
    new.history_backend = "dense_fast"
    for module in (old, new):
        module.init_kv_cache("pos", capacity, 2, 16, device, torch.float32, 1)
    # Varied sizes force partial equal-age eviction, including non-aligned windows.
    for step in range(36):
        for flag, tokens in ((0, 7), (0, 7), (1, 7), (0, 3), (1, 3)):
            x = torch.randn(1, tokens, 32, device=device)
            rope = torch.polar(
                torch.ones(1, tokens, 1, 8, device=device),
                torch.randn(1, tokens, 1, 8, device=device),
            )
            with torch.no_grad():
                expected = old(x, x, x, rope, update_cache=flag)
                actual = new(x, x, x, rope, update_cache=flag)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        for module in (old, new):
            module.clear_pred_cache("pos")
        for tokens in (7, 3):
            x = torch.randn(1, tokens, 32, device=device)
            with torch.no_grad():
                torch.testing.assert_close(
                    new(x, x, x, None, update_cache=2),
                    old(x, x, x, None, update_cache=2),
                    rtol=2e-5,
                    atol=2e-6,
                )
    for module in (old, new):
        module.clear_cache("pos")
        module.init_kv_cache("pos", capacity, 2, 16, device, torch.float32, 1)
    x = torch.randn(1, 7, 32, device=device)
    with torch.no_grad():
        torch.testing.assert_close(
            new(x, x, x, None), old(x, x, x, None), rtol=0, atol=0
        )


def test_hybrid_uses_dense_fast_for_unconverted_layers():
    from types import SimpleNamespace

    model = SimpleNamespace(
        blocks=[
            SimpleNamespace(
                attn1=WanAttention(32, heads=2, dim_head=16, attn_mode="torch")
            )
            for _ in range(4)
        ]
    )
    configure_history_attention(
        model, backend="local_gdn", fraction=0.5, dense_backend="dense_fast"
    )
    assert [b.attn1.history_backend for b in model.blocks] == [
        "dense_fast",
        "local_gdn",
        "dense_fast",
        "local_gdn",
    ]


def test_explicit_rollback_removes_only_its_inserted_slots():
    from wan_va.modules.dense_cache import DenseFastCache

    cache = DenseFastCache(12, 1, 1, 2, "cpu", torch.float32)
    a = torch.ones(1, 3, 1, 2)
    b = torch.ones(1, 2, 1, 2) * 2
    first = cache.append(a, a, False)
    cache.append(b, b, True)
    cache.restore(first)
    assert cache.length == 2
    torch.testing.assert_close(cache.k[:, : cache.length], b, rtol=0, atol=0)
    cache.clear_pred()
    assert cache.length == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_robotwin_bf16_tokens_and_eviction_sets_match_original():
    from wan_va.modules.dense_cache import DenseFastCache
    from wan_va.modules.model import custom_sdpa

    torch.manual_seed(91)
    cap = 9792
    old = WanAttention(8, heads=1, dim_head=8, attn_mode="torch").cuda()
    old.init_kv_cache("pos", cap, 24, 128, "cuda", torch.bfloat16, 2)
    fast = DenseFastCache(cap, 2, 24, 128, "cuda", torch.bfloat16)
    for chunk in range(40):
        for flag, tokens in ((0, 240), (1, 240), (0, 32), (1, 32), (2, 240), (2, 32)):
            if flag == 2 and tokens == 240:
                old.clear_pred_cache("pos")
                fast.clear_pred()
            q, k, v = [
                torch.randn(2, tokens, 24, 128, device="cuda", dtype=torch.bfloat16)
                for _ in range(3)
            ]
            slots = old.update_cache("pos", k, v, flag == 1)
            c = old.attn_caches["pos"]
            valid = c["mask"].nonzero().squeeze(-1)
            expected = custom_sdpa(q, c["k"][:, valid], c["v"][:, valid])
            actual = fast.attend(q, k, v, flag, custom_sdpa)
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
            if flag == 0:
                old.restore_cache("pos", slots)
            valid = c["mask"].nonzero().squeeze(-1)
            assert sorted(fast.slots) == valid.cpu().tolist()
            order = torch.tensor(
                sorted(range(fast.length), key=lambda i: fast.slots[i]), device="cuda"
            )
            torch.testing.assert_close(
                fast.k[:, : fast.length].index_select(1, order),
                c["k"][:, valid],
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("backend", ["full", "dense_fast", "local_gdn"])
def test_benchmark_snapshot_restores_history_after_eviction(backend):
    from types import SimpleNamespace

    from benchmarks.robotwin_cache.state import restore, snapshot

    a = WanAttention(32, heads=2, dim_head=16, attn_mode="torch")
    model = SimpleNamespace(blocks=[SimpleNamespace(attn1=a)])
    configure_history_attention(model, backend=backend, kernel="reference")
    a.init_kv_cache("pos", 13, 2, 16, "cpu", torch.float32, 1)
    x = torch.randn(1, 7, 32)
    with torch.no_grad():
        a(x, x, x, None, update_cache=2)
        saved = snapshot(model)
        expected = a(x, x, x, None)
        for _ in range(4):
            a(x * 3, x * 3, x * 3, None, update_cache=1)
        a.clear_pred_cache("pos")
        restore(model, saved)
        actual = a(x, x, x, None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dense_fast_cache_names_are_independent():
    a = WanAttention(32, heads=2, dim_head=16, attn_mode="torch")
    a.history_backend = "dense_fast"
    for name in ["pos", "neg"]:
        a.init_kv_cache(name, 32, 2, 16, "cpu", torch.float32, 1)
    x = torch.randn(1, 7, 32)
    with torch.no_grad():
        a(x, x, x, None, update_cache=2, cache_name="neg")
        expected = a(x, x, x, None, cache_name="neg")
        a(x * 3, x * 3, x * 3, None, update_cache=1, cache_name="pos")
        a.clear_cache("pos")
        actual = a(x, x, x, None, cache_name="neg")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert a.attn_caches["pos"] is None


def test_temporary_dense_fast_hot_path_does_not_scan_or_gather():
    from wan_va.modules.dense_cache import DenseFastCache
    from wan_va.modules.model import custom_sdpa

    cache = DenseFastCache(32, 1, 2, 16, "cpu", torch.float32)
    q, k, v = [torch.randn(1, 4, 2, 16) for _ in range(3)]
    cache.append(k, v, False)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        for _ in range(4):
            cache.attend(q, k, v, 0, custom_sdpa)
    names = {e.key for e in profile.key_averages()}
    assert not names.intersection(
        {
            "aten::nonzero",
            "aten::argsort",
            "aten::index",
            "aten::index_select",
            "aten::_local_scalar_dense",
        }
    )
    assert cache.length == 4
