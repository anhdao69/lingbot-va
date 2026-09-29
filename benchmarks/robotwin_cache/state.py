"""Host snapshots used OUTSIDE measured intervals to match repeated histories."""

import torch


def snapshot(model):
    result = []
    for block in model.blocks:
        a = block.attn1
        if a.history_memory is not None:
            states = {
                name: type(s)(
                    s.committed.to("cpu", copy=True)
                    if s.committed is not None
                    else None,
                    s.speculative.to("cpu", copy=True)
                    if s.speculative is not None
                    else None,
                    s.committed_tokens,
                    s.speculative_tokens,
                )
                for name, s in a.history_memory.states.items()
            }
            result.append(("gdn", states))
        else:
            caches = {}
            for name, c in a.attn_caches.items():
                if c is None:
                    caches[name] = None
                elif hasattr(c, "length"):
                    caches[name] = {
                        "k": c.k[:, : c.length].cpu().clone(),
                        "v": c.v[:, : c.length].cpu().clone(),
                        "length": c.length,
                        "slots": list(c.slots),
                        "ids": c.ids.copy(),
                        "pred": c.pred.copy(),
                    }
                else:
                    caches[name] = {k: v.cpu().clone() for k, v in c.items()}
            result.append(("dense", caches))
    return result


def restore(model, saved):
    for block, (kind, states) in zip(model.blocks, saved):
        a = block.attn1
        device = a.to_q.weight.device
        if kind == "gdn":
            a.history_memory.states = {
                name: type(s)(
                    s.committed.to(device, copy=True)
                    if s.committed is not None
                    else None,
                    s.speculative.to(device, copy=True)
                    if s.speculative is not None
                    else None,
                    s.committed_tokens,
                    s.speculative_tokens,
                )
                for name, s in states.items()
            }
        else:
            for name, s in states.items():
                if s is None:
                    a.attn_caches[name] = None
                    continue
                c = a.attn_caches[name]
                if "length" in s:
                    c.length = s["length"]
                    c.slots = list(s["slots"])
                    c.ids = s["ids"].copy()
                    c.pred = s["pred"].copy()
                    c.k[:, : c.length].copy_(s["k"])
                    c.v[:, : c.length].copy_(s["v"])
                else:
                    for k, v in s.items():
                        c[k].copy_(v)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def snapshot_policy(model):
    wrappers = [model.streaming_vae, model.streaming_vae_half]
    return {
        "transformer": snapshot(model.transformer),
        "frame": model.frame_st_id,
        "vae": [
            [t.to("cpu", copy=True) if torch.is_tensor(t) else t for t in w.feat_cache]
            if w is not None
            else None
            for w in wrappers
        ],
    }


def restore_policy(model, saved):
    restore(model.transformer, saved["transformer"])
    model.frame_st_id = saved["frame"]
    for w, cache in zip([model.streaming_vae, model.streaming_vae_half], saved["vae"]):
        if w is not None:
            device = next(w.vae.parameters()).device
            w.feat_cache = [
                t.to(device, copy=True) if torch.is_tensor(t) else t for t in cache
            ]
    torch.cuda.synchronize()
