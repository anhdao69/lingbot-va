"""Capture frozen dense attention targets; separate from all speed measurements."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.gdn.common import load_model, parser


class Capture:
    def __init__(self, layer, out):
        self.layer = layer
        self.out = out
        self.chunk = 0

    def prepare(self, q, k, v, flag):
        if flag != 1 or self.chunk not in (1, 5, 10, 14):
            return None
        return {
            "q": q.detach().cpu(),
            "k": k.detach().cpu(),
            "v": v.detach().cpu(),
            "chunk": self.chunk,
            "layer": self.layer,
            "phase": "video" if q.shape[1] == 128 else "action",
        }

    def finish(self, sample, teacher, cache, slots):
        valid = cache["mask"].clone()
        valid[slots] = False
        indices = valid.nonzero(as_tuple=False).flatten()
        order = torch.argsort(cache["id"][indices], stable=True)
        indices = indices[order]
        sample.update(
            teacher=teacher.detach().cpu(),
            history_k=cache["k"][:, indices].cpu(),
            history_v=cache["v"][:, indices].cpu(),
        )
        torch.save(
            sample,
            self.out
            / f"layer{self.layer:02d}_chunk{self.chunk:02d}_{sample['phase']}.pt",
        )


@torch.no_grad()
def main():
    p = parser()
    p.add_argument("--recording", required=True)
    a = p.parse_args()
    assert a.variant == "dense"
    m = load_model(a)
    out = Path(a.out)
    captures = []
    for i, b in enumerate(m.transformer.blocks):
        c = Capture(i, out)
        b.attn1._gdn_capture = c
        captures.append(c)
    files = sorted(
        Path(a.recording).glob("obs_data_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
    )
    obs = [torch.load(p, map_location="cpu", weights_only=False) for p in files]
    prompt = "put both the alphabet soup and the tomato sauce in the basket"
    m.infer({"reset": True, "prompt": prompt})
    for i in range(15):
        for c in captures:
            c.chunk = i
        reply = m.infer({"obs": obs[0][0], "prompt": prompt})
        frames = obs[0] if i == 0 else obs[1 + (i - 1) % (len(obs) - 1)]
        m.infer(
            {
                "obs": frames,
                "compute_kv_cache": True,
                "imagine": False,
                "state": reply["action"],
            }
        )
        print("CAPTURE_CHUNK", i, flush=True)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "layers": 30,
                "chunks": [1, 5, 10, 14],
                "train_chunks": [1, 5, 10],
                "heldout_chunks": [14],
                "source": "frozen dense; recorded RGB replay; post-RoPE attention targets",
                "ordering": "stable cache chunk-id ordering; current tokens excluded; bounded dense history",
            },
            indent=2,
        )
    )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
