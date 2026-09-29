"""Small teacher-forced alignment: optimize only three scalar gates per head."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wan_va.modules.history_attention import GDNHistory, local_softmax


def loss_for(mem, s):
    state = mem.update_state(s["history_k"], s["history_v"])
    pred = mem.read(s["q"], state, s["local"])
    return (pred.float() - s["teacher"].float()).square().mean() / (
        s["teacher"].float().square().mean() + 1e-8
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=36)
    p.add_argument("--lr", type=float, default=0.025)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    states = {}
    rows = []
    begin = time.perf_counter()
    for layer in range(30):
        samples = []
        for file in sorted(Path(a.capture).glob(f"layer{layer:02d}_*.pt")):
            s = torch.load(file, map_location="cuda", weights_only=True)
            s["local"] = local_softmax(s["q"], s["k"], s["v"]).detach()
            samples.append(s)
        assert len(samples) == 8
        train = [s for s in samples if s["chunk"] != 14]
        test = [s for s in samples if s["chunk"] == 14]
        mem = GDNHistory(24, 128, kernel="chunk", read_kernel="torch").cuda()
        opt = torch.optim.Adam(mem.parameters(), lr=a.lr)

        def evaluate(mem=mem, samples=samples):
            with torch.no_grad():
                return [float(loss_for(mem, s)) for s in samples]

        before = evaluate()
        for step in range(a.steps):
            opt.zero_grad(set_to_none=True)
            loss = loss_for(mem, train[step % len(train)])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite alignment loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mem.parameters(), 1.0)
            opt.step()
        after = evaluate()
        states[str(layer)] = {k: v.detach().cpu() for k, v in mem.state_dict().items()}
        for s, b, f in zip(samples, before, after):
            local_loss = float(
                (s["local"].float() - s["teacher"].float()).square().mean()
                / (s["teacher"].float().square().mean() + 1e-8)
            )
            rows.append(
                {
                    "layer": layer,
                    "chunk": s["chunk"],
                    "phase": s["phase"],
                    "heldout": s["chunk"] == 14,
                    "local_nmse": local_loss,
                    "gdn_before_nmse": b,
                    "gdn_after_nmse": f,
                }
            )
        print(
            "ALIGNED",
            layer,
            "train",
            sum(after[:-2]) / 6,
            "last_losses",
            after[-2:],
            flush=True,
        )
        torch.save(states, out / "adapter.pt")
        (out / "alignment.json").write_text(
            json.dumps(
                {
                    "steps_per_layer": a.steps,
                    "lr": a.lr,
                    "parameter_count": 30 * 24 * 3,
                    "original_weights_loaded": False,
                    "original_weights_updated": False,
                    "elapsed_s": time.perf_counter() - begin,
                    "samples": rows,
                },
                indent=2,
            )
        )
        del samples, train, test, mem, opt
    print("ALIGN_DONE", time.perf_counter() - begin, flush=True)


if __name__ == "__main__":
    main()
