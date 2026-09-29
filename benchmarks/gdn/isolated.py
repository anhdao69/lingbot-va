"""Run a bounded GPU experiment while retaining Day-1 evaluation state.

The owned evaluation processes on --port are SIGSTOPped, never terminated.
A separate watchdog resumes them if this controller disappears. A timeout
terminates only the new experiment process group, then resumes evaluation.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": fields[19], "state": fields[0]}


def resume(items):
    for item in items:
        try:
            if identity(item["pid"])["start_ticks"] == item["start_ticks"]:
                os.kill(item["pid"], signal.SIGCONT)
        except (FileNotFoundError, ProcessLookupError):
            pass


def save_state(path, data):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def recover(data):
    """Verify and stop the experiment before allowing Day-1 to run again."""
    experiment = data.get("experiment")
    if experiment:
        try:
            current = identity(experiment["pid"])
        except (FileNotFoundError, ProcessLookupError):
            current = None
        if current and current["start_ticks"] == experiment["start_ticks"]:
            # The workload owns this session/process group. A killed controller
            # cannot perform graceful cleanup, so recovery stops the whole group.
            try:
                os.killpg(experiment["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 10
            while True:
                try:
                    current = identity(experiment["pid"])
                except (FileNotFoundError, ProcessLookupError):
                    break
                if (
                    current["state"] == "Z"
                    or current["start_ticks"] != experiment["start_ticks"]
                ):
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "Experiment did not exit; refusing concurrent evaluation"
                    )
                time.sleep(0.05)
    resume(data["processes"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="30056")
    p.add_argument("--gpu", default="0")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--log", required=True)
    p.add_argument("--watchdog", action="store_true")
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    log = Path(a.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    state = log.with_suffix(".pause.json")
    if a.watchdog:
        time.sleep(a.timeout + 30)
        data = json.loads(state.read_text())
        if not data.get("resumed"):
            recover(data)
            data["resumed"] = time.time()
            save_state(state, data)
        return
    items = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            args = (path / "cmdline").read_bytes().decode().split("\0")
            if "torch.distributed.run" in args or "--port" not in args:
                continue
            if args[args.index("--port") + 1] != a.port:
                continue
            if not ({".eval_setup/server.py", ".eval_setup/client.py"} & set(args)):
                continue
            items.append(identity(int(path.name)))
        except (OSError, UnicodeError):
            pass
    assert all(i["state"] not in ("T", "t") for i in items), (
        "Another experiment already paused evaluation"
    )
    data = {"processes": items, "started": time.time(), "command": a.command}
    save_state(state, data)
    watcher = subprocess.Popen(
        [
            sys.executable,
            __file__,
            "--watchdog",
            "--timeout",
            str(a.timeout),
            "--log",
            str(log),
        ],
        start_new_session=True,
    )
    child = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for item in items:
            os.kill(item["pid"], signal.SIGSTOP)
        print("PAUSED", items, flush=True)
        time.sleep(3)
        cmd = a.command[1:] if a.command and a.command[0] == "--" else a.command
        with log.open("w") as f:
            gate_read, gate_write = os.pipe()
            # The child cannot touch CUDA until its verified identity is durable.
            # If the controller dies before writing "1", EOF makes it exit.
            launcher = (
                "import os,sys; fd=int(sys.argv[1]); go=os.read(fd,1); os.close(fd); "
                "sys.exit(125) if go!=b'1' else os.execvp(sys.argv[2],sys.argv[2:])"
            )
            child = subprocess.Popen(
                [sys.executable, "-c", launcher, str(gate_read), *cmd],
                pass_fds=(gate_read,),
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu),
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            os.close(gate_read)
            try:
                data["experiment"] = identity(child.pid)
                save_state(state, data)
                os.write(gate_write, b"1")
            finally:
                os.close(gate_write)
            result = child.wait(timeout=a.timeout)
        if result:
            raise SystemExit(result)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        resume(items)
        data["resumed"] = time.time()
        save_state(state, data)
        watcher.terminate()
        watcher.wait()
        print("RESUMED", items, flush=True)


if __name__ == "__main__":
    main()
