"""Recovery must stop a real orphaned workload before resuming evaluation."""

import importlib.util
import subprocess
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "gdn_isolated", Path(__file__).resolve().parents[2] / "benchmarks/gdn/isolated.py"
)
isolated = importlib.util.module_from_spec(spec)
spec.loader.exec_module(isolated)


def test_recovery_kills_owned_experiment_before_resuming(monkeypatch):
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(90)"], start_new_session=True
    )
    recorded = {"pid": child.pid, "start_ticks": "test-owned-child", "state": "S"}

    # The Linux procfs identity adapter is the only platform-specific boundary;
    # process creation, signals, termination and ordering remain real on macOS.
    def identity(pid):
        assert pid == child.pid
        if child.poll() is not None:
            raise ProcessLookupError(pid)
        return recorded

    resumed = []

    def resume(items):
        assert child.poll() is not None, (
            "evaluation resumed while benchmark still alive"
        )
        resumed.extend(items)

    monkeypatch.setattr(isolated, "identity", identity)
    monkeypatch.setattr(isolated, "resume", resume)
    try:
        isolated.recover({"experiment": recorded, "processes": [{"pid": 123}]})
        assert resumed == [{"pid": 123}]
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


def test_recovery_does_not_kill_reused_pid(monkeypatch):
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(90)"], start_new_session=True
    )
    monkeypatch.setattr(
        isolated,
        "identity",
        lambda pid: {"pid": pid, "start_ticks": "new-process", "state": "S"},
    )
    monkeypatch.setattr(isolated, "resume", lambda items: None)
    try:
        isolated.recover(
            {
                "experiment": {"pid": child.pid, "start_ticks": "old-process"},
                "processes": [],
            }
        )
        assert child.poll() is None
    finally:
        child.kill()
        child.wait()
