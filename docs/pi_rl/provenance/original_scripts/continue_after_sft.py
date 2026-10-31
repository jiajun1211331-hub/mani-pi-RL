#!/usr/bin/env python3
"""Wait for the existing SFT evaluator, accept its result, then test local training."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import time
from pathlib import Path

from prepare_run import write_json
from prepare_training import verify_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--driver-pid", required=True, type=int)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "manifests/training_source_snapshot.json").read_text())
    snapshot = Path(manifest["directory"])
    verify_snapshot(snapshot, manifest["files"])
    state = json.loads((run / "manifests/run.json").read_text())
    python = state["environment"] + "/bin/python"
    if args.attempt < 1:
        parser.error("--attempt must be positive")
    report_name = (
        "sft_followup" if args.attempt == 1 else f"sft_followup_r{args.attempt}"
    )
    report_path = run / "reports" / f"{report_name}.json"
    if report_path.exists():
        raise RuntimeError(
            "A follow-up attempt already exists; inspect its PID and log"
        )
    report = {
        "pid": os.getpid(),
        "evaluation_driver_pid": args.driver_pid,
        "status": "waiting_for_existing_evaluator",
        "source_tree_sha256": manifest["tree_sha256"],
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "scope": "Accept the existing fixed 500-per-protocol SFT evaluation and run the separate local GPU lifecycle only. No cloud submission or task termination.",
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    try:
        while not all(
            (run / f"eval/sft/full/{protocol}/summary.json").exists()
            for protocol in ["recipe-240", "standalone-220"]
        ):
            command = Path(f"/proc/{args.driver_pid}/cmdline")
            if not command.exists() or b"run_evaluation.py" not in command.read_bytes():
                raise RuntimeError(
                    "The existing evaluation driver exited without both completed summaries"
                )
            # The evaluator owns its workers and output files; this monitor only reads.
            report["last_poll_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
            report["protocol_summaries"] = {
                protocol: (run / f"eval/sft/full/{protocol}/summary.json").exists()
                for protocol in ["recipe-240", "standalone-220"]
            }
            write_json(report_path, report)
            time.sleep(30)
        verify_snapshot(snapshot, manifest["files"])
        current = json.loads(
            (run / "manifests/training_source_snapshot.json").read_text()
        )
        if current["tree_sha256"] != manifest["tree_sha256"]:
            raise RuntimeError(
                "The training snapshot changed while waiting; inspect and restart this local watcher against the current snapshot"
            )
        report["status"] = "checking_sft_acceptance"
        write_json(report_path, report)
        subprocess.run(
            [python, str(snapshot / "harness/accept_sft.py"), "--run", str(run)],
            check=True,
        )
        report["status"] = "running_local_gpu_lifecycle"
        write_json(report_path, report)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
        env.pop("MUJOCO_EGL_DEVICE_ID", None)
        subprocess.run(
            [
                python,
                str(snapshot / "harness/launch_training.py"),
                "--run",
                str(run),
                "--phase",
                "local-smoke",
            ],
            env=env,
            check=True,
        )
        report["status"] = "local_gpu_lifecycle_completed"
    except BaseException as error:
        report.update(
            status="needs_diagnosis", error=f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        report["updated_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
        write_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
