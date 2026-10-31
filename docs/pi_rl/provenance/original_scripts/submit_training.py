#!/usr/bin/env python3
"""Submit one admitted Volc task, preserving uncertain outcomes for reconciliation."""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path

from prepare_run import sha256, write_json
from prepare_training import verify_snapshot


def parse_task_list(output: str) -> list[dict]:
    decoder = json.JSONDecoder()
    for offset, char in enumerate(output):
        if char != "[":
            continue
        try:
            value, _ = decoder.raw_decode(output[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and all(
            isinstance(row, dict) and {"JobId", "JobName", "Status"} <= row.keys()
            for row in value
        ):
            return value
    raise RuntimeError("Volc task-list output contains no valid JSON task list")


def exact_task(rows: list[dict], name: str) -> dict | None:
    matches = [row for row in rows if row["JobName"] == name]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple Volc tasks already use the exact name: {name}")
    return matches[0] if matches else None


def query(volc: str, name: str, root: Path) -> list[dict]:
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    result = subprocess.run(
        [
            volc,
            "ml_task",
            "list",
            "--name",
            name,
            "--status",
            "Queue,Staging,Running,Killing,Success,Failed,Killed,Initialized",
            "--limit",
            "100",
            "--output",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    (root / f"query-{stamp}.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Volc task query failed with exit {result.returncode}")
    return parse_task_list(result.stdout)


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--phase", choices=["cloud-preflight", "formal"], required=True)
    parser.add_argument("--volc", default="/root/.volc/bin/volc")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run, phase = args.run.resolve(), args.phase
    state = json.loads((run / "manifests/run.json").read_text())
    manifest = json.loads((run / "manifests/training_source_snapshot.json").read_text())
    snapshot = Path(manifest["directory"])
    if Path(__file__).resolve().parent != snapshot / "harness":
        raise RuntimeError("Submit from the frozen source_snapshot harness")
    verify_snapshot(snapshot, manifest["files"])
    spec_path = run / f"configs/volc-{phase}.yaml"
    if sha256(spec_path) != manifest["volc_specs_sha256"][phase]:
        raise RuntimeError("Frozen Volc task specification changed")
    subprocess.run(
        [
            state["environment"] + "/bin/python",
            str(snapshot / "harness/launch_training.py"),
            "--run",
            str(run),
            "--phase",
            phase,
            "--check-only",
        ],
        check=True,
    )
    spec = yaml.safe_load(spec_path.read_text())
    if args.check_only:
        print(
            json.dumps(
                {
                    "admitted_for_submission": True,
                    "phase": phase,
                    "task_name": spec["TaskName"],
                    "submitted": False,
                }
            )
        )
        return
    root = run / "train" / phase / "submission"
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "submission.json"
    existing = exact_task(query(args.volc, spec["TaskName"], root), spec["TaskName"])
    if existing is not None:
        receipt = {
            "task_id": existing["JobId"],
            "task_name": existing["JobName"],
            "status": existing["Status"],
            "reconciled_existing": True,
            "source_tree_sha256": manifest["tree_sha256"],
            "spec_sha256": sha256(spec_path),
        }
        write_json(receipt_path, receipt)
        print(json.dumps(receipt))
        return
    if receipt_path.exists():
        raise RuntimeError(
            "A prior submission intent exists but its task is not visible; reconcile it before another submission"
        )
    receipt = {
        "task_name": spec["TaskName"],
        "phase": phase,
        "status": "submission_in_flight",
        "source_tree_sha256": manifest["tree_sha256"],
        "spec_sha256": sha256(spec_path),
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    write_json(receipt_path, receipt)
    with (root / "submit.log").open("x") as stream:
        result = subprocess.run(
            [
                args.volc,
                "ml_task",
                "submit",
                "--conf",
                str(spec_path),
                "--output",
                "json",
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    receipt["submit_cli_returncode"] = result.returncode
    receipt["status"] = "awaiting_task_identity"
    write_json(receipt_path, receipt)
    existing = exact_task(query(args.volc, spec["TaskName"], root), spec["TaskName"])
    if existing is None:
        raise RuntimeError(
            "Submission outcome is uncertain; the intent is preserved. Query the exact task name before considering another submission"
        )
    receipt.update(task_id=existing["JobId"], status=existing["Status"])
    write_json(receipt_path, receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
