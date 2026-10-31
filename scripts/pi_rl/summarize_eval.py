#!/usr/bin/env python3
"""Validate exact episode coverage before reporting Spatial success rates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson(successes: int, count: int) -> list[float]:
    if not 0 <= successes <= count or count <= 0:
        raise ValueError(
            "Wilson interval requires 0 <= successes <= count and count > 0"
        )
    z = 1.959963984540054
    p = successes / count
    scale = 1 + z * z / count
    center = (p + z * z / (2 * count)) / scale
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / scale
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize(
    records: list[dict], task_ids: list[int], trials: int, protocol: str, seed: int
) -> dict:
    if len(set(task_ids)) != len(task_ids) or not 1 <= trials <= 50:
        raise ValueError("Expected unique task IDs and 1..50 initial states per task")
    expected = {(task, trial) for task in task_ids for trial in range(trials)}
    completed = {}
    errors = []
    identity = set()
    for record in records:
        if record["protocol"] != protocol or record["seed"] != seed:
            raise ValueError("Cannot combine different protocols or seeds")
        if record.get("save_video") is not False:
            raise ValueError("The no-video requirement was not recorded")
        identity.add((record["checkpoint_sha256"], record["wrapper_sha256"]))
        key = (record["task_id"], record["init_state_id"])
        if key not in expected:
            raise ValueError(f"Unexpected episode ID: {key}")
        if record["status"] != "completed":
            errors.append(record)
            continue
        if key in completed:
            raise ValueError(f"Duplicate completed episode ID: {key}")
        if type(record.get("success")) is not bool:
            raise ValueError(f"Success must be a boolean for episode {key}")
        completed[key] = record
    if len(identity) != 1:
        raise ValueError(
            "Cannot combine different checkpoints or evaluation implementations"
        )
    missing = sorted(expected - completed.keys())
    tasks = []
    for task in task_ids:
        rows = [record for (tid, _), record in completed.items() if tid == task]
        successes = sum(x["success"] for x in rows)
        tasks.append(
            {
                "task_id": task,
                "n": len(rows),
                "successes": successes,
                "success_rate": successes / len(rows) if rows else None,
                "wilson95": wilson(successes, len(rows)) if rows else None,
            }
        )
    count = len(completed)
    successes = sum(r["success"] for r in completed.values())
    complete = not missing and not errors
    sr = successes / count if complete and count else None
    return {
        "complete": complete,
        "protocol": protocol,
        "seed": seed,
        "expected_n": len(expected),
        "n": count,
        "successes": successes,
        "success_rate": sr,
        "wilson95": wilson(successes, count) if sr is not None else None,
        "difference_from_paper_sft_pp": 100 * (sr - 0.846) if sr is not None else None,
        "missing_episodes": missing,
        "system_error_count": len(errors),
        "per_task": tasks,
        "checkpoint_sha256": next(iter(identity))[0],
        "wrapper_sha256": next(iter(identity))[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--trials", default=50, type=int)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for path in args.files
        for line in path.read_text().splitlines()
        if line
    ]
    summary = summarize(records, args.task_ids, args.trials, args.protocol, args.seed)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ["complete", "n", "successes", "success_rate", "wilson95"]
            }
        )
    )
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
