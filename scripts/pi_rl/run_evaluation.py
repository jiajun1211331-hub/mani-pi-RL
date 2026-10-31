#!/usr/bin/env python3
"""Schedule two resident policy shards and validate their exact episode coverage."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from summarize_eval import summarize


def require_passed(path: Path) -> None:
    if not path.is_file() or json.loads(path.read_text()).get("passed") is not True:
        raise RuntimeError(f"GPU rendering/model-load gate has not passed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=["smoke", "full"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--kind", choices=["sft", "rl"], default="sft")
    args = parser.parse_args()
    run = args.run.resolve()
    scripts = Path(__file__).resolve().parent
    if (args.kind == "rl") != (args.checkpoint is not None):
        parser.error(
            "RL evaluation requires --checkpoint; SFT evaluation must use the original SFT"
        )
    for gpu in [0, 1]:
        require_passed(run / f"reports/runtime_gpu{gpu}.json")
    if args.stage == "full":
        readiness = run / "reports/native_eval_readiness.json"
        if (
            not readiness.is_file()
            or json.loads(readiness.read_text()).get("ready_for_native_measurement")
            is not True
        ):
            raise RuntimeError(
                f"Native evaluation readiness review has not passed: {readiness}"
            )
    trials = 1 if args.stage == "smoke" else 50
    for protocol, seed in [("recipe-240", 0), ("standalone-220", 7)]:
        root = run / f"eval/{args.kind}/{args.stage}/{protocol}"
        if args.stage == "full":
            prior = run / f"eval/{args.kind}/smoke/{protocol}/summary.json"
            if not prior.exists() or not json.loads(prior.read_text()).get("complete"):
                raise RuntimeError(f"Ten-episode smoke has not passed: {prior}")
        root.mkdir(parents=True, exist_ok=True)
        paths = [root / f"episodes_gpu{gpu}.jsonl" for gpu in [0, 1]]
        if any(path.exists() for path in paths):
            raise RuntimeError(
                f"Evaluation output exists; preserve it and select a new attempt directory: {root}"
            )
        processes = []
        streams = []
        try:
            for gpu, tasks in [(0, list(range(5))), (1, list(range(5, 10)))]:
                env = os.environ.copy()
                env.update(
                    CUDA_VISIBLE_DEVICES=str(gpu),
                    MUJOCO_EGL_DEVICE_ID=str(gpu),
                    TOKENIZERS_PARALLELISM="false",
                    OMP_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    XLA_PYTHON_CLIENT_PREALLOCATE="false",
                )
                command = [
                    sys.executable,
                    str(scripts / "evaluate_spatial.py"),
                    "--run",
                    str(run),
                    "--protocol",
                    protocol,
                    "--task-ids",
                    *map(str, tasks),
                    "--trials",
                    str(trials),
                    "--output",
                    str(paths[gpu]),
                ]
                if args.checkpoint:
                    command += ["--checkpoint", str(args.checkpoint.resolve())]
                stream = (root / f"gpu{gpu}.log").open("w")
                streams.append(stream)
                process = subprocess.Popen(
                    command, env=env, stdout=stream, stderr=subprocess.STDOUT
                )
                processes.append(process)
                (root / f"gpu{gpu}.process.json").write_text(
                    json.dumps(
                        {
                            "pid": process.pid,
                            "command": command,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n"
                )
                print(f"{protocol}: GPU {gpu} worker PID {process.pid}", flush=True)
            outcomes = [process.wait() for process in processes]
        finally:
            for stream in streams:
                stream.close()
        if any(outcomes):
            raise SystemExit(
                f"Evaluation shard failed: return codes {outcomes}; inspect {root}"
            )
        records = [
            json.loads(line)
            for path in paths
            for line in path.read_text().splitlines()
            if line
        ]
        report = summarize(records, list(range(10)), trials, protocol, seed)
        report["source_episode_files"] = list(map(str, paths))
        (root / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        if not report["complete"]:
            raise SystemExit(f"Incomplete evaluation: {root}")
        print(
            json.dumps(
                {k: report[k] for k in ["protocol", "n", "successes", "success_rate"]}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
