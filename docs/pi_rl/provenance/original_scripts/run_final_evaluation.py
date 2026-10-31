#!/usr/bin/env python3
"""Evaluate the final official-budget checkpoint with the frozen SFT protocols."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
from pathlib import Path

from accept_sft import inspect
from launch_training import require_passed
from prepare_run import sha256, write_json
from prepare_training import verify_snapshot


def final_checkpoint(run: Path, cfg, lifecycle: dict) -> Path:
    if cfg.runner.max_epochs != 1000 or cfg.runner.max_steps != -1:
        raise RuntimeError(
            "Final evaluation requires the frozen official 1000-step budget"
        )
    expected = (
        Path(cfg.runner.logger.log_path)
        / cfg.runner.logger.experiment_name
        / "checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt"
    ).resolve()
    if Path(lifecycle["checkpoint"]["path"]).resolve() != expected:
        raise RuntimeError(
            "The lifecycle report selects a checkpoint other than the final official-budget checkpoint"
        )
    if not expected.is_file() or sha256(expected) != lifecycle["checkpoint"]["sha256"]:
        raise RuntimeError("The final official-budget checkpoint is missing or changed")
    return expected


def main() -> None:
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "manifests/training_source_snapshot.json").read_text())
    snapshot = Path(manifest["directory"])
    verify_snapshot(snapshot, manifest["files"])
    if Path(__file__).resolve().parent != snapshot / "harness":
        raise RuntimeError("Run final evaluation from the frozen harness")
    cfg_path = run / "configs/formal.yaml"
    if sha256(cfg_path) != manifest["config_sha256"]["formal"]:
        raise RuntimeError("The formal configuration changed")
    lifecycle = require_passed(
        run / "train/formal/reports/lifecycle.json", tree=manifest["tree_sha256"]
    )
    checkpoint = final_checkpoint(run, OmegaConf.load(cfg_path), lifecycle)
    admission = inspect(run, verify_weights=True)
    if not admission["passed"]:
        raise RuntimeError(
            f"Original SFT/input provenance changed: {admission['status']}"
        )
    if args.check_only:
        print(
            json.dumps(
                {"admitted_for_final_evaluation": True, "checkpoint": str(checkpoint)}
            )
        )
        return
    record_path = run / "reports/final_evaluation_launch.json"
    if record_path.exists():
        raise RuntimeError(
            "A final evaluation attempt already exists; inspect its process and outputs"
        )
    state = json.loads((run / "manifests/run.json").read_text())
    python = state["environment"] + "/bin/python"
    view = run / "runtime/final_evaluation_view"
    (view / "manifests").mkdir(parents=True, exist_ok=False)
    (view / "runtime").mkdir()
    (view / "reports").mkdir()
    # Wrapper bytes, assets, weights, and protocol settings stay fixed; the view pins the source location.
    state["rlinf_repo"] = str(snapshot / "official")
    state["run_root"] = str(view)
    write_json(view / "manifests/run.json", state)
    (view / "manifests/sft_files.json").symlink_to(run / "manifests/sft_files.json")
    (view / "runtime/libero_config").symlink_to(
        run / "runtime/libero_config", target_is_directory=True
    )
    (view / "eval").symlink_to(run / "eval", target_is_directory=True)
    for name in [
        "runtime_gpu0.json",
        "runtime_gpu1.json",
        "native_eval_readiness.json",
    ]:
        (view / "reports" / name).symlink_to(run / "reports" / name)
    record = {
        "pid": os.getpid(),
        "status": "starting",
        "view": str(view),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": lifecycle["checkpoint"]["sha256"],
        "source_tree_sha256": manifest["tree_sha256"],
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "protocols": {
            "recipe-240": {"seed": 0, "episodes": 500},
            "standalone-220": {"seed": 7, "episodes": 500},
        },
    }
    write_json(record_path, record)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="0,1")
    try:
        for stage in ["smoke", "full"]:
            command = [
                python,
                str(snapshot / "evaluation/run_evaluation.py"),
                "--run",
                str(view),
                "--stage",
                stage,
                "--kind",
                "rl",
                "--checkpoint",
                str(checkpoint),
            ]
            record.update(status=f"running_{stage}", command=command)
            write_json(record_path, record)
            with (run / f"reports/rl_{stage}_driver.log").open("x") as log:
                process = subprocess.Popen(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT
                )
                record["driver_pid"] = process.pid
                write_json(record_path, record)
                returncode = process.wait()
            if returncode:
                raise RuntimeError(
                    f"Final {stage} evaluation failed with exit {returncode}"
                )
        record["status"] = "writing_paired_comparison"
        write_json(record_path, record)
        subprocess.run(
            [python, str(snapshot / "harness/compare_results.py"), "--run", str(run)],
            env=env,
            check=True,
        )
        record["status"] = "evaluations_and_paired_comparison_completed"
    except BaseException as error:
        record.update(
            status="needs_diagnosis", error=f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        record["updated_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
        write_json(record_path, record)


if __name__ == "__main__":
    main()
