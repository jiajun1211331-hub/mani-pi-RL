#!/usr/bin/env python3
"""Run an admitted, immutable official GRPO experiment to its configured end."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from prepare_run import sha256, write_json
from prepare_training import PHASES, verify_snapshot
from training_policy import budget_overrides


def private_ray_environment(private: Path) -> dict:
    token = private / "ray_auth_token"
    if not token.exists():
        descriptor = os.open(token, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(secrets.token_hex(32))
    if token.stat().st_mode & 0o077:
        raise RuntimeError("The private Ray token must have mode 0600")
    return {"RAY_AUTH_MODE": "token", "RAY_AUTH_TOKEN_PATH": str(token)}


def short_temporary_directory(private: Path, ray_root: Path) -> Path:
    disk = private / "tmp"
    disk.mkdir(parents=True, exist_ok=True)
    short = ray_root / "tmp"
    if len(os.fsencode(short)) > 60:
        raise RuntimeError(
            "Temporary directory must leave room for AF_UNIX socket names"
        )
    short.symlink_to(disk.resolve(), target_is_directory=True)
    return short


def collective_probe_command(
    python: Path, snapshot: Path, run: Path, phase: str, world_size: int
) -> list[str]:
    return [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        "--",
        str(snapshot / "harness/gpu_collective_probe.py"),
        "--run",
        str(run),
        "--phase",
        phase,
    ]


def training_entrypoint_command(
    python: Path, snapshot: Path, config: Path, private: Path
) -> list[str]:
    return [
        str(python),
        str(snapshot / "harness/run_official_training.py"),
        str(snapshot / "official/examples/embodiment/train_embodied_agent.py"),
        "--config-path",
        str(config.parent),
        "--config-name",
        config.stem,
        "hydra.run.dir=" + json.dumps(str(private / "work"), ensure_ascii=False),
        "hydra.output_subdir=null",
        "hydra.job.chdir=false",
    ]


def require_passed(path: Path, *, tree: str | None = None) -> dict:
    report = json.loads(path.read_text())
    if report.get("passed") is not True or (
        tree is not None and report.get("source_tree_sha256") != tree
    ):
        raise RuntimeError(
            f"Prerequisite has not passed for this source snapshot: {path}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run, phase = args.run.resolve(), args.phase
    state = json.loads((run / "manifests/run.json").read_text())
    expected_python = Path(state["environment"]) / "bin/python"
    if Path(sys.prefix).resolve() != Path(state["environment"]).resolve():
        os.execv(str(expected_python), [str(expected_python), *sys.argv])
    manifest = json.loads((run / "manifests/training_source_snapshot.json").read_text())
    snapshot = Path(manifest["directory"])
    if Path(__file__).resolve().parent != snapshot / "harness":
        raise RuntimeError("Launch from the frozen harness in source_snapshot/harness")
    verify_snapshot(snapshot, manifest["files"])
    config = run / f"configs/{phase}.yaml"
    if sha256(config) != manifest["config_sha256"][phase]:
        raise RuntimeError(f"Frozen {phase} config changed")
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config)
    selected_budget = budget_overrides(state)
    if phase == "formal":
        for key, value in selected_budget.items():
            if OmegaConf.select(cfg, key) != value:
                raise RuntimeError(f"Frozen configuration violates user budget: {key}")
    if (
        cfg.actor.model.model_path != state["sft_path"]
        or cfg.rollout.model.model_path != state["sft_path"]
    ):
        raise RuntimeError("Frozen configuration and admitted SFT paths differ")
    for gpu in [0, 1]:
        require_passed(run / f"reports/runtime_gpu{gpu}.json")
    require_passed(run / "manifests/actor_parameters.json")
    if phase != "local-smoke":
        from accept_sft import inspect

        admission = inspect(run, verify_weights=True)
        write_json(run / "reports/sft_admission_at_launch.json", admission)
        if not admission["passed"]:
            raise RuntimeError(
                f"SFT admission pending or failed: {admission['status']}"
            )
        if state.get("training_policy", {}).get("local_lifecycle") == "reuse_baseline":
            baseline = Path(state["baseline_run"])
            previous = json.loads(
                (baseline / "manifests/training_source_snapshot.json").read_text()
            )
            verify_snapshot(Path(previous["directory"]), previous["files"])
            critical = {
                "harness/run_official_training.py",
                "harness/training_observer.py",
                "harness/gpu_collective_probe.py",
            }
            source_keys = {
                key
                for key in manifest["files"] | previous["files"]
                if key.startswith(("official/", "evaluation/")) or key in critical
            }
            if any(
                manifest["files"].get(key) != previous["files"].get(key)
                for key in source_keys
            ):
                raise RuntimeError(
                    "Cannot reuse baseline lifecycle after numerical/runtime source changes"
                )
            inherited = require_passed(
                baseline / "train/local-smoke/reports/lifecycle.json",
                tree=previous["tree_sha256"],
            )
            write_json(
                run / "reports/baseline_lifecycle_reuse.json",
                {
                    "passed": True,
                    "baseline_run": str(baseline),
                    "baseline_source_tree_sha256": inherited["source_tree_sha256"],
                    "current_source_tree_sha256": manifest["tree_sha256"],
                    "verified_identical_source_files": len(source_keys),
                    "scope": "User requested direct cloud submission; reuse actual baseline lifecycle for identical official, evaluation, collective and observation code. No new local GPU lifecycle is claimed.",
                },
            )
        else:
            require_passed(
                run / "train/local-smoke/reports/lifecycle.json",
                tree=manifest["tree_sha256"],
            )
    if (
        phase == "formal"
        and state.get("training_policy", {}).get("cloud_preflight", "separate")
        == "separate"
    ):
        require_passed(
            run / "train/cloud-preflight/reports/lifecycle.json",
            tree=manifest["tree_sha256"],
        )
    root = run / "train" / phase
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    if (reports / "launch.json").exists():
        raise RuntimeError(
            f"An attempt already exists; inspect it instead of restarting: {root}"
        )
    # Shared local GPUs remain reserved for the live baseline evaluator.
    if phase == "local-smoke":
        for worker in state.get("eval_processes", {}).values():
            command = Path(f"/proc/{worker['pid']}/cmdline")
            if command.exists() and b"evaluate_spatial.py" in command.read_bytes():
                raise RuntimeError("The local SFT evaluator is still using the GPUs")
        for protocol in ["recipe-240", "standalone-220"]:
            summary = run / "eval/sft/full" / protocol / "summary.json"
            if not summary.exists() or not json.loads(summary.read_text()).get(
                "complete"
            ):
                raise RuntimeError(
                    f"Wait for the baseline evaluator to finish: {protocol}"
                )
    if args.check_only:
        print(
            json.dumps(
                {
                    "admission_gates_passed": True,
                    "phase": phase,
                    "gpu_environment_probe_still_required": True,
                }
            )
        )
        return
    if shutil.disk_usage("/dev/shm").free < 8 << 30:
        raise RuntimeError("Private Ray runtime needs at least 8 GiB free in /dev/shm")
    if shutil.disk_usage(run).free < 200 << 30:
        raise RuntimeError(
            "Insufficient shared filesystem space for checkpoint lifecycle"
        )
    expected_gpus = 2 if phase == "local-smoke" else 8
    official = snapshot / "official"
    env = os.environ.copy()
    temporary = Path(tempfile.mkdtemp(prefix=f"pirl-{phase}-", dir="/dev/shm"))
    private = root / "runtime"
    for name in ["tmp", "torch_extensions", "inductor", "triton", "matplotlib"]:
        (private / name).mkdir(parents=True, exist_ok=True)
    short_tmp = short_temporary_directory(private, temporary)
    env.update(
        {
            "PATH": str(expected_python.parent) + os.pathsep + env.get("PATH", ""),
            "PYTHONPATH": os.pathsep.join([str(official), str(snapshot / "harness")]),
            "PYTHONDONTWRITEBYTECODE": "1",
            "EMBODIED_PATH": str(official / "examples/embodiment"),
            "LIBERO_CONFIG_PATH": str(run / "runtime/libero_config"),
            "DOWNLOAD_DIR": str(run / "runtime/model_assets"),
            "OPENPI_DATA_HOME": str(run / "runtime/model_assets/.cache/openpi"),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "ROBOT_PLATFORM": "LIBERO",
            "LIBERO_TYPE": "standard",
            "TOKENIZERS_PARALLELISM": "false",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TMPDIR": str(short_tmp),
            "TORCH_EXTENSIONS_DIR": str(private / "torch_extensions"),
            "TORCHINDUCTOR_CACHE_DIR": str(private / "inductor"),
            "TRITON_CACHE_DIR": str(private / "triton"),
            "MPLCONFIGDIR": str(private / "matplotlib"),
            "RAY_ADDRESS": "local",
            "RAY_TMPDIR": str(temporary),
            "RAY_USAGE_STATS_ENABLED": "0",
            "RLINF_CODE_WORKING_DIR": "0",
            "RLINF_EXT_MODULE": "training_observer",
            "PIRL_OBSERVER_DIR": str(reports / "observer"),
        }
    )
    env.update(private_ray_environment(private))
    env.pop("RAY_AUTH_TOKEN", None)
    # These belong to the platform's entrypoint, not to the independent local Ray workers.
    for key in [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ]:
        env.pop(key, None)
    os.environ.update(env)
    os.environ.pop("RAY_AUTH_TOKEN", None)
    sys.path.insert(0, str(official))
    from runtime import environment_manifest

    environment = environment_manifest()
    import torch

    if torch.cuda.device_count() != expected_gpus:
        raise RuntimeError(
            f"Expected {expected_gpus} allocated GPUs; found {torch.cuda.device_count()}"
        )
    write_json(reports / "environment.json", environment)
    command = training_entrypoint_command(expected_python, snapshot, config, private)
    launch = {
        "phase": phase,
        "command": command,
        "cwd": str(official),
        "source_tree_sha256": manifest["tree_sha256"],
        "config_sha256": sha256(config),
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "ray_tmpdir": str(temporary),
        "temporary_directory": str(short_tmp),
        "disk_temporary_directory": str(private / "tmp"),
        "ray_address": "local",
        "gpu_count": expected_gpus,
        "status": "probing",
        "formal": phase == "formal",
        "budget": {
            "outer_steps": cfg.runner.max_epochs,
            "parallel_envs": cfg.env.train.total_num_envs,
            "rollout_epochs": cfg.env.train.rollout_epoch,
            "global_batch_size": cfg.actor.global_batch_size,
            "micro_batch_size": cfg.actor.micro_batch_size,
        },
    }
    write_json(reports / "launch.json", launch)
    collective = collective_probe_command(
        expected_python, snapshot, run, phase, expected_gpus
    )
    with (reports / "collective.log").open("x") as stream:
        result = subprocess.run(
            collective,
            env=env,
            cwd=official,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        launch.update(status="probe_failed", returncode=result.returncode)
        write_json(reports / "launch.json", launch)
        raise SystemExit(result.returncode)
    with (reports / "training.log").open("x") as stream:
        process = subprocess.Popen(
            command, env=env, cwd=official, stdout=stream, stderr=subprocess.STDOUT
        )
        launch.update(status="training", pid=process.pid)
        write_json(reports / "launch.json", launch)
        print(
            json.dumps(
                {
                    "phase": phase,
                    "pid": process.pid,
                    "log": str(reports / "training.log"),
                }
            ),
            flush=True,
        )
        returncode = process.wait()
    launch.update(
        status="training_completed" if returncode == 0 else "training_failed",
        returncode=returncode,
        ended_utc=datetime.datetime.now(datetime.UTC).isoformat(),
    )
    write_json(reports / "launch.json", launch)
    if returncode:
        raise SystemExit(returncode)
    subprocess.run(
        [
            str(expected_python),
            str(snapshot / "harness/verify_training.py"),
            "--run",
            str(run),
            "--phase",
            phase,
        ],
        env=env,
        cwd=official,
        check=True,
    )


if __name__ == "__main__":
    main()
