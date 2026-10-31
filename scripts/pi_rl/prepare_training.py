#!/usr/bin/env python3
"""Freeze official source, audited phase configs, and concrete Volc task specs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path

from audit_config import audit
from prepare_run import OFFICIAL_COMMIT, sha256, write_json
from training_policy import budget_overrides

PHASES = ("local-smoke", "cloud-preflight", "formal")


def phase_overrides(run: Path, state: dict, phase: str) -> dict:
    if phase not in PHASES:
        raise ValueError(f"Unknown training phase: {phase}")
    log = run / "train/logs" if phase == "formal" else run / "train" / phase / "logs"
    name = state["run_id"] if phase == "formal" else state["run_id"] + "-" + phase
    values = {
        "actor.model.model_path": state["sft_path"],
        "rollout.model.model_path": state["sft_path"],
        "runner.logger.log_path": str(log),
        "runner.logger.experiment_name": name,
        "env.train.video_cfg.save_video": False,
        "env.eval.video_cfg.save_video": False,
        "env.train.video_cfg.video_base_dir": str(log / "video/train"),
        "env.eval.video_cfg.video_base_dir": str(log / "video/eval"),
    }
    if phase == "formal":
        values.update(budget_overrides(state))
    if phase != "formal":
        values.update(
            {
                "runner.max_epochs": 1 if phase == "local-smoke" else 2,
                "runner.save_interval": 1,
                "runner.val_check_interval": 1,
                "env.eval.total_num_envs": 2 if phase == "local-smoke" else 16,
            }
        )
    if phase == "local-smoke":
        values["env.train.total_num_envs"] = 16
    return values


def file_identity(path: Path) -> dict:
    if path.is_symlink():
        return {"kind": "symlink", "target": os.readlink(path)}
    return {"kind": "file", "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_snapshot(root: Path, files: dict) -> None:
    actual = {
        str(p.relative_to(root)): file_identity(p)
        for p in root.rglob("*")
        if (p.is_file() or p.is_symlink())
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
    }
    if actual != files:
        changed = sorted(
            k for k in actual.keys() | files.keys() if actual.get(k) != files.get(k)
        )
        raise RuntimeError(f"Frozen training source changed: {changed[:20]}")


def prepare(run: Path, snapshot_name: str = "source_snapshot") -> None:
    import yaml
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    state = json.loads((run / "manifests/run.json").read_text())
    repo = Path(state["rlinf_repo"])
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD"],
        cwd=repo,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    if head != OFFICIAL_COMMIT:
        raise RuntimeError(f"Expected {OFFICIAL_COMMIT}; found {head}")
    if (
        not snapshot_name.startswith("source_snapshot")
        or Path(snapshot_name).name != snapshot_name
    ):
        raise ValueError(
            "Snapshot name must be a local basename beginning with source_snapshot"
        )
    snapshot = run / snapshot_name
    if snapshot.exists() and any(snapshot.iterdir()):
        raise RuntimeError(f"Source snapshot already contains files: {snapshot}")
    snapshot.mkdir(exist_ok=True)
    archive = run / "manifests/official_source.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", f"--output={archive}", head],
        cwd=repo,
        check=True,
    )
    official = snapshot / "official"
    official.mkdir()
    with tarfile.open(archive) as stream:
        stream.extractall(official, filter="data")
    harness = snapshot / "harness"
    harness.mkdir()
    for script in Path(__file__).resolve().parent.glob("*.py"):
        shutil.copy2(script, harness / script.name)
    shutil.copytree(
        run / "scripts/eval_snapshot",
        snapshot / "evaluation",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    cfg_root = official / "examples/embodiment/config"
    os.environ["EMBODIED_PATH"] = str(official / "examples/embodiment")
    with initialize_config_dir(config_dir=str(cfg_root), version_base="1.1"):
        baseline = compose(config_name="libero_spatial_grpo_openpi_pi05")
        expected = OmegaConf.to_container(baseline, resolve=True)
        for phase in PHASES:
            allowed = phase_overrides(run, state, phase)
            cfg = compose(
                config_name="libero_spatial_grpo_openpi_pi05",
                overrides=[
                    f"{key}={json.dumps(value, ensure_ascii=False)}"
                    for key, value in allowed.items()
                ],
            )
            actual = OmegaConf.to_container(cfg, resolve=True)
            report = audit(expected, actual, allowed)
            if not report["passed"]:
                raise RuntimeError(
                    f"Config audit failed: {phase}: {report['violations']}"
                )
            (run / f"configs/{phase}.yaml").write_text(
                OmegaConf.to_yaml(cfg, resolve=True)
            )
            write_json(run / f"configs/{phase}.audit.json", report)
            if phase == "formal" and actual != OmegaConf.to_container(
                OmegaConf.load(run / "configs/train.resolved.yaml"), resolve=True
            ):
                raise RuntimeError(
                    "Frozen formal recipe differs from the earlier audited recipe"
                )
    files = {
        str(p.relative_to(snapshot)): file_identity(p)
        for p in sorted(snapshot.rglob("*"))
        if p.is_file() or p.is_symlink()
    }
    configs = {phase: sha256(run / f"configs/{phase}.yaml") for phase in PHASES}
    manifest = {
        "official_commit": head,
        "official_archive_sha256": sha256(archive),
        "directory": str(snapshot),
        "files": files,
        "config_sha256": configs,
        "tree_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        "scope": "Official git archive plus observation/gating/launch harness. No numerical source changes. Finite phases are separate experiments; formal uses the audited recipe and any explicitly recorded user budget overrides.",
    }
    cloud = state.get("volc", {})
    for phase in ["cloud-preflight", "formal"]:
        command = [
            state["environment"] + "/bin/python",
            str(harness / "launch_training.py"),
            "--run",
            str(run),
            "--phase",
            phase,
        ]
        spec = {
            "TaskName": (state["run_id"] + "-" + phase).lower(),
            "Description": (
                "Pi05 Spatial official GRPO; "
                + (
                    "finite two-step lifecycle preflight"
                    if phase == "cloud-preflight"
                    else f"{state.get('training_policy', {}).get('outer_steps', 1000)} outer steps; 64 envs x 8 rollout epochs"
                )
            ),
            "Entrypoint": "bash -lc " + shlex.quote("exec " + shlex.join(command)),
            "Framework": "PyTorchDDP",
            "ResourceQueueID": cloud.get("queue_id", "q-20260403140809-n4flj"),
            "UserCodePath": str(
                Path(__file__).resolve().parents[2] / "scripts/volc_empty_code"
            ),
            "Envs": [
                {
                    "Name": "NVIDIA_DRIVER_CAPABILITIES",
                    "Value": "compute,utility,graphics",
                    "IsPrivate": False,
                }
            ],
            "TaskRoleSpecs": [
                {
                    "RoleName": "worker",
                    "RoleReplicas": 1,
                    "Flavor": cloud.get("flavor", "ml.hpcpni2l.28xlarge"),
                }
            ],
            "ImageUrl": cloud.get(
                "image_url",
                "vemlp-cn-beijing.cr.volces.com/preset-images/pytorch:2.10.0-cu12.9.1-py3.13-ubuntu24.04",
            ),
            "Storages": [
                {
                    "Type": "Vepfs",
                    "MountPath": cloud.get("mount_path", "/vepfs-C区"),
                    "ReadOnly": False,
                    "SubPath": "",
                    "VepfsId": cloud.get("vepfs_id", "vepfs-cnbjba6dad11eac7"),
                }
            ],
            "AccessType": "Private",
        }
        (run / f"configs/volc-{phase}.yaml").write_text(
            yaml.safe_dump(spec, allow_unicode=True, sort_keys=False)
        )
    manifest["volc_specs_sha256"] = {
        phase: sha256(run / f"configs/volc-{phase}.yaml")
        for phase in ["cloud-preflight", "formal"]
    }
    manifest_path = run / "manifests/training_source_snapshot.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        backup = (
            run
            / f"manifests/training_source_snapshot.{previous['tree_sha256'][:12]}.json"
        )
        if not backup.exists():
            shutil.copy2(manifest_path, backup)
    write_json(manifest_path, manifest)
    verify_snapshot(snapshot, files)
    for p in snapshot.rglob("*"):
        if p.is_file() and not p.is_symlink():
            p.chmod(0o444)
    print(
        json.dumps(
            {"frozen_files": len(files), "config_sha256": configs, "submitted": False},
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--snapshot-name", default="source_snapshot")
    args = parser.parse_args()
    prepare(args.run.resolve(), args.snapshot_name)


if __name__ == "__main__":
    main()
