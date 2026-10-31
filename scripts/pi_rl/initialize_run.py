#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
"""Create a fresh run manifest and freeze its evaluation scripts before installation."""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
from pathlib import Path

from prepare_run import OFFICIAL_COMMIT, SFT_REVISION, prepare, sha256, write_json
from training_policy import budget_overrides


def initialize(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(
            f"Expected official commit {OFFICIAL_COMMIT}; found {commit}"
        )
    assets = args.asset_root.resolve()
    for name in ["assets", "bddl_files/libero_spatial", "init_files/libero_spatial"]:
        if not (assets / name).is_dir():
            raise RuntimeError(f"Missing simulator input directory: {assets / name}")
    info_path = repo / "docs/pi_rl/provenance/baseline/manifests/hf_model_info.json"
    info = json.loads(info_path.read_text())
    if info["sha"] != SFT_REVISION:
        raise RuntimeError("The bundled model metadata has an unexpected revision")
    run = args.run.resolve()
    state = {
        "run_id": run.name,
        "run_root": str(run),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rlinf_repo": str(repo),
        "rlinf_commit": commit,
        "environment": str(args.environment.resolve()),
        "asset_root": str(assets),
        "sft_model": info["repo_id"],
        "sft_revision": SFT_REVISION,
        "sft_path": str(args.sft.resolve()),
        "scope": ["libero_spatial"],
        "save_video": False,
        "openpi_data_home": str(run / "runtime/model_assets/.cache/openpi"),
        "status": "initialized_not_validated",
        "training_policy": {
            "outer_steps": args.outer_steps,
            "save_interval": args.save_interval,
            "cloud_preflight": "separate",
            "local_lifecycle": "same_snapshot",
        },
    }
    budget_overrides(state)
    if args.volc_config:
        cloud = json.loads(args.volc_config.read_text())
        required = {"queue_id", "flavor", "image_url", "mount_path", "vepfs_id"}
        if set(cloud) != required or not all(
            isinstance(x, str) and x for x in cloud.values()
        ):
            raise ValueError(f"Volc configuration requires exactly {sorted(required)}")
        state["volc"] = cloud
    run.mkdir(parents=True, exist_ok=False)
    for name in [
        "manifests",
        "configs",
        "reports",
        "train",
        "eval/sft",
        "eval/rl",
        "runtime/libero_config",
        "runtime/tmp",
        "runtime/torch_extensions",
    ]:
        (run / name).mkdir(parents=True, exist_ok=True)
    write_json(run / "manifests/run.json", state)
    shutil.copy2(info_path, run / "manifests/hf_model_info.json")
    frozen = run / "scripts/eval_snapshot"
    frozen.mkdir(parents=True)
    for script in Path(__file__).resolve().parent.glob("*.py"):
        shutil.copy2(script, frozen / script.name)
    write_json(
        run / "manifests/evaluation_source_snapshot.json",
        {
            "directory": str(frozen),
            "files": {p.name: sha256(p) for p in sorted(frozen.glob("*.py"))},
        },
    )
    prepare(run, state)
    for script in frozen.glob("*.py"):
        script.chmod(0o444)
    print(json.dumps({"run": str(run), "initialized": True, "validated": False}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--sft", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--outer-steps", type=int, default=200)
    parser.add_argument("--save-interval", type=int, default=40)
    parser.add_argument("--volc-config", type=Path)
    initialize(parser.parse_args())


if __name__ == "__main__":
    main()
