#!/usr/bin/env python3
"""Reject any numerical deviation from the fixed official Spatial GRPO config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from training_policy import budget_overrides


def differences(expected: object, actual: object, prefix: str = "") -> list[dict]:
    result = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() | actual.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in expected or key not in actual:
                result.append(
                    {
                        "path": path,
                        "expected": expected.get(key),
                        "actual": actual.get(key),
                        "kind": "added_or_removed",
                    }
                )
            else:
                result.extend(differences(expected[key], actual[key], path))
    elif type(expected) is not type(actual) or expected != actual:
        result.append(
            {"path": prefix, "expected": expected, "actual": actual, "kind": "value"}
        )
    return result


def audit(expected: dict, actual: dict, allowed: dict) -> dict:
    diff = differences(expected, actual)
    violations = [
        item
        for item in diff
        if item["kind"] != "value"
        or item["path"] not in allowed
        or type(item["actual"]) is not type(allowed[item["path"]])
        or item["actual"] != allowed[item["path"]]
    ]
    return {
        "passed": not violations,
        "differences": diff,
        "violations": violations,
        "allowed_overrides": allowed,
    }


def main() -> None:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        help="Optional resolved YAML produced by the official entrypoint",
    )
    args = parser.parse_args()
    run = args.run.resolve()
    state = json.loads((run / "manifests/run.json").read_text())
    source = json.loads((run / "manifests/official_source.json").read_text())
    repo = Path(state["rlinf_repo"])
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if head != source["commit"]:
        raise RuntimeError(
            f"Official commit changed: expected {source['commit']}, found {head}"
        )
    cfg_root = repo / "examples/embodiment/config"
    for name, digest in source["config_files"].items():
        if hashlib.sha256((cfg_root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Official baseline config changed: {name}")
    os.environ["EMBODIED_PATH"] = str(repo / "examples/embodiment")
    allowed = {
        "actor.model.model_path": state["sft_path"],
        "rollout.model.model_path": state["sft_path"],
        "runner.logger.log_path": str(run / "train/logs"),
        "runner.logger.experiment_name": state["run_id"],
        "env.train.video_cfg.save_video": False,
        "env.eval.video_cfg.save_video": False,
        "env.train.video_cfg.video_base_dir": str(run / "train/logs/video/train"),
        "env.eval.video_cfg.video_base_dir": str(run / "train/logs/video/eval"),
    }
    allowed.update(budget_overrides(state))
    with initialize_config_dir(config_dir=str(cfg_root), version_base="1.1"):
        baseline = compose(config_name="libero_spatial_grpo_openpi_pi05")
        effective = compose(
            config_name="libero_spatial_grpo_openpi_pi05",
            overrides=[
                f"{key}={json.dumps(value, ensure_ascii=False)}"
                for key, value in allowed.items()
            ],
        )
    if args.candidate:
        effective = OmegaConf.load(args.candidate)
    original = OmegaConf.to_container(baseline, resolve=True)
    actual = OmegaConf.to_container(effective, resolve=True)
    report = audit(original, actual, allowed)
    report["official_commit"] = head
    report["budget"] = {
        "outer_rl_steps": actual["runner"]["max_epochs"],
        "raw_trajectory_slots_per_step": 64 * 8,
        "expected_optimizer_updates_per_step": 12,
    }
    report_path = run / "configs/train_config_audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("Official GRPO config audit FAILED: " + str(report_path))
    (run / "configs/train.official.resolved.yaml").write_text(
        OmegaConf.to_yaml(baseline, resolve=True)
    )
    (run / "configs/train.resolved.yaml").write_text(
        OmegaConf.to_yaml(effective, resolve=True)
    )
    (run / "configs/train_overrides.json").write_text(
        json.dumps(allowed, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "passed": True,
                "allowed_differences": len(report["differences"]),
                "outer_rl_steps": report["budget"]["outer_rl_steps"],
            }
        )
    )


if __name__ == "__main__":
    main()
