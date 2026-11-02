# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify one completed native mani_v1 train/val/save and reload for inference."""

import argparse
import json
import math
import sys
import time
from pathlib import Path


def read_events(path: Path) -> list[dict]:
    """Read the contemporaneous scalar events from one worker."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def validation_metrics(log_path: Path) -> dict:
    """Read native environment success metrics from real TensorBoard events."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    values = {}
    for path in (log_path / "tensorboard").glob("events.out.tfevents.*"):
        events = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
        for tag in events.Tags().get("scalars", []):
            if tag.startswith("eval/"):
                values.setdefault(tag, []).extend(
                    {"step": row.step, "value": row.value}
                    for row in events.Scalars(tag)
                )
    if not values or not any("success" in key for key in values):
        raise RuntimeError("Native validation success metrics are missing.")
    if not all(math.isfinite(row["value"]) for rows in values.values() for row in rows):
        raise RuntimeError("Native validation contains nonfinite metrics.")
    return values


def verify(run: Path, training_returncode: int) -> dict:
    """Require genuine updates, source reports, val, strict reload, and GPU infer."""
    if training_returncode != 0:
        raise RuntimeError(f"Native training exited with {training_returncode}.")
    exit_record = run / "runtime/training_exit.json"
    process_exit = json.loads(exit_record.read_text()) if exit_record.exists() else None
    if process_exit is not None and process_exit["returncode"] != training_returncode:
        raise RuntimeError("Recorded native training return code differs.")
    root = run / "source"
    sys.path.insert(0, str(root))
    import torch
    from omegaconf import OmegaConf
    from torch.distributed.checkpoint import FileSystemReader

    from rlinf.algorithms.manifold.audit import replay_advantages
    from rlinf.algorithms.manifold.snapshot import sha256, verify_snapshot
    from rlinf.models import get_model
    from rlinf.scheduler import Worker

    started = time.monotonic()
    manifest = verify_snapshot(run)
    cfg = OmegaConf.load(run / "configs/train.yaml")
    enabled = cfg.algorithm.manifold.mode != "disabled"
    observed = run / "runtime/observer"
    actors, rollouts = {}, {}
    for rank in range(2):
        rows = read_events(observed / f"actor_rank{rank}.jsonl")
        init = [row for row in rows if row["event"] == "initialized"]
        updates = [row for row in rows if row["event"] == "optimizer_step"]
        completed = [row for row in rows if row["event"] == "training_completed"]
        saved = [row for row in rows if row["event"] == "checkpoint_saved"]
        advantage = [row for row in rows if row["event"] == "advantages_ready"]
        if len(init) != 1 or not init[0]["optimizer_topology_exact"]:
            raise RuntimeError(
                f"Actor {rank} optimizer does not own exactly trainable parameters."
            )
        if (
            len(completed) != cfg.runner.max_epochs
            or not updates
            or not all(row["finite"] for row in updates + completed)
        ):
            raise RuntimeError(f"Actor {rank} finite completed updates are missing.")
        if not any(row["grad_norm"] > 0 for row in updates):
            raise RuntimeError(f"Actor {rank} has no nonzero training gradient.")
        if not any(row["global_step"] == cfg.runner.max_epochs for row in saved):
            raise RuntimeError(f"Actor {rank} final checkpoint event is missing.")
        if enabled and not any(
            row["metrics"].get("manifold/advantage/geometry_l2", 0) > 0
            for row in advantage
        ):
            raise RuntimeError(
                f"Actor {rank} saw no nonzero geometry signal from real rollout."
            )
        actors[str(rank)] = {
            "optimizer_updates": len(updates),
            "training": completed,
            "advantages": advantage,
            "memory": [row for row in rows if row["event"] == "training_memory"],
        }
        predictions = read_events(observed / f"rollout_rank{rank}.jsonl")
        train = [row for row in predictions if row.get("mode") == "train"]
        evaluation = [row for row in predictions if row.get("mode") == "eval"]
        if (
            not train
            or not evaluation
            or any(row["hidden_shape"] is not None for row in evaluation)
        ):
            raise RuntimeError(
                f"Rollout {rank} train/eval collection evidence is incomplete."
            )
        if (
            enabled != (train[0]["hidden_shape"] is not None)
            or not train[0]["hidden_detached"]
        ):
            raise RuntimeError(
                f"Rollout {rank} training collection flag/detach differs."
            )
        rollouts[str(rank)] = predictions
    source_reports = [
        json.loads(path.read_text())
        for path in (run / "runtime/source_reports").glob("*.json")
    ]
    for role, minimum in (("driver", 1), ("actor", 2), ("rollout", 2), ("env", 2)):
        selected = [row for row in source_reports if row["role"] == role]
        if len(selected) < minimum or any(
            row["source_tree_sha256"] != manifest["tree_sha256"] for row in selected
        ):
            raise RuntimeError(
                f"Expected frozen {role} source reports from all workers."
            )
    audit = []
    if enabled:
        directory = (
            Path(cfg.runner.logger.log_path)
            / cfg.runner.logger.experiment_name
            / "manifold"
        )
        for path in sorted(directory.glob("rank_*/step_*.jsonl")):
            events = read_events(path)
            batch = events[0]
            replay = replay_advantages(batch)
            update = [row for row in events if row["event"] == "updates"]
            if not update:
                raise RuntimeError(
                    f"Audit has no matching completed update event: {path}"
                )
            audit.append(
                {
                    "path": str(path),
                    "rank": batch["rank"],
                    "geometry_l2": float(replay.geometry.norm()),
                    "updates": update,
                }
            )
        if len(audit) < 2 or not any(row["geometry_l2"] > 0 for row in audit):
            raise RuntimeError("Scalar replay of real geometry batches is missing.")
    log_path = Path(cfg.runner.logger.log_path)
    validation = validation_metrics(log_path)
    checkpoint_root = (
        log_path
        / cfg.runner.logger.experiment_name
        / f"checkpoints/global_step_{cfg.runner.max_epochs}/actor"
    )
    full = checkpoint_root / "model_state_dict/full_weights.pt"
    metadata = FileSystemReader(checkpoint_root / "dcp_checkpoint").read_metadata()
    keys = list(metadata.state_dict_metadata)
    if not any("optimizers" in key for key in keys) or not any(
        "lr_schedulers" in key for key in keys
    ):
        raise RuntimeError("Native DCP checkpoint lacks optimizer/scheduler metadata.")
    if any("manifold_hidden" in key or "manifold_context" in key for key in keys):
        raise RuntimeError("Collection fields leaked into checkpoint state.")
    model = get_model(cfg.actor.model).cpu()
    expected = model.state_dict()
    weights = torch.load(full, map_location="cpu", weights_only=True, mmap=True)
    if weights.keys() != expected.keys():
        raise RuntimeError("Full checkpoint parameter keys differ from the real model.")
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    changed = []
    for name, baseline in expected.items():
        actual = weights[name]
        if (
            actual.shape != baseline.shape
            or actual.dtype != baseline.dtype
            or not torch.isfinite(actual).all()
        ):
            raise RuntimeError(
                f"Checkpoint tensor shape/dtype/finiteness differs: {name}"
            )
        equal = torch.equal(actual, baseline)
        if name not in trainable and not equal:
            raise RuntimeError(f"Frozen checkpoint tensor changed: {name}")
        if name in trainable and not equal:
            changed.append(name)
    if not changed:
        raise RuntimeError(
            "No action-expert parameters changed in the real checkpoint."
        )
    loaded = model.load_state_dict(weights, strict=True)
    checkpoint = {
        "file": str(full),
        "sha256": sha256(full),
        "bytes": full.stat().st_size,
        "changed_trainable_tensors": changed,
        "frozen_tensors_unchanged": len(expected) - len(trainable),
        "missing_keys": list(loaded.missing_keys),
        "unexpected_keys": list(loaded.unexpected_keys),
        "dcp_entries": len(keys),
        "optimizer_resume_tested": False,
    }
    del weights, expected
    device = torch.device(
        Worker.torch_device_type, Worker.torch_platform.current_device()
    )
    model.eval().to(device)
    obs = torch.load(
        observed / "observation_rank0.pt", map_location="cpu", weights_only=False
    )
    with torch.inference_mode():
        actions, extras = model.predict_action_batch(obs, mode="eval")
    if not torch.isfinite(actions).all() or extras.get("manifold_hidden") is not None:
        raise RuntimeError(
            "Independent reloaded inference is nonfinite or carries hidden."
        )
    report = {
        "passed": True,
        "run": str(run),
        "training_returncode": training_returncode,
        "process_exit": process_exit,
        "mode": cfg.algorithm.manifold.mode,
        "score": cfg.algorithm.manifold.geometry.score,
        "source_tree_sha256": manifest["tree_sha256"],
        "config_sha256": manifest["config_sha256"],
        "verification_harness_sha256": sha256(Path(__file__)),
        "actors": actors,
        "rollouts": rollouts,
        "audit": audit,
        "validation": validation,
        "checkpoint": checkpoint,
        "inference": {
            "actions_shape": list(actions.shape),
            "finite": True,
            "manifold_hidden": False,
            "observation": str(observed / "observation_rank0.pt"),
        },
        "verification_seconds": time.monotonic() - started,
    }
    destination = run / "runtime/lifecycle.json"
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--training-returncode", required=True, type=int)
    args = parser.parse_args()
    report = verify(args.run.resolve(), args.training_returncode)
    from rlinf.utils.logging import get_logger

    get_logger().info(
        "Verified real mani_v1 lifecycle: %s (%s/%s)",
        report["run"],
        report["mode"],
        report["score"],
    )


if __name__ == "__main__":
    main()
