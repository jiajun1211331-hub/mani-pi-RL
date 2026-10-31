#!/usr/bin/env python3
"""Verify real training, frozen weights, checkpoint reload, val, and inference."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from prepare_run import sha256, write_json
from prepare_training import PHASES, verify_snapshot


def read_validation_metrics(log_path: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    validation = {}
    # Official MetricLogger writes to log_path/tensorboard, while checkpoints
    # are stored separately under log_path/experiment_name/checkpoints.
    for event_file in (log_path / "tensorboard").rglob("events.out.tfevents.*"):
        events = EventAccumulator(
            str(event_file), size_guidance={"scalars": 0}
        ).Reload()
        for tag in events.Tags().get("scalars", []):
            if tag.startswith("eval/"):
                validation.setdefault(tag, []).extend(
                    {"step": x.step, "value": x.value} for x in events.Scalars(tag)
                )
    return validation


def compare_weights(model, weights: dict) -> dict:
    import torch

    expected = model.state_dict()
    if set(weights) != set(expected):
        raise RuntimeError(
            f"Checkpoint keys differ: missing={sorted(set(expected) - set(weights))}, extra={sorted(set(weights) - set(expected))}"
        )
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    failures, changed = [], []
    for name, baseline in expected.items():
        actual = weights[name]
        if actual.shape != baseline.shape or actual.dtype != baseline.dtype:
            failures.append(f"Shape/dtype changed: {name}")
            continue
        if not torch.isfinite(actual).all():
            failures.append(f"Nonfinite checkpoint tensor: {name}")
        equal = torch.equal(actual, baseline)
        if name not in trainable and not equal:
            failures.append(f"Frozen parameter/buffer changed: {name}")
        if name in trainable and not equal:
            changed.append(name)
    if not changed:
        failures.append("No action-expert tensor changed after training")
    if failures:
        raise RuntimeError("Checkpoint verification failed: " + str(failures[:20]))
    loaded = model.load_state_dict(weights, strict=True)
    return {
        "passed": True,
        "tensor_count": len(weights),
        "changed_trainable_tensors": changed,
        "frozen_tensors_unchanged": len(expected) - len(trainable),
        "missing_keys": list(loaded.missing_keys),
        "unexpected_keys": list(loaded.unexpected_keys),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--phase", choices=PHASES, required=True)
    args = parser.parse_args()
    run, phase = args.run.resolve(), args.phase
    root = run / "train" / phase
    reports = root / "reports"
    manifest = json.loads((run / "manifests/training_source_snapshot.json").read_text())
    snapshot = Path(manifest["directory"])
    verify_snapshot(snapshot, manifest["files"])
    launch = json.loads((reports / "launch.json").read_text())
    if launch["source_tree_sha256"] != manifest["tree_sha256"]:
        raise RuntimeError("Training source and current verification snapshot differ")
    if launch["returncode"] != 0 or launch["status"] != "training_completed":
        raise RuntimeError(
            "The real official training process did not complete successfully"
        )
    sys.path.insert(0, str(snapshot / "official"))
    import torch
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openpi import get_model

    cfg = OmegaConf.load(run / f"configs/{phase}.yaml")
    steps = cfg.runner.max_epochs
    expected_optimizer_steps = (
        cfg.env.train.total_num_envs
        * cfg.env.train.rollout_epoch
        * (
            cfg.env.train.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )
        // cfg.actor.global_batch_size
        * cfg.algorithm.update_epoch
        * steps
    )
    actors = {}
    for rank in range(launch["gpu_count"]):
        path = reports / f"observer/actor_rank{rank}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        init = [x for x in rows if x["event"] == "initialized"]
        updates = [x for x in rows if x["event"] == "optimizer_step"]
        training = [x for x in rows if x["event"] == "training_completed"]
        saves = [x for x in rows if x["event"] == "checkpoint_saved"]
        if (
            len(init) != 1
            or not init[0]["optimizer_topology_exact"]
            or len(updates) != expected_optimizer_steps
            or len(training) != steps
            or not all(x["finite"] for x in updates + training)
            or not any(x["grad_norm"] > 0 for x in updates)
            or not any(x["global_step"] == steps for x in saves)
        ):
            raise RuntimeError(f"Actor {rank} actual lifecycle evidence failed")
        trainable_numel = sum(
            x["numel"] for x in init[0]["parameters"] if x["requires_grad"]
        )
        if trainable_numel != 430_098_464:
            raise RuntimeError(
                f"Unexpected post-FSDP trainable parameter count: {trainable_numel}"
            )
        actors[str(rank)] = {
            "optimizer_updates": len(updates),
            "training_steps": len(training),
            "post_fsdp_trainable_parameters": trainable_numel,
            "optimizer_groups": init[0]["optimizer_groups"],
            "optimizer_state_dtypes": init[0]["optimizer_state_dtypes"],
        }
    log_root = Path(cfg.runner.logger.log_path) / cfg.runner.logger.experiment_name
    actor_root = log_root / f"checkpoints/global_step_{steps}/actor"
    full = actor_root / "model_state_dict/full_weights.pt"
    if not full.is_file() or not (actor_root / "dcp_checkpoint/.metadata").is_file():
        raise RuntimeError(
            "The final full-model or DCP training-state checkpoint is missing"
        )
    from torch.distributed.checkpoint import FileSystemReader

    metadata = FileSystemReader(actor_root / "dcp_checkpoint").read_metadata()
    keys = list(metadata.state_dict_metadata)
    if not any("optimizers" in x for x in keys) or not any(
        "lr_schedulers" in x for x in keys
    ):
        raise RuntimeError("DCP metadata omits optimizer or scheduler state")
    model = get_model(cfg.actor.model)
    weights = torch.load(full, map_location="cpu", weights_only=True, mmap=True)
    checkpoint = compare_weights(model, weights)
    checkpoint.update(
        path=str(full),
        sha256=sha256(full),
        bytes=full.stat().st_size,
        dcp_metadata_entries=len(keys),
        optimizer_resume_tested=False,
    )
    write_json(reports / "checkpoint.json", checkpoint)
    del model, weights
    validation = {}
    if phase != "formal":
        validation = read_validation_metrics(Path(cfg.runner.logger.log_path))
        if not validation or not all(
            math.isfinite(x["value"]) for row in validation.values() for x in row
        ):
            raise RuntimeError(
                "Real official validation metrics are missing or nonfinite"
            )
    state = json.loads((run / "manifests/run.json").read_text())
    view = reports / "evaluation_view"
    (view / "manifests").mkdir(parents=True, exist_ok=False)
    (view / "runtime").mkdir()
    state["rlinf_repo"] = str(snapshot / "official")
    write_json(view / "manifests/run.json", state)
    (view / "manifests/sft_files.json").symlink_to(run / "manifests/sft_files.json")
    (view / "runtime/libero_config").symlink_to(
        run / "runtime/libero_config", target_is_directory=True
    )
    inference = []
    for protocol in ["recipe-240", "standalone-220"]:
        workers = []
        for gpu, task in [(0, 0), (1, 5)]:
            output = reports / f"inference-{protocol}-gpu{gpu}.jsonl"
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu))
            command = [
                sys.executable,
                str(snapshot / "evaluation/evaluate_spatial.py"),
                "--run",
                str(view),
                "--protocol",
                protocol,
                "--task-ids",
                str(task),
                "--trials",
                "1",
                "--checkpoint",
                str(full),
                "--output",
                str(output),
            ]
            stream = output.with_suffix(".log").open("x")
            workers.append(
                (
                    subprocess.Popen(
                        command, env=env, stdout=stream, stderr=subprocess.STDOUT
                    ),
                    stream,
                    output,
                )
            )
        codes = []
        for worker, stream, output in workers:
            codes.append(worker.wait())
            stream.close()
        if any(codes):
            raise RuntimeError(f"Reloaded GPU inference failed: {protocol}: {codes}")
        for _, _, output in workers:
            rows = [
                json.loads(line) for line in output.read_text().splitlines() if line
            ]
            if (
                len(rows) != 1
                or rows[0]["status"] != "completed"
                or rows[0]["checkpoint_sha256"] != checkpoint["sha256"]
            ):
                raise RuntimeError(f"Reloaded inference provenance failed: {output}")
            inference.extend(rows)
    report = {
        "passed": True,
        "phase": phase,
        "source_tree_sha256": manifest["tree_sha256"],
        "verification_tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__)),
        },
        "config_sha256": sha256(run / f"configs/{phase}.yaml"),
        "actors": actors,
        "checkpoint": checkpoint,
        "validation": validation,
        "inference": inference,
        "scope": "Actual official runner training and save; strict CPU full-weight reload and frozen-VLM equality; GPU inference with reloaded weights. Finite phases include official runner validation. DCP optimizer/scheduler metadata checked; optimizer resume is not tested. Formal performance still requires both full 500-episode evaluations.",
    }
    write_json(reports / "lifecycle.json", report)
    print(
        json.dumps(
            {
                "passed": True,
                "phase": phase,
                "checkpoint": str(full),
                "source_tree_sha256": manifest["tree_sha256"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
