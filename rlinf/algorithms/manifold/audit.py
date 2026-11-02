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

"""Online scalar evidence, bounded feature dumps, and scalar-only replay."""

import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .advantages import ManifoldAdvantages, combine_advantages
from .diagnostics import metric_records
from .groups import GroupBatch, GroupKey, GroupScore, MemberScore
from .metrics import ManifoldMetrics, Metric


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


def _source_provenance() -> dict:
    root = Path(__file__).resolve().parents[3]
    files = [
        *root.joinpath("rlinf/algorithms/manifold").glob("*.py"),
        *(
            root / name
            for name in (
                "rlinf/config.py",
                "rlinf/algorithms/advantages.py",
                "rlinf/algorithms/utils.py",
                "rlinf/algorithms/losses.py",
                "rlinf/algorithms/registry.py",
                "rlinf/hybrid_engines/fsdp/fsdp_model_manager.py",
                "rlinf/data/schema/embodied_types.py",
                "rlinf/models/embodiment/openpi/pi0.py",
                "rlinf/models/embodiment/openpi/tasks/rl.py",
                "rlinf/workers/actor/embodied_fsdp_actor_worker.py",
                "rlinf/workers/rollout/hf/huggingface_worker.py",
                "rlinf/workers/env/env_worker.py",
                "rlinf/envs/sim/libero/libero_env.py",
            )
        ),
    ]
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }
    head = None
    if (root / ".git").exists():
        try:
            head = (
                subprocess.check_output(
                    ["git", "-C", str(root), "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                .decode()
                .strip()
            )
        except (subprocess.SubprocessError, OSError):
            head = None
    versions = {}
    for name in ("numpy", "torch", "scikit-dimension", "scikit-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    snapshot = None
    selected = os.environ.get("RLINF_MANIFOLD_RUN")
    if selected is not None:
        from .snapshot import verify_snapshot

        run = Path(selected).resolve()
        manifest = verify_snapshot(run)
        if root != run / "source":
            raise RuntimeError(
                f"Expected manifold audit source in {run / 'source'}, found {root}."
            )
        snapshot = {
            "run": str(run),
            "head": manifest["head"],
            "tree_sha256": manifest["tree_sha256"],
            "development_snapshot": manifest["development_snapshot"],
        }
    return {
        "source_root": str(root),
        "git_head": head,
        "code_sha256": hashlib.sha256(_json_bytes(hashes)).hexdigest(),
        "source_files": hashes,
        "packages": versions,
        "snapshot": snapshot,
    }


class ManifoldAudit:
    """Own one rank's immutable batches and its run-wide bounded hidden dumps."""

    def __init__(self, cfg: DictConfig, rank: int):
        options = cfg.algorithm.manifold
        diagnostics = options.get("diagnostics", {})
        base = diagnostics.get("audit_dir")
        if base is None:
            base = (
                Path(cfg.runner.logger.log_path)
                / cfg.runner.logger.experiment_name
                / "manifold"
            )
        self.directory = Path(base).expanduser().resolve() / f"rank_{rank:05d}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.max_hidden = diagnostics.get("debug_hidden_max_trajectories", 0)
        existing = list(self.directory.glob("debug_hidden_*.npz"))
        self.saved_hidden = len(existing)
        self.next_hidden = (
            max((int(path.stem.rsplit("_", 1)[1]) for path in existing), default=-1) + 1
        )
        geometry = options.get("geometry", {})
        normalization = options.get("normalization", {})
        self.settings = {
            "group_size": cfg.algorithm.group_size,
            "mode": options.mode,
            "score": geometry.get("score", "volume_exp_neg_dim"),
            "eligible_groups": options.get("eligible_groups", "any_success"),
            "weight": options.get("weight", 0.2),
            "epsilon": normalization.get("epsilon", 1e-6),
            "min_group_std": normalization.get("min_group_std", 1e-6),
            "tle_k": geometry.get("tle_k", 10),
            "chunk_size": cfg.actor.model.num_action_chunks,
        }
        provenance = {
            "config": OmegaConf.to_container(cfg, resolve=True),
            **_source_provenance(),
        }
        encoded = _json_bytes(provenance)
        self.provenance_sha256 = hashlib.sha256(encoded).hexdigest()
        self.provenance_path = (
            self.directory / f"provenance_{self.provenance_sha256}.json"
        )
        if self.provenance_path.exists():
            if self.provenance_path.read_bytes() != encoded:
                raise ValueError(
                    "Expected matching immutable manifold provenance content."
                )
        else:
            with self.provenance_path.open("xb") as stream:
                stream.write(encoded)
        self.last_batch: Path | None = None

    def write_batch(
        self,
        groups: GroupBatch,
        advantages: ManifoldAdvantages,
        hidden: np.ndarray,
        *,
        iteration: int,
        geometry_seconds: float,
        hidden_bytes: int,
    ) -> Path:
        """Write contemporaneous scalar evidence and optional exact-prefix features."""
        rows = []
        dumps = []
        chunks, batch, _ = advantages.final.shape
        chunk_size = self.settings["chunk_size"]
        for group in groups.groups:
            members = []
            for member in group.members:
                column = member.column
                native = advantages.action_mask[:, column, 0]
                valid_chunks = int(native.sum())
                expected = torch.arange(chunks, device=native.device) < valid_chunks
                if not torch.equal(native, expected):
                    raise ValueError(
                        "Expected a contiguous native GRPO chunk mask for scalar audit."
                    )
                values = advantages.task[:, column, 0][native]
                task = float(values[0]) if valid_chunks else 0.0
                if valid_chunks and not (values == task).all():
                    raise ValueError(
                        "Expected a constant native GRPO advantage within each episode."
                    )
                members.append(
                    {
                        "column": column,
                        "member_id": member.member_id,
                        "task_id": member.task_id,
                        "trial_id": member.trial_id,
                        "reset_state_id": member.reset_state_id,
                        "length": member.length,
                        "success": member.success,
                        "N": member.metrics.n if member.metrics else None,
                        "metrics": metric_records(member, group.reason),
                        "metric_diagnostics": member.metrics.diagnostics
                        if member.metrics
                        else {},
                        "z_score": float(groups.standardized_scores[column]),
                        "task_advantage": task,
                        "native_valid_chunks": valid_chunks,
                    }
                )
                if self.saved_hidden < self.max_hidden:
                    points = min(member.length, group.prefix_length or member.length)
                    path = self.directory / f"debug_hidden_{self.next_hidden:06d}.npz"
                    prefix = (
                        hidden[: math.ceil(points / chunk_size), column]
                        .reshape(-1, 1024)[:points]
                        .copy()
                    )
                    with path.open("xb") as stream:
                        np.savez_compressed(stream, hidden=prefix)
                    dumps.append(
                        {
                            "path": path.name,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "group_key": asdict(group.key),
                            "member_id": member.member_id,
                            "column": column,
                            "actions": points,
                            "complete_metric_prefix": group.prefix_length is not None
                            and points == group.prefix_length,
                        }
                    )
                    self.saved_hidden += 1
                    self.next_hidden += 1
            columns = [member.column for member in group.members]
            rows.append(
                {
                    "key": asdict(group.key),
                    "category": group.category,
                    "K": group.prefix_length,
                    "score": group.score_name,
                    "mean": group.mean,
                    "std": group.std,
                    "valid": group.valid,
                    "reason": group.reason,
                    "eligible": bool(advantages.group_mask[:, columns].any()),
                    "members": members,
                }
            )
        record = {
            "event": "batch",
            "schema_version": 1,
            "rank": self.rank,
            "iteration": iteration,
            "created_ns": time.time_ns(),
            "settings": self.settings,
            "layout": [chunks, batch, 1],
            "provenance_file": self.provenance_path.name,
            "provenance_sha256": self.provenance_sha256,
            "geometry_seconds": geometry_seconds,
            "hidden_bytes": hidden_bytes,
            "groups": rows,
            "debug_hidden": dumps,
        }
        path = self.directory / f"step_{iteration:08d}_{uuid.uuid4().hex}.jsonl"
        with path.open("xb") as stream:
            stream.write(_json_bytes(record))
        self.last_batch = path
        return path

    def write_updates(
        self,
        *,
        updates: int,
        skips: int,
        optimizer_steps: int,
        scheduler_advanced: bool,
    ) -> None:
        """Append training results to the batch that supplied its advantages."""
        if self.last_batch is None:
            raise ValueError(
                "Expected a manifold batch audit before recording optimizer updates."
            )
        record = {
            "event": "updates",
            "created_ns": time.time_ns(),
            "updates": updates,
            "skips": skips,
            "optimizer_steps": optimizer_steps,
            "scheduler_advanced": scheduler_advanced,
        }
        with self.last_batch.open("ab") as stream:
            stream.write(_json_bytes(record))


def replay_advantages(record: dict) -> ManifoldAdvantages:
    """Reconstruct group selection and both advantages from batch scalars alone."""
    if record.get("event") != "batch" or record.get("schema_version") != 1:
        raise ValueError(
            "Expected a manifold scalar batch audit with schema_version=1."
        )
    chunks, batch, width = record["layout"]
    if min(chunks, batch) < 1 or width != 1:
        raise ValueError("Expected a nonempty [T,B,1] scalar audit layout.")
    settings = record["settings"]
    chunk_size = settings["chunk_size"]
    prefix_mask = np.zeros((chunks, batch, 1), dtype=bool)
    standardized = np.zeros(batch, dtype=np.float64)
    task = torch.zeros(chunks, batch, 1, dtype=torch.float32)
    native = torch.zeros_like(task, dtype=torch.bool)
    seen = []
    groups = []
    for row in record["groups"]:
        members = row["members"]
        columns = [member["column"] for member in members]
        if any(not 0 <= column < batch for column in columns):
            raise ValueError("Expected in-range scalar audit trajectory columns.")
        seen.extend(columns)
        if {member["member_id"] for member in members} != set(
            range(settings["group_size"])
        ) or len(members) != settings["group_size"]:
            raise ValueError("Expected unique members in every scalar audit group.")
        successful = [member["length"] for member in members if member["success"]]
        prefix = min(successful, default=None)
        category = (
            "all_failed"
            if not successful
            else "all_success"
            if len(successful) == len(members)
            else "mixed"
        )
        if (
            prefix != row["K"]
            or category != row["category"]
            or row["score"] != settings["score"]
        ):
            raise ValueError(
                "Scalar audit prefix, category or score disagrees with its member evidence."
            )
        reason = "all_failed" if prefix is None else None
        mean = std = None
        if prefix is not None:
            if min(member["length"] for member in members) < prefix:
                reason = "member_shorter_than_prefix"
            else:
                label = "V" if settings["score"] == "volume" else "G"
                selected = [member["metrics"][label] for member in members]
                if not all(metric["valid"] for metric in selected):
                    reason = "invalid_member_score"
                else:
                    scores = np.asarray(
                        [metric["value"] for metric in selected], dtype=np.float64
                    )
                    if not np.isfinite(scores).all():
                        raise ValueError("Expected finite valid selected audit scores.")
                    mean, std = float(scores.mean()), float(scores.std(ddof=1))
                    if not math.isfinite(mean) or not math.isfinite(std):
                        reason = "nonfinite_group_statistics"
                    elif std <= settings["min_group_std"]:
                        reason = "constant_score"
                    else:
                        standardized[columns] = (scores - mean) / (
                            std + settings["epsilon"]
                        )
        for name, expected in (("mean", mean), ("std", std)):
            actual = row[name]
            if (actual is None) != (expected is None) or (
                expected is not None
                and not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise ValueError(
                    f"Scalar audit {name} disagrees with selected metric values."
                )
        accepted = category == "mixed" or (
            settings["eligible_groups"] == "any_success"
            and category == "all_success"
            and reason is None
        )
        if (
            row["valid"] != (reason is None)
            or row["reason"] != reason
            or row["eligible"] != accepted
        ):
            raise ValueError(
                "Scalar audit validity or eligibility disagrees with its evidence."
            )
        decoded = []
        for member in members:
            column = member["column"]
            if (
                column < 0
                or column >= batch
                or not 0 <= member["native_valid_chunks"] <= chunks
            ):
                raise ValueError(
                    "Expected in-range scalar audit columns and native chunk counts."
                )
            if not math.isclose(
                member["z_score"], standardized[column], rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    "Scalar audit z score disagrees with selected group normalization."
                )
            native[:, column, 0] = torch.arange(chunks) < member["native_valid_chunks"]
            task[:, column, 0] = member["task_advantage"] * native[:, column, 0]
            if prefix is not None:
                prefix_mask[:, column, 0] = (np.arange(chunks) + 1) * chunk_size <= min(
                    prefix, member["length"]
                )
            metrics = None
            if member["N"] is not None:
                decoded_metrics = {
                    field: Metric(
                        member["metrics"][label]["value"],
                        member["metrics"][label]["reason"],
                    )
                    for label, field in (
                        ("D", "dimension"),
                        ("V", "volume"),
                        ("G", "score"),
                        ("R", "ratio"),
                    )
                }
                metrics = ManifoldMetrics(
                    member["N"], settings["tle_k"], **decoded_metrics
                )
            decoded.append(
                MemberScore(
                    column,
                    member["member_id"],
                    member["task_id"],
                    member["trial_id"],
                    member["reset_state_id"],
                    member["length"],
                    member["success"],
                    metrics,
                )
            )
        groups.append(
            GroupScore(
                GroupKey(**row["key"]),
                tuple(decoded),
                category,
                prefix,
                settings["score"],
                mean,
                std,
                reason is None,
                reason,
            )
        )
    if sorted(seen) != list(range(batch)):
        raise ValueError("Expected every scalar audit trajectory column exactly once.")
    return combine_advantages(
        task,
        native,
        GroupBatch(tuple(groups), prefix_mask, standardized),
        mode=settings["mode"],
        eligible_groups=settings["eligible_groups"],
        weight=settings["weight"],
    )
