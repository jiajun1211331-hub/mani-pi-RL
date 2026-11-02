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

"""Finite scalar diagnostics with weighted aggregation across scoring shards."""

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist

from .advantages import ManifoldAdvantages
from .groups import GroupBatch, MemberScore


@dataclass
class DiagnosticTotals:
    """Local sufficient statistics; values contain only finite scalar samples."""

    counts: Counter = field(default_factory=Counter)
    values: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))


def metric_records(member: MemberScore, group_reason: str | None) -> dict[str, dict]:
    """Describe D/V/G/D:V independently, including unrequested diagnostics."""
    result = {}
    for label, field_name in (
        ("D", "dimension"),
        ("V", "volume"),
        ("G", "score"),
        ("R", "ratio"),
    ):
        if member.metrics is None:
            result[label] = {
                "value": None,
                "valid": False,
                "state": "not_computed",
                "reason": group_reason,
            }
            continue
        metric = getattr(member.metrics, field_name)
        if metric.valid:
            if not math.isfinite(metric.value):
                raise ValueError(
                    f"Expected finite valid manifold {label}, found {metric.value}."
                )
            state = "valid"
        elif metric.reason == "dimension_disabled":
            state = "disabled"
        elif metric.reason == "tle_dependency_unavailable":
            state = "unavailable"
        else:
            state = "invalid"
        result[label] = {
            "value": metric.value,
            "valid": metric.valid,
            "state": state,
            "reason": metric.reason,
        }
    return result


def mode_diagnostics(options: Mapping[str, Any]) -> dict[str, float]:
    """Expose the declared mode, score, and group selector as numeric log tags."""
    mode = options.get("mode", "disabled")
    score = options.get("geometry", {}).get("score", "volume_exp_neg_dim")
    eligible = options.get("eligible_groups", "any_success")
    return {
        "manifold/enabled": float(mode != "disabled"),
        f"manifold/mode/{mode}": 1.0,
        f"manifold/score/{score}": 1.0,
        f"manifold/eligible_groups/{eligible}": 1.0,
    }


def local_diagnostics(
    groups: GroupBatch,
    advantages: ManifoldAdvantages,
    *,
    geometry_seconds: float,
    hidden_bytes: int,
    audit_seconds: float = 0.0,
) -> DiagnosticTotals:
    """Measure this shard's groups and masked signals without retaining hidden."""
    totals = DiagnosticTotals()
    counts = totals.counts
    for name in (
        "K",
        "D",
        "V",
        "G",
        "R",
        "selected_score",
        "score_group_mean",
        "score_group_std",
        "episode_actions",
    ):
        totals.values[name] = []
    for name in (
        "raw",
        "eligible",
        "mixed",
        "all_success",
        "all_failed",
        "valid",
        "invalid",
        "constant",
        "geometry",
        "signal",
    ):
        counts[f"groups/{name}"] = 0
    for label in ("D", "V", "G", "R"):
        for state in ("valid", "invalid", "disabled", "unavailable", "not_computed"):
            counts[f"{label}/{state}"] = 0
    counts["hidden_bytes"] = hidden_bytes
    totals.values["geometry_seconds"].append(geometry_seconds)
    totals.values["audit_seconds"].append(audit_seconds)
    counts["chunks/native"] = int(advantages.action_mask.sum())
    counts["chunks/selected"] = int(advantages.loss_mask.sum())
    counts["chunks/prefix"] = int(
        (advantages.prefix_mask & advantages.action_mask).sum()
    )
    counts["chunks/geometry"] = 0
    for group in groups.groups:
        columns = [member.column for member in group.members]
        counts["groups/raw"] += 1
        counts[f"groups/{group.category}"] += 1
        successes = sum(member.success for member in group.members)
        counts[f"groups/success_{successes}_of_{len(group.members)}"] += 1
        accepted = bool(advantages.group_mask[:, columns].any())
        counts["groups/eligible"] += int(accepted)
        counts["groups/valid"] += int(group.valid)
        counts["groups/constant"] += int(group.reason == "constant_score")
        counts["groups/invalid"] += int(
            group.reason not in {None, "all_failed", "constant_score"}
        )
        counts["groups/geometry"] += int(group.valid and accepted)
        counts["groups/signal"] += int(
            (
                (advantages.final[:, columns] != 0) & advantages.loss_mask[:, columns]
            ).any()
        )
        if group.reason is not None:
            counts[f"groups/reason/{re.sub(r'[^A-Za-z0-9_]+', '_', group.reason)}"] += 1
        if group.prefix_length is not None:
            totals.values["K"].append(float(group.prefix_length))
            bucket = next(
                (
                    label
                    for bound, label in (
                        (10, "0_10"),
                        (50, "11_50"),
                        (100, "51_100"),
                        (200, "101_200"),
                        (480, "201_480"),
                    )
                    if group.prefix_length <= bound
                ),
                "gt_480",
            )
            counts[f"K/bin_{bucket}"] += 1
        if group.mean is not None:
            totals.values["score_group_mean"].append(group.mean)
        if group.std is not None:
            totals.values["score_group_std"].append(group.std)
        if group.valid and accepted:
            counts["chunks/geometry"] += int(
                (
                    advantages.prefix_mask[:, columns]
                    & advantages.action_mask[:, columns]
                ).sum()
            )
        for member in group.members:
            totals.values["episode_actions"].append(float(member.length))
            records = metric_records(member, group.reason)
            for label, record in records.items():
                counts[f"{label}/{record['state']}"] += 1
                if record["reason"] is not None:
                    reason = re.sub(r"[^A-Za-z0-9_]+", "_", record["reason"])
                    counts[f"{label}/reason/{reason}"] += 1
                if record["valid"]:
                    totals.values[label].append(record["value"])
            selected = records["V" if group.score_name == "volume" else "G"]
            if selected["valid"]:
                totals.values["selected_score"].append(selected["value"])
    mask = advantages.loss_mask
    for name, values in (
        ("task", advantages.task),
        ("geometry", advantages.geometry),
        ("final", advantages.final),
    ):
        counts[f"advantage/{name}_squared"] = float(
            values[mask].double().square().sum()
        )
    paired = mask & (advantages.task != 0) & (advantages.geometry != 0)
    counts["advantage/paired"] = int(paired.sum())
    counts["advantage/same_sign"] = int(
        (paired & ((advantages.task > 0) == (advantages.geometry > 0))).sum()
    )
    return totals


def aggregate_diagnostics(
    local: DiagnosticTotals, device: torch.device | str | int
) -> dict[str, float]:
    """Sum sufficient statistics before computing global means and L2 norms."""
    count_keys, sample_keys = set(local.counts), set(local.values)
    if dist.is_initialized():
        keys = [None] * dist.get_world_size()
        dist.all_gather_object(keys, (count_keys, sample_keys))
        count_keys = set().union(*(entry[0] for entry in keys))
        sample_keys = set().union(*(entry[1] for entry in keys))
    count_keys, sample_keys = sorted(count_keys), sorted(sample_keys)
    counters = torch.tensor(
        [local.counts[key] for key in count_keys], dtype=torch.float64, device=device
    )
    moments = torch.zeros((len(sample_keys), 3), dtype=torch.float64, device=device)
    minima = torch.full(
        (len(sample_keys),), float("inf"), dtype=torch.float64, device=device
    )
    maxima = torch.full_like(minima, -float("inf"))
    for index, key in enumerate(sample_keys):
        values = np.asarray(local.values.get(key, []), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Expected finite manifold diagnostic samples for {key}.")
        if len(values):
            moments[index] = torch.tensor(
                [values.sum(), np.square(values).sum(), len(values)],
                dtype=torch.float64,
                device=device,
            )
            minima[index], maxima[index] = float(values.min()), float(values.max())
    if dist.is_initialized():
        if counters.numel():
            dist.all_reduce(counters, op=dist.ReduceOp.SUM)
        if moments.numel():
            dist.all_reduce(moments, op=dist.ReduceOp.SUM)
            dist.all_reduce(minima, op=dist.ReduceOp.MIN)
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
    result = dict(zip(count_keys, counters.cpu().tolist(), strict=True))
    for index, key in enumerate(sample_keys):
        total, squares, count = moments[index].cpu().tolist()
        result[f"{key}/count"] = count
        if count:
            mean = total / count
            result.update(
                {
                    f"{key}/sum": total,
                    f"{key}/mean": mean,
                    f"{key}/min": minima[index].item(),
                    f"{key}/max": maxima[index].item(),
                    f"{key}/std": math.sqrt(
                        max(0.0, (squares - total * mean) / (count - 1))
                    )
                    if count > 1
                    else 0.0,
                }
            )
    for name in ("task", "geometry", "final"):
        key = f"advantage/{name}_squared"
        if key in result:
            result[f"advantage/{name}_l2"] = math.sqrt(result.pop(key))
    if "chunks/native" in result:
        result["geometry_chunk_fraction"] = result.get("chunks/geometry", 0.0) / max(
            1.0, result["chunks/native"]
        )
    if "advantage/paired" in result:
        result["advantage/same_sign_fraction"] = result["advantage/same_sign"] / max(
            1.0, result["advantage/paired"]
        )
    return {f"manifold/{key}": value for key, value in result.items()}
