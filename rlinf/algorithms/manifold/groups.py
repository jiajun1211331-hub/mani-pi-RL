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

"""Explicit episode grouping and matched action-prefix manifold scoring."""

import math
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .metrics import ManifoldMetrics, compute_metrics, require_tle


@dataclass(frozen=True)
class GroupKey:
    """Identity of one group under a single behavior policy and collector."""

    policy_version: int
    collector_step: int
    rollout_epoch: int
    env_rank: int
    stage_id: int
    group_slot: int


@dataclass(frozen=True)
class MemberScore:
    """Scalar episode evidence for one original batch column."""

    column: int
    member_id: int
    task_id: int
    trial_id: int
    reset_state_id: int
    length: int
    success: bool
    metrics: ManifoldMetrics | None = None


@dataclass(frozen=True)
class GroupScore:
    """Matched-prefix scores and independent group numerical validity."""

    key: GroupKey
    members: tuple[MemberScore, ...]
    category: str
    prefix_length: int | None
    score_name: str
    mean: float | None
    std: float | None
    valid: bool
    reason: str | None


@dataclass(frozen=True)
class GroupBatch:
    """Scores in original column order, without retaining high-dimensional data."""

    groups: tuple[GroupScore, ...]
    # Complete valid chunks within K, independent of selected-score validity.
    prefix_mask: np.ndarray
    # Per-column z scores; invalid and constant groups have zero signal.
    standardized_scores: np.ndarray


def score_groups(
    hidden: np.ndarray,
    context: Mapping[str, np.ndarray],
    *,
    group_size: int = 8,
    score: str = "volume_exp_neg_dim",
    k: int = 10,
    tle_epsilon: float = 1e-4,
    alpha: float = 0.1,
    normalization_epsilon: float = 1e-6,
    min_group_std: float = 1e-6,
    dimension_diagnostics: bool = False,
) -> GroupBatch:
    """Score complete groups using their minimum successful action prefix.

    Args:
        hidden: Detached FP32 [chunks,batch,actions,1024] action features.
        context: Integer identity arrays [chunks,batch] and boolean per-action
            arrays [chunks,batch,actions] carried by the trajectory schema.
        group_size: Required unique members per source group, at least two.
        score: Either volume_exp_neg_dim (G) or volume (V).
        k: Shared minimum-prefix gate and TLE neighbor count.
        tle_epsilon: TLE tolerance.
        alpha: G dimension penalty; ignored by V scoring.
        normalization_epsilon: Additive denominator after sample standard deviation.
        min_group_std: Groups at or below this sample std carry no geometry signal.
        dimension_diagnostics: Also attempt D/G diagnostics for V scoring.

    Returns:
        Per-group scalar evidence, full-chunk masks, and per-member normalized
        scores. Identity/shape errors raise ValueError; numerical failures retain
        the entire original group and report an invalid reason.
    """
    x = np.asarray(hidden)
    if x.ndim != 4 or x.shape[-1] != 1024 or x.dtype != np.float32:
        raise ValueError(
            f"Expected FP32 [T,B,C,1024] manifold features, found {x.shape} {x.dtype}."
        )
    chunks, batch_size, chunk_size, _ = x.shape
    if min(chunks, batch_size, chunk_size) < 1:
        raise ValueError(
            f"Expected nonempty manifold rollout dimensions, found {x.shape}."
        )
    if score not in {"volume", "volume_exp_neg_dim"}:
        raise ValueError(
            f"Expected volume or volume_exp_neg_dim scoring, found {score!r}."
        )
    if not isinstance(group_size, int) or group_size < 2:
        raise ValueError(f"Expected group_size>=2, found {group_size}.")
    if (
        not isinstance(k, int)
        or k < 1
        or not math.isfinite(tle_epsilon)
        or tle_epsilon <= 0
        or not math.isfinite(alpha)
        or alpha < 0
        or not math.isfinite(normalization_epsilon)
        or normalization_epsilon <= 0
        or not math.isfinite(min_group_std)
        or min_group_std < 0
    ):
        raise ValueError(
            "Expected finite positive TLE/normalization parameters "
            "and nonnegative alpha/min_group_std."
        )
    identity_fields = (
        "policy_version",
        "collector_step",
        "rollout_epoch",
        "env_rank",
        "stage_id",
        "group_slot",
        "member_id",
        "task_id",
        "trial_id",
        "reset_state_id",
    )
    identities = {}
    for name in (*identity_fields, "chunk_id"):
        if name not in context:
            raise ValueError(f"Expected manifold identity field {name!r}.")
        values = np.asarray(context[name])
        if (
            values.shape != (chunks, batch_size)
            or not np.issubdtype(values.dtype, np.integer)
            or (values < 0).any()
        ):
            raise ValueError(
                f"Expected nonnegative integer {name} [T,B], found {values.shape} {values.dtype}."
            )
        if name == "chunk_id":
            if not (values == np.arange(chunks)[:, None]).all():
                raise ValueError(
                    "Expected ordered contiguous chunk IDs starting at zero for every episode."
                )
        else:
            if not (values == values[:1]).all():
                raise ValueError(
                    f"Expected stable manifold {name} across all episode chunks."
                )
            identities[name] = values[0]
    outcomes = {}
    for name in ("action_dones", "action_successes", "action_valid"):
        if name not in context:
            raise ValueError(f"Expected exact per-action manifold field {name!r}.")
        values = np.asarray(context[name])
        if values.shape != (chunks, batch_size, chunk_size) or values.dtype != np.bool_:
            raise ValueError(
                f"Expected boolean {name} [T,B,C], found {values.shape} {values.dtype}."
            )
        outcomes[name] = values.transpose(1, 0, 2).reshape(batch_size, -1)
    dones = outcomes["action_dones"]
    successes = outcomes["action_successes"]
    valid = outcomes["action_valid"]
    if (successes & ~dones).any():
        raise ValueError(
            "Expected every successful action to be an explicit episode termination."
        )
    total_actions = chunks * chunk_size
    first_done = np.where(dones.any(axis=1), dones.argmax(axis=1), total_actions)
    has_done = first_done < total_actions
    if not valid[np.arange(batch_size)[has_done], first_done[has_done]].all():
        raise ValueError(
            "Expected the first terminal action to be valid and included in its episode."
        )
    effective = valid & (np.arange(total_actions)[None] <= first_done[:, None])
    if (effective[:, 1:] & ~np.logical_and.accumulate(effective, axis=1)[:, :-1]).any():
        raise ValueError(
            "Expected valid manifold actions to form one contiguous episode prefix."
        )
    lengths = effective.sum(axis=1)
    successful = np.zeros(batch_size, dtype=bool)
    successful[has_done] = successes[
        np.arange(batch_size)[has_done], first_done[has_done]
    ]
    valid_chunks = effective.reshape(batch_size, chunks, chunk_size).all(axis=-1).T
    if score == "volume_exp_neg_dim":
        require_tle()
    grouped: dict[GroupKey, list[int]] = {}
    for column in range(batch_size):
        key = GroupKey(
            *(int(identities[name][column]) for name in GroupKey.__dataclass_fields__)
        )
        grouped.setdefault(key, []).append(column)

    prefix_mask = np.zeros((chunks, batch_size, 1), dtype=bool)
    standardized = np.zeros(batch_size, dtype=np.float64)
    groups = []
    for key, columns in grouped.items():
        member_ids = identities["member_id"][columns]
        if len(columns) != group_size or set(member_ids.tolist()) != set(
            range(group_size)
        ):
            raise ValueError(
                f"Expected a complete local manifold group with {group_size} unique members "
                f"0..{group_size - 1}, found {member_ids.tolist()} for {key}."
            )
        for name in ("task_id", "trial_id", "reset_state_id"):
            if len(set(identities[name][columns].tolist())) != 1:
                raise ValueError(
                    f"Expected identical {name} across manifold group {key}."
                )
        members = tuple(
            MemberScore(
                column=column,
                member_id=int(identities["member_id"][column]),
                task_id=int(identities["task_id"][column]),
                trial_id=int(identities["trial_id"][column]),
                reset_state_id=int(identities["reset_state_id"][column]),
                length=int(lengths[column]),
                success=bool(successful[column]),
            )
            for column in columns
        )
        success_count = int(successful[columns].sum())
        category = (
            "all_failed"
            if success_count == 0
            else "all_success"
            if success_count == group_size
            else "mixed"
        )
        prefix = min(
            (member.length for member in members if member.success), default=None
        )
        mean = std = None
        reason = None
        if prefix is None:
            reason = "all_failed"
        else:
            prefix_mask[:, columns, 0] = (
                (np.arange(chunks) + 1) * chunk_size <= prefix
            )[:, None] & valid_chunks[:, columns]
            if min(member.length for member in members) < prefix:
                reason = "member_shorter_than_prefix"
            else:
                members = tuple(
                    replace(
                        member,
                        metrics=compute_metrics(
                            x[: math.ceil(prefix / chunk_size), member.column].reshape(
                                -1, 1024
                            )[:prefix],
                            k=k,
                            epsilon=tle_epsilon,
                            alpha=alpha,
                            compute_dimension=score == "volume_exp_neg_dim"
                            or dimension_diagnostics,
                        ),
                    )
                    for member in members
                )
                metrics = [
                    member.metrics.volume if score == "volume" else member.metrics.score
                    for member in members
                ]
                if not all(metric.valid for metric in metrics):
                    reason = "invalid_member_score"
                else:
                    scores = np.asarray(
                        [metric.value for metric in metrics], dtype=np.float64
                    )
                    mean, std = float(scores.mean()), float(scores.std(ddof=1))
                    if not math.isfinite(mean) or not math.isfinite(std):
                        reason = "nonfinite_group_statistics"
                    elif std <= min_group_std:
                        reason = "constant_score"
                    else:
                        standardized[columns] = (scores - mean) / (
                            std + normalization_epsilon
                        )
        groups.append(
            GroupScore(
                key=key,
                members=members,
                category=category,
                prefix_length=prefix,
                score_name=score,
                mean=mean,
                std=std,
                valid=reason is None,
                reason=reason,
            )
        )
    return GroupBatch(tuple(groups), prefix_mask, standardized)
