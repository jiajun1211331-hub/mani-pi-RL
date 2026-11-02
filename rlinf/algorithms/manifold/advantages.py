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

"""Detached task/geometry advantages and synchronized empty-update detection."""

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist

from rlinf.algorithms.registry import calculate_adv_and_returns

from .groups import GroupBatch


@dataclass(frozen=True)
class ManifoldAdvantages:
    """Independent masks and signals in the original [T,B,1] chunk layout."""

    task: torch.Tensor
    geometry: torch.Tensor
    final: torch.Tensor
    action_mask: torch.Tensor
    group_mask: torch.Tensor
    prefix_mask: torch.Tensor
    loss_mask: torch.Tensor


def native_group_advantages(groups: GroupBatch, **kwargs) -> dict[str, torch.Tensor]:
    """Run native GRPO on explicit groups, then restore the original columns."""
    if kwargs.get("adv_type") != "grpo" or kwargs.get("reward_type") != "chunk_level":
        raise ValueError(
            "Expected native chunk-level GRPO for manifold task advantages."
        )
    columns = [member.column for group in groups.groups for member in group.members]
    batch_size = kwargs["rewards"].shape[1]
    if sorted(columns) != list(range(batch_size)):
        raise ValueError(
            "Expected manifold groups to cover every original trajectory exactly once."
        )
    permutation = torch.tensor(columns, dtype=torch.long)
    ordered = dict(kwargs)
    for name in (
        "rewards",
        "dones",
        "values",
        "prev_logprobs",
        "teacher_logprobs",
        "loss_mask",
        "loss_mask_sum",
    ):
        value = ordered.get(name)
        if isinstance(value, torch.Tensor):
            ordered[name] = value.index_select(1, permutation.to(value.device))
    result = calculate_adv_and_returns(**ordered)
    inverse = permutation.argsort()
    return {
        name: value.index_select(1, inverse.to(value.device))
        for name, value in result.items()
    }


def combine_advantages(
    task_advantages: torch.Tensor,
    action_mask: torch.Tensor,
    groups: GroupBatch,
    *,
    mode: str,
    eligible_groups: str = "any_success",
    weight: float = 0.2,
) -> ManifoldAdvantages:
    """Add the selected geometry signal without re-normalizing final advantages.

    Group eligibility changes the native loss mask. Geometry validity, nonzero
    advantages, and the full-chunk prefix never shrink its denominator.
    """
    if mode not in {"task_plus_geometry", "geometry_only"}:
        raise ValueError(f"Expected a manifold advantage mode, found {mode!r}.")
    if eligible_groups not in {"mixed_only", "any_success"}:
        raise ValueError(
            f"Expected mixed_only or any_success, found {eligible_groups!r}."
        )
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(
            f"Expected a finite nonnegative manifold weight, found {weight}."
        )
    if (
        task_advantages.ndim != 3
        or task_advantages.shape[-1] != 1
        or task_advantages.shape != action_mask.shape
        or action_mask.dtype != torch.bool
        or tuple(groups.prefix_mask.shape) != tuple(task_advantages.shape)
        or groups.standardized_scores.shape != (task_advantages.shape[1],)
    ):
        raise ValueError(
            "Expected matching [T,B,1] task advantages, action mask and manifold prefix."
        )
    task = task_advantages.detach()
    action_mask = action_mask.detach()
    if not torch.isfinite(task).all():
        raise ValueError(
            "Expected finite native task advantages for manifold training."
        )
    group_mask = torch.zeros_like(action_mask)
    for group in groups.groups:
        accepted = group.category == "mixed" or (
            eligible_groups == "any_success"
            and group.category == "all_success"
            and group.valid
        )
        if accepted:
            group_mask[:, [member.column for member in group.members]] = True
    prefix = torch.as_tensor(groups.prefix_mask, device=task.device, dtype=torch.bool)
    z = torch.as_tensor(
        groups.standardized_scores, device=task.device, dtype=task.dtype
    )
    if not torch.isfinite(z).all():
        raise ValueError("Expected finite standardized manifold scores.")
    geometry = weight * z[None, :, None] * prefix * group_mask
    final = geometry if mode == "geometry_only" else task * group_mask + geometry
    loss_mask = action_mask & group_mask
    return ManifoldAdvantages(
        task,
        geometry.detach(),
        final.detach(),
        action_mask,
        group_mask,
        prefix,
        loss_mask,
    )


def has_global_signal(
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    device: torch.device | str | int,
) -> bool:
    """Agree on whether an optimizer boundary has any nonzero masked advantage."""
    local = bool(((advantages.detach() != 0) & loss_mask.to(torch.bool)).any())
    signal = torch.tensor(int(local), dtype=torch.int32, device=device)
    if dist.is_initialized():
        dist.all_reduce(signal, op=dist.ReduceOp.MAX)
    return bool(signal.item())
