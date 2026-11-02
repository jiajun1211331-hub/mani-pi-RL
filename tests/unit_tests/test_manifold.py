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

"""Shared manifold metrics and matched-prefix scoring contracts."""

import hashlib
import importlib.metadata
import json
import math
from unittest.mock import patch

import numpy as np
import pytest
import torch

from rlinf.algorithms.manifold.advantages import (
    combine_advantages,
    has_global_signal,
    native_group_advantages,
)
from rlinf.algorithms.manifold.audit import ManifoldAudit, replay_advantages
from rlinf.algorithms.manifold.diagnostics import (
    aggregate_diagnostics,
    local_diagnostics,
)
from rlinf.algorithms.manifold.groups import score_groups
from rlinf.algorithms.manifold.metrics import compute_metrics, require_tle
from rlinf.utils.metric_utils import compute_loss_mask
from scripts.pi_rl.manifold_geometry import metrics as offline_metrics


def test_centered_gram_known_spectrum():
    result = compute_metrics(np.eye(12), compute_dimension=False)
    assert result.volume.valid
    assert result.volume.value == pytest.approx(0.5 * 11 * np.log(2), abs=1e-12)
    assert result.dimension.reason == "dimension_disabled"
    assert result.score.reason == "dimension_disabled"


def test_volume_translation_rotation_and_fp64():
    rng = np.random.default_rng(71)
    x = rng.normal(size=(21, 16)).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    original = compute_metrics(x, compute_dimension=False)
    transformed = compute_metrics(
        x.astype(np.float64) @ q + 100, compute_dimension=False
    )
    assert transformed.volume.value == pytest.approx(original.volume.value, abs=1e-11)


@pytest.mark.parametrize("n", [0, 1, 10])
def test_prefix_requires_more_than_k_points(n):
    result = compute_metrics(np.ones((n, 12)))
    for metric in (result.dimension, result.volume, result.score, result.ratio):
        assert not metric.valid
        assert metric.reason == "insufficient_points"


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_features(bad):
    x = np.ones((12, 4))
    x[0, 0] = bad
    result = compute_metrics(x)
    assert result.volume.reason == result.score.reason == "nonfinite_input"


def test_constant_features_have_no_volume():
    result = compute_metrics(np.ones((12, 4)), compute_dimension=False)
    assert not result.volume.valid
    assert result.volume.reason == "invalid_volume"
    assert not result.score.valid


def test_volume_survives_missing_optional_tle():
    x = np.random.default_rng(31).normal(size=(15, 8))
    with patch.object(
        importlib.metadata,
        "version",
        side_effect=importlib.metadata.PackageNotFoundError("scikit-dimension"),
    ):
        result = compute_metrics(x)
        assert result.volume.valid
        assert (
            result.dimension.reason
            == result.score.reason
            == "tle_dependency_unavailable"
        )
        with pytest.raises(ImportError, match=r"scikit-dimension==0.3.7"):
            require_tle()


def test_volume_does_not_load_tle():
    with patch.object(
        importlib.metadata, "version", side_effect=AssertionError("TLE loaded")
    ):
        assert compute_metrics(np.eye(12), compute_dimension=False).volume.valid


def test_wrong_tle_version_is_rejected():
    with patch.object(importlib.metadata, "version", return_value="0.3.6"):
        with pytest.raises(ImportError, match="requirements/manifold.txt"):
            require_tle()


def test_volume_survives_invalid_tle_on_duplicate_points():
    pytest.importorskip("skdim")
    x = np.repeat(np.eye(2), 12, axis=0)
    result = compute_metrics(x)
    assert result.volume.valid
    assert not result.dimension.valid
    assert not result.score.valid
    assert not result.ratio.valid


def test_full_metrics_and_offline_pi05_adapter():
    pytest.importorskip("skdim")
    x = np.random.default_rng(17).normal(size=(21, 1024))
    result = compute_metrics(x)
    offline = offline_metrics(x)
    assert result.dimension.valid and result.volume.valid
    assert result.score.valid and result.ratio.valid
    assert offline["status"] == "valid"
    assert offline["D"] == result.dimension.value
    assert offline["V"] == result.volume.value
    assert offline["G"] == pytest.approx(
        result.volume.value * np.exp(-0.1 * result.dimension.value)
    )
    assert offline["R"] == pytest.approx(result.dimension.value / result.volume.value)
    other_alpha = compute_metrics(x, alpha=0.9)
    assert other_alpha.volume == result.volume
    assert other_alpha.dimension == result.dimension
    assert other_alpha.score.value < result.score.value
    with pytest.raises(ValueError, match="1024"):
        offline_metrics(x[:, :10])


@pytest.mark.parametrize("shape", [(12,), (12, 0), (12, 3, 2)])
def test_invalid_feature_shape_is_contract_error(shape):
    with pytest.raises(ValueError, match="Expected.*features|Expected.*dimension"):
        compute_metrics(np.empty(shape))


@pytest.mark.parametrize(
    "kwargs", [{"k": 0}, {"epsilon": 0}, {"alpha": -1}, {"alpha": np.nan}]
)
def test_invalid_metric_parameters(kwargs):
    with pytest.raises(ValueError, match="Expected k"):
        compute_metrics(np.eye(12), **kwargs)


def test_overflowed_gram_is_reported():
    x = np.eye(12) * 1e200
    result = compute_metrics(x, compute_dimension=False)
    assert result.volume.reason == "nonfinite_gram"
    assert not result.score.valid


@pytest.mark.parametrize("negative, valid", [(-1e-12, True), (-1e-5, False)])
def test_spectrum_tolerance(negative, valid):
    # Model roundoff at the external eigensolver boundary.
    spectrum = np.array([negative] + [1.0] * 11)
    with patch.object(np.linalg, "eigvalsh", return_value=spectrum):
        result = compute_metrics(np.eye(12), compute_dimension=False)
    assert result.volume.valid is valid
    if valid:
        assert result.volume.value == pytest.approx(0.5 * 11 * np.log(2))
    else:
        assert result.volume.reason == "negative_spectrum_outside_tolerance"


def _group_rollout(chunks=3, size=8, first_success=17, all_success=False):
    rng = np.random.default_rng(52)
    hidden = rng.normal(size=(chunks, size, 10, 1024)).astype(np.float32)
    hidden *= np.linspace(0.5, 2, size, dtype=np.float32)[None, :, None, None]
    ids = np.arange(size)
    context = {
        name: np.zeros((chunks, size), dtype=np.int64)
        for name in (
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
            "chunk_id",
        )
    }
    context["chunk_id"][:] = np.arange(chunks)[:, None]
    for name in ("group_slot", "task_id"):
        context[name][:] = ids // 8
    context["reset_state_id"][:] = ids // 8 * 50
    context["member_id"][:] = ids % 8
    dones = np.zeros((chunks, size, 10), dtype=bool)
    successes = np.zeros_like(dones)
    for member in range(size):
        successful = first_success is not None and (all_success or member % 8 == 0)
        length = first_success if successful else chunks * 10
        dones[(length - 1) // 10, member, (length - 1) % 10] = True
        successes[(length - 1) // 10, member, (length - 1) % 10] = successful
    context.update(
        action_dones=dones, action_successes=successes, action_valid=np.ones_like(dones)
    )
    return hidden, context


@pytest.mark.parametrize("score", ["volume", "volume_exp_neg_dim"])
def test_group_prefix_uses_all_173_points_but_only_17_full_chunks(score):
    if score == "volume_exp_neg_dim":
        pytest.importorskip("skdim")
    x, context = _group_rollout(chunks=18, first_success=173)
    # Unexecuted / post-episode features must not enter the matched prefix.
    x[17, 0, 3:] = np.nan
    result = score_groups(x, context, score=score)
    group = result.groups[0]
    assert group.valid and group.category == "mixed"
    assert group.prefix_length == 173
    assert [member.length for member in group.members] == [173] + [180] * 7
    assert result.prefix_mask[:17].all()
    assert not result.prefix_mask[17:].any()
    values = []
    for member in group.members:
        assert member.metrics.n == 173
        expected = compute_metrics(
            x[:, member.column].reshape(-1, 1024)[:173],
            compute_dimension=score == "volume_exp_neg_dim",
        )
        assert member.metrics.volume.value == expected.volume.value
        values.append(
            expected.volume.value if score == "volume" else expected.score.value
        )
    values = np.asarray(values)
    expected_z = (values - values.mean()) / (values.std(ddof=1) + 1e-6)
    np.testing.assert_allclose(result.standardized_scores, expected_z, rtol=0, atol=0)


def test_group_identity_preserves_nonadjacent_members():
    x, context = _group_rollout(size=16)
    original = score_groups(x, context, score="volume")
    permutation = np.ravel(np.column_stack((np.arange(8), np.arange(8, 16))))
    shuffled = score_groups(
        x[:, permutation],
        {k: v[:, permutation] for k, v in context.items()},
        score="volume",
    )
    assert len(shuffled.groups) == 2
    np.testing.assert_allclose(
        shuffled.standardized_scores, original.standardized_scores[permutation]
    )
    np.testing.assert_array_equal(
        shuffled.prefix_mask, original.prefix_mask[:, permutation]
    )
    assert [member.member_id for member in shuffled.groups[0].members] == list(range(8))


@pytest.mark.parametrize(
    "field",
    [
        "member_id",
        "policy_version",
        "rollout_epoch",
        "task_id",
        "trial_id",
        "reset_state_id",
    ],
)
def test_group_rejects_duplicate_or_mixed_identity(field):
    x, context = _group_rollout()
    context[field][:, 1] = context[field][:, 0] if field == "member_id" else 1
    with pytest.raises(ValueError, match="complete local|identical"):
        score_groups(x, context, score="volume")


def test_group_rejects_incomplete_scoring_shard():
    x, context = _group_rollout()
    with pytest.raises(ValueError, match="complete local"):
        score_groups(
            x[:, :7], {k: v[:, :7] for k, v in context.items()}, score="volume"
        )


@pytest.mark.parametrize(
    "field",
    ["policy_version", "collector_step", "rollout_epoch", "chunk_id", "task_id"],
)
def test_group_rejects_identity_drift_within_episode(field):
    x, context = _group_rollout()
    context[field][1, 0] += 1
    with pytest.raises(ValueError, match="stable manifold|ordered contiguous"):
        score_groups(x, context, score="volume")


def test_first_done_prevents_later_success_from_relabelling_failure():
    x, context = _group_rollout()
    context["action_dones"][1, 1, 9] = True
    context["action_dones"][2, 1, 4] = True
    context["action_successes"][2, 1, 4] = True
    result = score_groups(x, context, score="volume")
    assert result.groups[0].valid
    member = result.groups[0].members[1]
    assert member.length == 20 and not member.success
    assert result.groups[0].prefix_length == 17


def test_early_failure_invalidates_whole_group_without_shortening_k():
    x, context = _group_rollout()
    context["action_dones"][0, 1, 3] = True
    result = score_groups(x, context, score="volume")
    group = result.groups[0]
    assert group.prefix_length == 17
    assert group.reason == "member_shorter_than_prefix" and not group.valid
    assert group.members[1].length == 4
    assert not result.standardized_scores.any()


@pytest.mark.parametrize("all_success", [False, True])
def test_group_all_failed_and_all_success_are_distinct(all_success):
    x, context = _group_rollout(
        first_success=17 if all_success else None, all_success=all_success
    )
    result = score_groups(x, context, score="volume")
    group = result.groups[0]
    if all_success:
        assert group.category == "all_success" and group.valid
        assert group.prefix_length == 17
    else:
        assert group.category == "all_failed" and group.reason == "all_failed"
        assert group.prefix_length is None
        assert not result.prefix_mask.any() and not result.standardized_scores.any()


def test_one_invalid_member_never_reduces_the_normalization_group():
    x, context = _group_rollout()
    x[0, 3, 0, 0] = np.nan
    result = score_groups(x, context, score="volume")
    assert result.groups[0].reason == "invalid_member_score"
    assert result.groups[0].members[3].metrics.volume.reason == "nonfinite_input"
    assert len(result.groups[0].members) == 8
    assert not result.standardized_scores.any()


def test_selected_constant_score_has_no_geometry_signal():
    x, context = _group_rollout()
    x[:, 1:] = x[:, :1]
    result = score_groups(x, context, score="volume")
    assert result.groups[0].reason == "constant_score"
    assert result.groups[0].std == 0
    assert result.prefix_mask[0].all()
    assert not result.standardized_scores.any()


def test_group_volume_is_independent_of_missing_dimension_diagnostics():
    x, context = _group_rollout()
    with patch.object(
        importlib.metadata,
        "version",
        side_effect=importlib.metadata.PackageNotFoundError("skdim"),
    ):
        without = score_groups(x, context, score="volume", alpha=0.1)
        diagnostics = score_groups(
            x, context, score="volume", alpha=0.9, dimension_diagnostics=True
        )
    assert diagnostics.groups[0].valid
    assert (
        diagnostics.groups[0].members[0].metrics.dimension.reason
        == "tle_dependency_unavailable"
    )
    np.testing.assert_array_equal(
        diagnostics.standardized_scores, without.standardized_scores
    )


def test_volume_group_remains_valid_when_all_tle_fits_fail():
    pytest.importorskip("skdim")
    x, context = _group_rollout(first_success=24)
    points = np.zeros((24, 1024), dtype=np.float32)
    points[:12, 0] = 1
    points[12:, 1] = 1
    for member in range(8):
        padded = np.concatenate(
            (points * (member + 1), np.zeros((6, 1024), dtype=np.float32))
        )
        x[:, member] = padded.reshape(3, 10, 1024)
    volume = score_groups(x, context, score="volume", dimension_diagnostics=True)
    geometry = score_groups(x, context, score="volume_exp_neg_dim")
    assert volume.groups[0].valid
    assert all(member.metrics.volume.valid for member in volume.groups[0].members)
    assert all(
        not member.metrics.dimension.valid for member in volume.groups[0].members
    )
    assert geometry.groups[0].reason == "invalid_member_score"
    assert not geometry.standardized_scores.any()


def test_constant_volume_does_not_suppress_varying_g_score():
    pytest.importorskip("skdim")
    x, context = _group_rollout(first_success=24)
    rng = np.random.default_rng(117)
    points = rng.normal(size=(24, 3))
    points -= points.mean(axis=0)
    # Orthogonal rotations in the centered row space preserve Gram eigenvalues
    # while changing distances between individual action points.
    basis_input = rng.normal(size=(24, 24))
    basis_input[:, 0] = 1
    basis, _ = np.linalg.qr(basis_input)
    for member in range(8):
        rotation, _ = np.linalg.qr(rng.normal(size=(23, 23)))
        block = np.eye(24)
        block[1:, 1:] = rotation
        rotated = basis @ block @ basis.T @ points
        padded = np.zeros((30, 1024), dtype=np.float32)
        padded[:24, :3] = rotated
        x[:, member] = padded.reshape(3, 10, 1024)
    volume = score_groups(x, context, score="volume")
    geometry = score_groups(x, context, score="volume_exp_neg_dim")
    assert volume.groups[0].reason == "constant_score"
    assert geometry.groups[0].valid and geometry.groups[0].std > 1e-6
    assert geometry.standardized_scores.any()


@pytest.mark.parametrize(
    "kind",
    [
        "width",
        "dtype",
        "context_shape",
        "success_without_done",
        "invalid_terminal",
        "gap",
    ],
)
def test_group_rejects_broken_feature_or_action_contract(kind):
    x, context = _group_rollout()
    if kind == "width":
        x = x[..., :1023]
    elif kind == "dtype":
        x = x.astype(np.float64)
    elif kind == "context_shape":
        context["action_dones"] = context["action_dones"][..., :1]
    elif kind == "success_without_done":
        context["action_successes"][0, 1, 0] = True
    elif kind == "invalid_terminal":
        context["action_valid"][1, 0, 6] = False
    else:
        context["action_valid"][0, 1, 1] = False
    with pytest.raises(ValueError, match="Expected"):
        score_groups(x, context, score="volume")


@pytest.mark.parametrize("score", ["volume", "volume_exp_neg_dim"])
def test_both_group_scores_require_more_than_k_points(score):
    if score == "volume_exp_neg_dim":
        pytest.importorskip("skdim")
    x, context = _group_rollout(first_success=10)
    result = score_groups(x, context, score=score)
    assert result.groups[0].prefix_length == 10
    assert result.groups[0].reason == "invalid_member_score"
    assert all(
        member.metrics.volume.reason == "insufficient_points"
        for member in result.groups[0].members
    )
    assert not result.standardized_scores.any()


def _native_task_inputs(context):
    raw_dones = torch.from_numpy(context["action_dones"])
    dones = torch.cat((torch.zeros_like(raw_dones[:1]), raw_dones))
    action_mask, denominator = compute_loss_mask(dones)
    return {
        "task_type": "embodied",
        "adv_type": "grpo",
        "group_size": 8,
        "reward_type": "chunk_level",
        "num_action_chunks": 10,
        "rewards": torch.from_numpy(context["action_successes"]).float(),
        "dones": dones,
        "loss_mask": action_mask.any(-1, keepdim=True),
        "loss_mask_sum": denominator[..., -1:],
    }


@pytest.mark.parametrize("mode", ["task_plus_geometry", "geometry_only"])
@pytest.mark.parametrize("score", ["volume", "volume_exp_neg_dim"])
@pytest.mark.parametrize("eligible", ["mixed_only", "any_success"])
@pytest.mark.parametrize(
    "state",
    [
        "mixed",
        "mixed_constant",
        "mixed_invalid",
        "all_success",
        "all_success_constant",
        "all_failed",
    ],
)
def test_manifold_advantage_branch_table(mode, score, eligible, state):
    if score == "volume_exp_neg_dim":
        pytest.importorskip("skdim")
    x, context = _group_rollout(
        first_success=None if state == "all_failed" else 17,
        all_success=state.startswith("all_success"),
    )
    x[..., 4:] = 0
    if state.endswith("constant"):
        x[:, 1:] = x[:, :1]
    if state.endswith("invalid"):
        x[0, 2, 0, 0] = np.nan
    groups = score_groups(x, context, score=score)
    inputs = _native_task_inputs(context)
    denominator = inputs["loss_mask_sum"].clone()
    task = native_group_advantages(groups, **inputs)["advantages"]
    result = combine_advantages(
        task, inputs["loss_mask"], groups, mode=mode, eligible_groups=eligible
    )
    rewards = inputs["rewards"].sum((0, 2))
    expected_task = (rewards - rewards.mean()) / (rewards.std(correction=1) + 1e-6)
    expected_task = expected_task[None, :, None] * inputs["loss_mask"]
    torch.testing.assert_close(task, expected_task, rtol=0, atol=0)
    group = groups.groups[0]
    assert group.valid is (state in {"mixed", "all_success"})
    accepted = state.startswith("mixed") or (
        eligible == "any_success" and state == "all_success"
    )
    expected_geometry = torch.zeros_like(task)
    if group.valid and accepted:
        raw = np.asarray(
            [
                member.metrics.volume.value
                if score == "volume"
                else member.metrics.score.value
                for member in group.members
            ]
        )
        z = (raw - raw.mean()) / (raw.std(ddof=1) + 1e-6)
        expected_geometry[0, :, 0] = torch.from_numpy(z).float() * 0.2
    expected = expected_geometry.clone()
    if mode == "task_plus_geometry" and accepted:
        expected += expected_task
    torch.testing.assert_close(result.final, expected, rtol=0, atol=0)
    torch.testing.assert_close(result.geometry, expected_geometry, rtol=0, atol=0)
    assert (result.group_mask == accepted).all()
    assert torch.equal(result.loss_mask, inputs["loss_mask"] & accepted)
    assert torch.equal(inputs["loss_mask_sum"], denominator)
    assert not result.final.requires_grad
    assert not result.geometry[1:].any()
    if mode == "geometry_only":
        assert not result.final[1:].any()
    if state in {"mixed_constant", "mixed_invalid"} and mode == "geometry_only":
        assert not has_global_signal(result.final, result.loss_mask, "cpu")


def test_native_task_grpo_uses_explicit_group_identity_after_column_shuffle():
    x, context = _group_rollout(size=16)
    x[..., 4:] = 0
    permutation = np.ravel(np.column_stack((np.arange(8), np.arange(8, 16))))
    x = x[:, permutation]
    context = {name: value[:, permutation] for name, value in context.items()}
    groups = score_groups(x, context, score="volume")
    inputs = _native_task_inputs(context)
    task = native_group_advantages(groups, **inputs)["advantages"]
    positive = (1 - 1 / 8) / (torch.tensor([1.0] + [0.0] * 7).std() + 1e-6)
    negative = (-1 / 8) / (torch.tensor([1.0] + [0.0] * 7).std() + 1e-6)
    expected = torch.full_like(task, negative)
    expected[:, :2] = positive
    expected *= inputs["loss_mask"]
    torch.testing.assert_close(task, expected, rtol=0, atol=0)


def test_geometry_only_loss_keeps_zero_advantage_chunks_in_native_denominator():
    from rlinf.algorithms.registry import policy_loss

    x, context = _group_rollout()
    x[..., 4:] = 0
    groups = score_groups(x, context, score="volume")
    inputs = _native_task_inputs(context)
    task = native_group_advantages(groups, **inputs)["advantages"]
    combined = combine_advantages(
        task, inputs["loss_mask"], groups, mode="geometry_only"
    )
    logprobs = (
        torch.linspace(-0.001, 0.001, 24 * 70).reshape(24, 10, 7).requires_grad_()
    )
    loss, _ = policy_loss(
        task_type="embodied",
        loss_type="actor",
        reward_type="chunk_level",
        logprob_type="chunk_level",
        single_action_dim=7,
        logprobs=logprobs,
        old_logprobs=torch.zeros_like(logprobs),
        advantages=combined.final.reshape(24, 1),
        loss_mask=combined.loss_mask.reshape(24, 1),
        loss_mask_sum=inputs["loss_mask_sum"].reshape(24, 1),
        max_episode_steps=30,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    ratio = logprobs.sum((1, 2)).exp()
    expected = (
        -ratio
        * combined.final.flatten()
        * combined.loss_mask.flatten()
        / (inputs["loss_mask_sum"].flatten() / 30)
    ).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logprobs.grad).all()
    assert not logprobs.grad.reshape(3, 8, 10, 7)[1:].any()
    assert combined.loss_mask[1:].any()


def test_geometry_advantages_are_detached_without_changing_input():
    x, context = _group_rollout()
    groups = score_groups(x, context, score="volume")
    task = torch.ones(3, 8, 1, requires_grad=True)
    result = combine_advantages(
        task, torch.ones_like(task, dtype=torch.bool), groups, mode="task_plus_geometry"
    )
    assert not result.final.requires_grad and result.final.grad_fn is None
    assert torch.equal(task, torch.ones_like(task))


def test_signal_check_respects_mask_and_zero_advantages():
    values = torch.tensor([1.0, 0.0])
    assert not has_global_signal(values, torch.tensor([False, True]), "cpu")
    assert has_global_signal(values, torch.tensor([True, False]), "cpu")


def _distributed_manifold_signal(rank, init_file):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        for local_signals, expected in [
            ((False, True), True),
            ((False, False), False),
            ((True, False), True),
        ]:
            advantage = torch.tensor([float(local_signals[rank])])
            assert has_global_signal(advantage, torch.tensor([True]), "cpu") is expected
        assert not has_global_signal(
            torch.ones(1), torch.zeros(1, dtype=torch.bool), "cpu"
        )
    finally:
        dist.destroy_process_group()


def test_manifold_signal_agreement_across_real_cpu_ranks(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_manifold_signal,
        args=(str(tmp_path / "signal_init"),),
        nprocs=2,
        join=True,
    )


def test_constant_g_does_not_suppress_volume_mixed_advantage():
    pytest.importorskip("skdim")
    from scipy.optimize import brentq

    x, context = _group_rollout(first_success=24)
    rng = np.random.default_rng(117)
    points = rng.normal(size=(24, 3))
    points -= points.mean(axis=0)
    basis_input = rng.normal(size=(24, 24))
    basis_input[:, 0] = 1
    basis, _ = np.linalg.qr(basis_input)
    for member in range(8):
        rotation, _ = np.linalg.qr(rng.normal(size=(23, 23)))
        block = np.eye(24)
        block[1:, 1:] = rotation
        transformed = basis @ block @ basis.T @ points
        features = np.zeros((24, 1024), dtype=np.float32)
        features[:, :3] = transformed

        def residual(scale):
            return (
                compute_metrics((features * scale).astype(np.float32)).score.value - 1.0
            )

        scale = brentq(residual, 0.01, 10, xtol=1e-10)
        padded = np.zeros((30, 1024), dtype=np.float32)
        padded[:24] = features * scale
        x[:, member] = padded.reshape(3, 10, 1024)
    volume = score_groups(x, context, score="volume")
    geometry = score_groups(x, context, score="volume_exp_neg_dim")
    assert volume.groups[0].valid and volume.groups[0].std > 1e-6
    assert geometry.groups[0].reason == "constant_score"
    assert geometry.groups[0].std <= 1e-6
    inputs = _native_task_inputs(context)
    task = native_group_advantages(volume, **inputs)["advantages"]
    v_mix = combine_advantages(
        task, inputs["loss_mask"], volume, mode="task_plus_geometry"
    )
    g_mix = combine_advantages(
        task, inputs["loss_mask"], geometry, mode="task_plus_geometry"
    )
    g_only = combine_advantages(
        task, inputs["loss_mask"], geometry, mode="geometry_only"
    )
    assert v_mix.geometry[:2].any()
    assert not v_mix.geometry[2:].any()
    assert torch.equal(g_mix.final, task)
    assert not g_only.final.any()


def _audit_config(directory, mode="task_plus_geometry", score="volume", limit=0):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "actor": {"model": {"num_action_chunks": 10}},
            "algorithm": {
                "group_size": 8,
                "manifold": {
                    "mode": mode,
                    "geometry": {"score": score},
                    "diagnostics": {
                        "audit_dir": str(directory),
                        "debug_hidden_max_trajectories": limit,
                    },
                },
            },
        }
    )


def _scored_audit_input(mode, score="volume", state="mixed"):
    x, context = _group_rollout(
        first_success=None if state == "all_failed" else 17,
        all_success=state == "all_success",
    )
    x[..., 4:] = 0
    if state == "constant":
        x[:, 1:] = x[:, :1]
    elif state == "invalid":
        x[0, 3, 0, 0] = np.nan
    elif state == "short_failure":
        context["action_dones"][0, 1, 3] = True
    groups = score_groups(x, context, score=score)
    inputs = _native_task_inputs(context)
    task = native_group_advantages(groups, **inputs)["advantages"]
    combined = combine_advantages(task, inputs["loss_mask"], groups, mode=mode)
    return x, groups, combined


@pytest.mark.parametrize("mode", ["task_plus_geometry", "geometry_only"])
@pytest.mark.parametrize("score", ["volume", "volume_exp_neg_dim"])
@pytest.mark.parametrize(
    "state",
    ["mixed", "all_success", "constant", "invalid", "short_failure", "all_failed"],
)
def test_scalar_audit_replays_masks_and_advantages_without_hidden(
    tmp_path, mode, score, state
):
    if score == "volume_exp_neg_dim":
        pytest.importorskip("skdim")
    x, groups, advantages = _scored_audit_input(mode, score, state)
    writer = ManifoldAudit(_audit_config(tmp_path, mode, score), rank=0)
    path = writer.write_batch(
        groups,
        advantages,
        x,
        iteration=31,
        geometry_seconds=0.125,
        hidden_bytes=x.nbytes,
    )
    record = json.loads(path.read_text())
    provenance = path.parent / record["provenance_file"]
    assert (
        hashlib.sha256(provenance.read_bytes()).hexdigest()
        == record["provenance_sha256"]
    )
    assert json.loads(provenance.read_text())["code_sha256"]
    assert not list(tmp_path.rglob("*.npz"))
    assert record["debug_hidden"] == []
    replayed = replay_advantages(record)
    for name in advantages.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(replayed, name), getattr(advantages, name), rtol=0, atol=0
        )
    writer.write_updates(updates=1, skips=2, optimizer_steps=5, scheduler_advanced=True)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events[1]["event"] == "updates"
    assert events[1]["updates"] == 1 and events[1]["skips"] == 2


def test_hidden_debug_limit_applies_across_batches_and_restarts(tmp_path):
    x, groups, advantages = _scored_audit_input("geometry_only")
    cfg = _audit_config(tmp_path, mode="geometry_only", limit=2)
    writer = ManifoldAudit(cfg, rank=0)
    path = writer.write_batch(
        groups, advantages, x, iteration=0, geometry_seconds=0, hidden_bytes=x.nbytes
    )
    first = json.loads(path.read_text())
    assert len(first["debug_hidden"]) == 2
    for item in first["debug_hidden"]:
        saved_path = path.parent / item["path"]
        assert hashlib.sha256(saved_path.read_bytes()).hexdigest() == item["sha256"]
        assert item["complete_metric_prefix"]
        with np.load(saved_path, allow_pickle=False) as data:
            np.testing.assert_array_equal(
                data["hidden"], x[:, item["column"]].reshape(-1, 1024)[:17]
            )
    for instance in (writer, ManifoldAudit(cfg, rank=0)):
        later = instance.write_batch(
            groups,
            advantages,
            x,
            iteration=1,
            geometry_seconds=0,
            hidden_bytes=x.nbytes,
        )
        assert json.loads(later.read_text())["debug_hidden"] == []
    assert len(list(tmp_path.rglob("*.npz"))) == 2
    assert len(list(tmp_path.rglob("step_*.jsonl"))) == 3


@pytest.mark.parametrize("field", ["K", "eligible", "std", "z_score"])
def test_scalar_replay_detects_inconsistent_evidence(tmp_path, field):
    x, groups, advantages = _scored_audit_input("task_plus_geometry")
    path = ManifoldAudit(_audit_config(tmp_path), 0).write_batch(
        groups, advantages, x, iteration=0, geometry_seconds=0, hidden_bytes=x.nbytes
    )
    record = json.loads(path.read_text())
    if field == "z_score":
        record["groups"][0]["members"][0][field] += 1
    elif field == "eligible":
        record["groups"][0][field] = False
    else:
        record["groups"][0][field] += 1
    with pytest.raises(ValueError, match="Scalar audit"):
        replay_advantages(record)


def test_local_diagnostic_counts_and_valid_means_conserve_members():
    x, groups, advantages = _scored_audit_input("task_plus_geometry", state="invalid")
    totals = local_diagnostics(
        groups, advantages, geometry_seconds=0.125, hidden_bytes=x.nbytes
    )
    result = aggregate_diagnostics(totals, "cpu")
    assert result["manifold/groups/raw"] == result["manifold/groups/mixed"] == 1
    assert result["manifold/groups/invalid"] == 1
    assert result["manifold/groups/eligible"] == 1
    assert result["manifold/groups/geometry"] == 0
    assert result["manifold/V/valid"] == 7 and result["manifold/V/invalid"] == 1
    assert result["manifold/V/count"] == 7
    assert result["manifold/hidden_bytes"] == x.nbytes
    assert result["manifold/K/mean"] == 17
    assert result["manifold/K/bin_11_50"] == 1
    expected = np.mean(
        [
            member.metrics.volume.value
            for member in groups.groups[0].members
            if member.metrics.volume.valid
        ]
    )
    assert result["manifold/V/mean"] == pytest.approx(expected)
    assert math.isfinite(result["manifold/V/mean"])
    assert result["manifold/advantage/geometry_l2"] == 0
    assert result["manifold/advantage/task_l2"] == result["manifold/advantage/final_l2"]
    for label in ("D", "V", "G", "R"):
        assert (
            sum(
                result[f"manifold/{label}/{state}"]
                for state in (
                    "valid",
                    "invalid",
                    "disabled",
                    "unavailable",
                    "not_computed",
                )
            )
            == 8
        )


def test_missing_metric_samples_do_not_emit_fake_zero_means():
    x, groups, advantages = _scored_audit_input("geometry_only", state="all_failed")
    result = aggregate_diagnostics(
        local_diagnostics(
            groups, advantages, geometry_seconds=0, hidden_bytes=x.nbytes
        ),
        "cpu",
    )
    assert result["manifold/groups/all_failed"] == 1
    assert result["manifold/V/count"] == 0
    assert "manifold/V/mean" not in result
    assert result["manifold/K/count"] == 0
    assert "manifold/K/mean" not in result
    assert result["manifold/geometry_chunk_fraction"] == 0


def _distributed_diagnostic_weighting(rank, init_file):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        x, context = _group_rollout(size=8 if rank == 0 else 16)
        x[..., 4:] = 0
        x *= 1 if rank == 0 else 10
        context["env_rank"][:] = rank
        if rank == 0:
            x[0, 3, 0, 0] = np.nan
        groups = score_groups(x, context, score="volume")
        inputs = _native_task_inputs(context)
        task = native_group_advantages(groups, **inputs)["advantages"]
        advantages = combine_advantages(
            task, inputs["loss_mask"], groups, mode="task_plus_geometry"
        )
        local = local_diagnostics(
            groups, advantages, geometry_seconds=rank + 0.5, hidden_bytes=x.nbytes
        )
        evidence = [None, None]
        dist.all_gather_object(
            evidence,
            {
                "volume": local.values["V"],
                "norm": local.counts["advantage/final_squared"],
                "hidden": x.nbytes,
            },
        )
        result = aggregate_diagnostics(local, "cpu")
        expected = [value for entry in evidence for value in entry["volume"]]
        assert result["manifold/groups/raw"] == 3
        assert result["manifold/groups/valid"] == 2
        assert result["manifold/groups/invalid"] == 1
        assert result["manifold/V/valid"] == 23
        assert result["manifold/V/invalid"] == 1
        assert result["manifold/V/reason/nonfinite_input"] == 1
        assert result["manifold/V/mean"] == pytest.approx(np.mean(expected), abs=1e-12)
        assert not np.isclose(
            result["manifold/V/mean"],
            np.mean([np.mean(entry["volume"]) for entry in evidence]),
        )
        assert result["manifold/advantage/final_l2"] == pytest.approx(
            math.sqrt(sum(entry["norm"] for entry in evidence))
        )
        assert result["manifold/hidden_bytes"] == sum(
            entry["hidden"] for entry in evidence
        )
        assert result["manifold/geometry_seconds/max"] == 1.5
        assert result["manifold/geometry_seconds/mean"] == 1
    finally:
        dist.destroy_process_group()


def test_diagnostic_means_are_weighted_across_real_scoring_ranks(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_diagnostic_weighting,
        args=(str(tmp_path / "diagnostic_init"),),
        nprocs=2,
        join=True,
    )


def test_audit_default_directory_is_inside_the_experiment(tmp_path):
    cfg = _audit_config(tmp_path)
    cfg.algorithm.manifold.diagnostics = {}
    cfg.runner = {
        "logger": {"log_path": str(tmp_path / "logs"), "experiment_name": "trial"}
    }
    x, groups, advantages = _scored_audit_input("task_plus_geometry")
    writer = ManifoldAudit(cfg, rank=2)
    path = writer.write_batch(
        groups, advantages, x, iteration=0, geometry_seconds=0, hidden_bytes=x.nbytes
    )
    assert path.parent == tmp_path / "logs" / "trial" / "manifold" / "rank_00002"
    assert not list(tmp_path.rglob("*.npz"))


def test_scalar_replay_preserves_nonadjacent_member_columns(tmp_path):
    x, context = _group_rollout(size=16)
    x[..., 4:] = 0
    order = np.ravel(np.column_stack((np.arange(8), np.arange(8, 16))))
    x = x[:, order]
    context = {name: value[:, order] for name, value in context.items()}
    groups = score_groups(x, context, score="volume")
    inputs = _native_task_inputs(context)
    task = native_group_advantages(groups, **inputs)["advantages"]
    advantages = combine_advantages(
        task, inputs["loss_mask"], groups, mode="task_plus_geometry"
    )
    path = ManifoldAudit(_audit_config(tmp_path), 0).write_batch(
        groups, advantages, x, iteration=0, geometry_seconds=0, hidden_bytes=x.nbytes
    )
    result = replay_advantages(json.loads(path.read_text()))
    torch.testing.assert_close(result.final, advantages.final, rtol=0, atol=0)
    assert json.loads(path.read_text())["groups"][0]["members"][1]["column"] == 2
