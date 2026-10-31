"""Ensure run-path exceptions cannot hide changes to the official RL recipe."""

import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/pi_rl/audit_config.py"
spec = importlib.util.spec_from_file_location("pi_rl_config_audit", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def baseline():
    return {
        "actor": {
            "optim": {"lr": 5e-6},
            "global_batch_size": 2048,
            "model": {"model_path": "/official/model"},
        },
        "env": {"train": {"video_cfg": {"save_video": True}}},
        "algorithm": {"group_size": 8, "adv_type": "grpo"},
        "cluster": {"placement": ["actor", "env", "rollout"]},
    }


def test_accepts_only_registered_path_and_video_values(baseline):
    candidate = copy.deepcopy(baseline)
    candidate["actor"]["model"]["model_path"] = "/run/sft"
    candidate["env"]["train"]["video_cfg"]["save_video"] = False
    allowed = {
        "actor.model.model_path": "/run/sft",
        "env.train.video_cfg.save_video": False,
    }
    assert module.audit(baseline, candidate, allowed)["passed"]
    candidate["actor"]["model"]["model_path"] = "/different/checkpoint"
    assert not module.audit(baseline, candidate, allowed)["passed"]


@pytest.mark.parametrize(
    "change", ["lr", "batch", "group", "new_field", "removed_field", "placement"]
)
def test_rejects_unregistered_recipe_changes(baseline, change):
    candidate = copy.deepcopy(baseline)
    if change == "lr":
        candidate["actor"]["optim"]["lr"] = 1e-5
    elif change == "batch":
        candidate["actor"]["global_batch_size"] = 1024
    elif change == "group":
        candidate["algorithm"]["group_size"] = 4
    elif change == "new_field":
        candidate["algorithm"]["extra_reward"] = 1
    elif change == "removed_field":
        del candidate["algorithm"]["adv_type"]
    else:
        candidate["cluster"]["placement"] = ["actor", "rollout"]
    assert not module.audit(baseline, candidate, {})["passed"]


def test_a_numeric_zero_is_not_the_video_boolean_exception(baseline):
    candidate = copy.deepcopy(baseline)
    candidate["env"]["train"]["video_cfg"]["save_video"] = 0
    assert not module.audit(
        baseline, candidate, {"env.train.video_cfg.save_video": False}
    )["passed"]
