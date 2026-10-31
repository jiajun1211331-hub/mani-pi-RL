"""Require genuinely paired episodes for the final SFT/RL comparison."""

import copy
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/pi_rl"))
comparison = importlib.import_module("compare_results")
final_evaluation = importlib.import_module("run_final_evaluation")


def records(checkpoint, successes):
    return [
        {
            "task_id": 0,
            "init_state_id": index,
            "task_name": "test-task",
            "protocol": "recipe-240",
            "seed": 0,
            "noise_seed": index,
            "env_seed": index * 5,
            "settling_steps": 15,
            "num_steps": 3,
            "action_chunk": 5,
            "save_video": False,
            "status": "completed",
            "success": success,
            "checkpoint_sha256": checkpoint,
            "wrapper_sha256": "same-implementation",
        }
        for index, success in enumerate(successes)
    ]


def test_reports_gains_and_regressions_instead_of_only_overall_sr():
    before = records("sft", [True, True, False, False])
    after = records("rl", [True, False, True, True])
    report = comparison.compare(
        before, after, "recipe-240", 0, "sft", "rl", task_ids=[0], trials=4
    )
    assert report["n"] == 4
    assert report["improvement_pp"] == 25
    assert report["paired_transitions"] == {
        "both_success": 1,
        "gained_success": 2,
        "regressed": 1,
        "both_failure": 0,
    }
    assert [row["change"] for row in report["pairs"]] == [0, -1, 1, 1]


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "error",
        "seed",
        "noise",
        "wrapper",
        "checkpoint",
        "task_name",
    ],
)
def test_rejects_false_pairs_and_wrong_final_checkpoint(change):
    before = records("sft", [True, False])
    after = copy.deepcopy(records("rl", [True, True]))
    if change == "missing":
        after.pop()
    elif change == "duplicate":
        after.append(copy.deepcopy(after[-1]))
    elif change == "error":
        after[-1]["status"] = "error"
    else:
        key = {
            "noise": "noise_seed",
            "wrapper": "wrapper_sha256",
            "checkpoint": "checkpoint_sha256",
        }.get(change, change)
        for row in after:
            row[key] = 9 if key in ["seed", "noise_seed"] else "changed"
    with pytest.raises(ValueError):
        comparison.compare(
            before, after, "recipe-240", 0, "sft", "rl", task_ids=[0], trials=2
        )


def test_equal_but_unregistered_noise_seeds_are_not_valid_pairs():
    before = records("sft", [True, False])
    after = records("rl", [True, True])
    for row in before + after:
        row["noise_seed"] = 99
    with pytest.raises(ValueError, match="registered protocol"):
        comparison.compare(
            before, after, "recipe-240", 0, "sft", "rl", task_ids=[0], trials=2
        )


@pytest.mark.parametrize("steps", [200, 1000])
@pytest.mark.parametrize(
    "change", ["valid", "earlier", "budget", "max_steps", "content"]
)
def test_final_evaluation_uses_the_final_official_budget_checkpoint(tmp_path, change, steps):
    full = (
        tmp_path
        / f"experiment/checkpoints/global_step_{steps}/actor/model_state_dict/full_weights.pt"
    )
    full.parent.mkdir(parents=True)
    full.write_bytes(b"checkpoint fixture")
    cfg = SimpleNamespace(
        runner=SimpleNamespace(
            max_epochs=steps,
            max_steps=-1,
            logger=SimpleNamespace(
                log_path=str(tmp_path), experiment_name="experiment"
            ),
        )
    )
    lifecycle = {"checkpoint": {"path": str(full), "sha256": comparison.sha256(full)}}
    if change == "earlier":
        lifecycle["checkpoint"]["path"] = str(full).replace(
            f"global_step_{steps}", f"global_step_{steps - 40}"
        )
    elif change == "budget":
        cfg.runner.max_epochs = 500
    elif change == "max_steps":
        cfg.runner.max_steps = 500
    elif change == "content":
        full.write_bytes(b"different weights")
    if change == "valid":
        assert final_evaluation.final_checkpoint(tmp_path, cfg, lifecycle) == full
    else:
        with pytest.raises(RuntimeError):
            final_evaluation.final_checkpoint(tmp_path, cfg, lifecycle)
