"""Do not turn missing, repeated, or mixed evaluation episodes into a valid SR."""

import copy
import importlib.util
from pathlib import Path

import pytest
from scipy.stats import binomtest

path = Path(__file__).resolve().parents[1] / "scripts/pi_rl/summarize_eval.py"
spec = importlib.util.spec_from_file_location("pi_rl_eval_summary", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def episodes():
    return [
        {
            "protocol": "recipe-240",
            "seed": 0,
            "save_video": False,
            "checkpoint_sha256": "weights",
            "wrapper_sha256": "implementation",
            "task_id": task,
            "init_state_id": trial,
            "status": "completed",
            "success": trial == 0,
        }
        for task in [0, 1]
        for trial in [0, 1]
    ]


def test_exact_coverage_and_per_task_successes():
    result = module.summarize(episodes(), [0, 1], 2, "recipe-240", 0)
    assert result["complete"]
    assert result["n"] == 4 and result["successes"] == 2
    assert [t["success_rate"] for t in result["per_task"]] == [0.5, 0.5]
    reference = binomtest(423, 500).proportion_ci(method="wilson")
    assert module.wilson(423, 500) == pytest.approx([reference.low, reference.high])


def test_missing_and_error_episodes_do_not_reduce_the_required_denominator():
    records = episodes()
    records[-1]["status"] = "error"
    result = module.summarize(records, [0, 1], 2, "recipe-240", 0)
    assert not result["complete"]
    assert result["expected_n"] == 4 and result["n"] == 3
    assert result["missing_episodes"] == [(1, 1)]
    assert result["system_error_count"] == 1
    assert result["success_rate"] is None and result["wilson95"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_sha256", "other"),
        ("wrapper_sha256", "other"),
        ("protocol", "standalone-220"),
        ("seed", 7),
        ("save_video", True),
    ],
)
def test_rejects_mixed_protocols_or_weights(field, value):
    records = episodes()
    records[-1][field] = value
    with pytest.raises(ValueError):
        module.summarize(records, [0, 1], 2, "recipe-240", 0)


def test_rejects_duplicate_completed_episode():
    records = episodes()
    records.append(copy.deepcopy(records[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        module.summarize(records, [0, 1], 2, "recipe-240", 0)
