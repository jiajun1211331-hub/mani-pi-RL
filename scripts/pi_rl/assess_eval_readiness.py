#!/usr/bin/env python3
"""Assess readiness to measure the native baseline without claiming legacy bitwise parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    root = args.run.resolve() / "reports"

    def read(name):
        return json.loads((root / name).read_text())

    probes = [read(f"runtime_gpu{gpu}.json") for gpu in [0, 1]]
    serial = read("serial_schedule_parity.json")
    legacy = read("native_external_calibration.json")
    fp32 = read("native_external_fp32_highest_diagnostic.json")
    batch = read("sharding_parity_diagnostics.json")
    checks = {
        "both_native_gpu_probes": all(p["passed"] for p in probes),
        "complete_native_weight_loading": all(
            load["matched_tensor_count"] == load["source_tensor_count"]
            and not load["missing_keys"]
            and not load["unexpected_keys"]
            for p in probes
            for load in p["model"]["loads"]
        ),
        "strict_external_weight_loading": all(
            load["strict"] and not load["missing"] and not load["unexpected"]
            for load in legacy["external_load_audit"]
        ),
        "fixed_input_batch_one_shard_api": serial["passed"],
        "fp32_reference_calibration": (
            fp32["passed"]
            and fp32["fp32_diagnostic"]
            and fp32["float32_matmul_precision"] == "highest"
            and fp32["cudnn_allow_tf32"] is False
            and fp32["max_action_difference"] <= 1e-3
        ),
        "simulation_state_reward_and_termination": all(
            row["input_differences"]["states"]["exact"]
            and row["transition_differences"]["states"]["exact"]
            and row["reward_termination_equal"]
            for row in batch["comparisons"]
        ),
    }
    raster = []
    for row in batch["comparisons"]:
        for phase in ["input_differences", "transition_differences"]:
            for key in ["main_images", "wrist_images"]:
                delta = row[phase][key]
                fraction = delta["different_elements"] / (256 * 256 * 3)
                raster.append(
                    {
                        "env_seed": row["env_seed"],
                        "phase": phase,
                        "camera": key,
                        "max_uint8_difference": delta["max_absolute_difference"],
                        "changed_channel_fraction": fraction,
                    }
                )
    checks["raster_difference_is_sparse_uint8_quantization"] = all(
        x["max_uint8_difference"] <= 1 and x["changed_channel_fraction"] <= 1e-3
        for x in raster
    )
    result = {
        "ready_for_native_measurement": all(checks.values()),
        "checks": checks,
        "production_policy_batch_size": 1,
        "production_precision": None,
        "production_torch_compile": True,
        "legacy_production_actions_passed_original_1e3_test": legacy["passed"],
        "legacy_production_max_action_difference": legacy["max_action_difference"],
        "fp32_combined_diagnostic_max_action_difference": fp32["max_action_difference"],
        "raster_diagnostics": raster,
        "scope": "Readiness for a current-official-native 500-episode measurement. Does not assert legacy production action equality, batch-size bitwise equality, exact paper software reproduction, or SFT SR acceptance.",
        "comparison_rule": "Use the same native policy, batch size, episode/noise manifest, and precision for both SFT and RL. Standalone-220 is the standalone environment protocol with this common native policy.",
        "required_next_gate": "Both full 500-episode protocols, complete hard checks, and no unexplained significant SFT SR discrepancy before formal GRPO.",
        "diagnostic_limit": "The FP32/highest/eager comparison changes several precision/kernel settings together; it does not identify a single cause of mixed-precision differences. Production settings are not changed.",
    }
    (root / "native_eval_readiness.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "ready_for_native_measurement": result["ready_for_native_measurement"],
                "checks": checks,
            }
        )
    )
    if not result["ready_for_native_measurement"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
