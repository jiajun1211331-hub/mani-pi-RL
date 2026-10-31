#!/usr/bin/env python3
"""Check the declared batch-one shard API against frozen upstream GPU fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime import configure, evaluation_config, load_native_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    state = configure(run)
    import numpy as np
    import torch
    from evaluate_spatial import canonical_recipe_env_seed, noise_seed, predict

    seeds = {
        canonical_recipe_env_seed(0, task, trial)
        for task in range(10)
        for trial in range(50)
    }
    if seeds != set(range(500)):
        raise RuntimeError(
            "Canonical two-worker environment seed mapping is not a permutation of 0..499"
        )
    derived = [noise_seed(0, task, trial) for task in range(10) for trial in range(50)]
    if len(set(derived)) != 500:
        raise RuntimeError("Per-episode policy noise seeds are not unique")
    cfg = evaluation_config(state)
    model, _ = load_native_model(cfg.rollout.model)
    results = []
    for gpu in [0, 1]:
        baseline = json.loads((run / f"reports/runtime_gpu{gpu}.json").read_text())
        fixture = torch.load(
            run / f"reports/runtime_gpu{gpu}.fixture.pt",
            map_location="cpu",
            weights_only=True,
        )
        generator = torch.Generator(device="cuda:0").manual_seed(0)
        expected_noise = torch.randn(
            fixture["noise"].shape, device="cuda:0", generator=generator
        ).cpu()
        same_noise = torch.equal(expected_noise, fixture["noise"])
        actual = predict(
            model, fixture["env_obs"], torch.Generator(device="cuda:0").manual_seed(0)
        )
        expected = np.asarray(baseline["actions"])
        difference = float(np.abs(actual - expected).max())
        results.append(
            {
                "upstream_fixture_gpu": gpu,
                "same_noise": same_noise,
                "max_action_difference": difference,
                "actions_equal": bool(
                    np.allclose(actual, expected, atol=1e-5, rtol=1e-5)
                ),
            }
        )
    report = {
        "passed": all(x["same_noise"] and x["actions_equal"] for x in results),
        "policy_batch_size": 1,
        "atol": 1e-5,
        "rtol": 1e-5,
        "canonical_environment_seeds": 500,
        "unique_episode_noise_seeds": 500,
        "comparisons": results,
        "save_video": False,
        "scope": "Identical frozen observations/noise, batch one, two physical-GPU fixtures. Batch-two numerical differences and renderer LSB variation are separate diagnostics.",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
