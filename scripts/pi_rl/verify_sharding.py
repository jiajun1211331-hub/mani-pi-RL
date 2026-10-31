#!/usr/bin/env python3
"""Compare upstream batched execution with serial policy/environment scheduling."""

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
    state = configure(args.run.resolve())
    import numpy as np
    import torch
    from evaluate_spatial import predict
    from omegaconf import OmegaConf
    from rlinf.envs.sim.libero.libero_env import LiberoEnv

    def tensor_differences(expected, actual):
        result = {}
        for key, value in expected.items():
            if torch.is_tensor(value):
                other = actual[key]
                result[key] = {
                    "exact": torch.equal(value, other),
                    "max_absolute_difference": float(
                        (value.double() - other.double()).abs().max()
                    ),
                    "different_elements": int((value != other).sum()),
                }
        return result

    cfg = evaluation_config(state)
    cfg.env.eval.specific_reset_id = 0
    env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env.eval, resolve=True))
    batch = LiberoEnv(env_cfg, 2, 0, 1, None)
    serial = LiberoEnv(env_cfg, 1, 0, 1, None)
    report = {
        "passed": False,
        "policy_atol": 1e-3,
        "policy_rtol": 1e-3,
        "upstream_batch_size": 2,
        "serial_batch_size": 1,
        "save_video": False,
    }
    try:
        batch_obs, _ = batch.reset(reset_state_ids=np.array([0, 0]))
        snapshot = {
            k: v.clone() if torch.is_tensor(v) else list(v)
            for k, v in batch_obs.items()
        }
        model, _ = load_native_model(cfg.rollout.model)
        noises = [
            torch.randn(
                (1, model.action_horizon, model.action_dim),
                device="cuda:0",
                generator=torch.Generator(device="cuda:0").manual_seed(i),
            )
            for i in [0, 1]
        ]
        batch_actions, _ = model.predict_action_batch(
            batch_obs, noise=torch.cat(noises)
        )
        batch_actions = batch_actions.detach().clone()
        report["input_mutation_after_policy"] = tensor_differences(snapshot, batch_obs)
        batch_prepared = model.env_obs_to_observation(batch_obs)
        batch_next, batch_rewards, batch_terms, batch_truncs, _ = batch.chunk_step(
            batch_actions.cpu().numpy()
        )
        comparisons = []
        for i in [0, 1]:
            serial.seed = i
            obs, _ = serial.reset(reset_state_ids=np.array([0]))
            input_differences = tensor_differences(
                {k: v[i : i + 1] for k, v in snapshot.items() if torch.is_tensor(v)},
                obs,
            )
            prepared = model.env_obs_to_observation(obs)
            prepared_differences = tensor_differences(
                {
                    "state": batch_prepared.state[i : i + 1],
                    "prompt": batch_prepared.tokenized_prompt[i : i + 1],
                    **{
                        f"image/{k}": value[i : i + 1]
                        for k, value in batch_prepared.images.items()
                    },
                },
                {
                    "state": prepared.state,
                    "prompt": prepared.tokenized_prompt,
                    **{f"image/{k}": value for k, value in prepared.images.items()},
                },
            )
            input_equal = all(
                torch.equal(value[i : i + 1], obs[key])
                for key, value in batch_obs.items()
                if torch.is_tensor(value)
            )
            serial_actions = predict(
                model, obs, torch.Generator(device="cuda:0").manual_seed(i)
            )
            reference = batch_actions[i : i + 1].cpu().numpy()
            action_equal = bool(
                np.allclose(serial_actions, reference, atol=1e-3, rtol=1e-3)
            )
            # Feed the same actions to isolate simulator scheduling from GEMM rounding.
            serial_next, rewards, terms, truncs, _ = serial.chunk_step(reference)
            transitions_equal = all(
                torch.equal(value[i : i + 1], serial_next[-1][key])
                for key, value in batch_next[-1].items()
                if torch.is_tensor(value)
            )
            metrics_equal = all(
                torch.equal(a[i : i + 1], b)
                for a, b in [
                    (batch_rewards, rewards),
                    (batch_terms, terms),
                    (batch_truncs, truncs),
                ]
            )
            comparisons.append(
                {
                    "env_seed": i,
                    "input_equal": input_equal,
                    "actions_allclose": action_equal,
                    "max_action_difference": float(
                        np.abs(serial_actions - reference).max()
                    ),
                    "transition_equal_for_identical_actions": transitions_equal,
                    "reward_termination_equal": metrics_equal,
                    "input_differences": input_differences,
                    "prepared_input_differences": prepared_differences,
                    "transition_differences": tensor_differences(
                        {
                            k: v[i : i + 1]
                            for k, v in batch_next[-1].items()
                            if torch.is_tensor(v)
                        },
                        serial_next[-1],
                    ),
                }
            )
        report["comparisons"] = comparisons
        report["passed"] = all(
            x["input_equal"]
            and x["actions_allclose"]
            and x["transition_equal_for_identical_actions"]
            and x["reward_termination_equal"]
            for x in comparisons
        )
    finally:
        batch.env.close()
        serial.env.close()
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
