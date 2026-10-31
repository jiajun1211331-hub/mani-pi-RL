#!/usr/bin/env python3
"""Check real EGL rendering and native policy inference without saving video."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from runtime import (
    configure,
    environment_manifest,
    evaluation_config,
    load_native_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    state = configure(run)
    import torch
    from libero.libero import benchmark
    from omegaconf import OmegaConf
    from rlinf.envs.sim.libero.libero_env import LiberoEnv

    report = {"environment": environment_manifest(), "save_video": False}
    suite = benchmark.get_benchmark("libero_spatial")()
    tasks = []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        if len(initial_states) != 50:
            raise RuntimeError(
                f"Expected 50 initial states for task {task_id}; got {len(initial_states)}"
            )
        tasks.append(
            {
                "task_id": task_id,
                "name": task.name,
                "language": task.language,
                "num_initial_states": len(initial_states),
            }
        )
    report["tasks"] = tasks
    cfg = evaluation_config(state)
    cfg.env.eval.specific_reset_id = 0
    cfg.env.eval.total_num_envs = 1
    report["eval_config"] = OmegaConf.to_container(cfg, resolve=True)
    env = LiberoEnv(
        cfg.env.eval, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None
    )
    try:
        observation, _ = env.reset()
        report["observation"] = {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": float(value.min()),
                "max": float(value.max()),
                "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            }
            for key, value in observation.items()
            if torch.is_tensor(value)
        }
        model, report["model"] = load_native_model(cfg.rollout.model)
        generator = torch.Generator(device="cuda:0").manual_seed(0)
        noise = torch.randn(
            (1, model.action_horizon, model.action_dim),
            device="cuda:0",
            generator=generator,
        )
        fixture = args.output.with_suffix(".fixture.pt")
        torch.save({"env_obs": observation, "noise": noise.cpu()}, fixture)
        report["fixed_input_fixture"] = str(fixture)
        started = time.monotonic()
        actions, _ = model.predict_action_batch(observation, noise=noise)
        torch.cuda.synchronize()
        report["first_inference_seconds"] = time.monotonic() - started
        if actions.shape != (1, 5, 7) or not torch.isfinite(actions).all():
            raise RuntimeError(
                f"Invalid policy action shape or values: {actions.shape}"
            )
        repeated, _ = model.predict_action_batch(observation, noise=noise)
        report["repeat_max_abs_difference"] = float((actions - repeated).abs().max())
        if not torch.allclose(actions, repeated, atol=1e-5, rtol=1e-5):
            raise RuntimeError("Fixed observation/noise inference is not reproducible")
        report["actions"] = actions.cpu().tolist()
        observations, rewards, terminations, truncations, infos = env.chunk_step(
            actions.cpu().numpy()
        )
        report["simulator_step"] = {
            "chunk_length": len(observations),
            "rewards": rewards.tolist(),
            "terminations": terminations.tolist(),
            "truncations": truncations.tolist(),
            "success_once": infos[-1]["episode"]["success_once"].tolist(),
        }
        report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["passed"] = True
    finally:
        env.env.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "output": str(args.output),
                "first_inference_seconds": report["first_inference_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
