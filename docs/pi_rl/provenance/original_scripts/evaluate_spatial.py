#!/usr/bin/env python3
"""Run deterministic task shards through the official native Pi0.5 policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

from runtime import configure, evaluation_config, load_native_model


def noise_seed(seed: int, task_id: int, init_state_id: int) -> int:
    return seed * 1_000_003 + task_id * 50 + init_state_id


def canonical_recipe_env_seed(seed: int, task_id: int, init_state_id: int) -> int:
    """Retain the initial per-env seeds of official 500-env/two-worker evaluation."""
    ordinal = init_state_id * 10 + task_id
    rank, local_index = ordinal % 2, ordinal // 2
    return (seed + rank) * 250 + local_index


def predict(model, observation, generator):
    import torch

    noise = torch.randn(
        (1, model.action_horizon, model.action_dim),
        device="cuda:0",
        generator=generator,
    )
    actions, _ = model.predict_action_batch(observation, mode="eval", noise=noise)
    if actions.shape != (1, 5, 7) or not torch.isfinite(actions).all():
        raise RuntimeError("Policy returned non-finite or malformed LIBERO actions")
    return actions.detach().cpu().numpy()


def native_episode(model, env, task_id, trial, seed, generator):
    import numpy as np

    reset_id = task_id * 50 + trial
    env.seed = canonical_recipe_env_seed(seed, task_id, trial)
    env.specific_reset_id = reset_id
    obs, _ = env.reset(reset_state_ids=np.array([reset_id]))
    if int(env.task_ids[0]) != task_id or int(env.trial_ids[0]) != trial:
        raise RuntimeError("Native LIBERO reset mapped to the wrong task/init-state")
    first_success = None
    for chunk in range(48):
        actions = predict(model, obs, generator)
        observations, _, _, _, infos = env.chunk_step(actions)
        for position, info in enumerate(infos):
            episode = info.get("final_info", info)["episode"]
            if bool(episode["success_once"][0]) and first_success is None:
                first_success = chunk * 5 + position + 1
        obs = observations[-1]
    return {
        "success": first_success is not None,
        "first_success_step": first_success,
        "policy_actions_executed": 240,
        "env_seed": canonical_recipe_env_seed(seed, task_id, trial),
        "settling_steps": 15,
    }


def wrap_raw_observation(raw, description):
    import numpy as np
    import torch
    from rlinf.envs.sim.libero.utils import quat2axisangle

    return {
        "main_images": torch.from_numpy(
            np.ascontiguousarray(raw["agentview_image"][::-1, ::-1])
        )[None],
        "wrist_images": torch.from_numpy(
            np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1])
        )[None],
        "states": torch.from_numpy(
            np.concatenate(
                [
                    raw["robot0_eef_pos"],
                    quat2axisangle(raw["robot0_eef_quat"].copy()),
                    raw["robot0_gripper_qpos"],
                ]
            )
        )[None],
        "task_descriptions": [description],
    }


def standalone_episode(model, env, initial_state, description, seed, generator):
    env.reset()
    raw = env.set_init_state(initial_state)
    for _ in range(10):
        raw, _, _, _ = env.step([0.0] * 6 + [-1.0])
    steps = 0
    success = False
    while steps < 220 and not success:
        actions = predict(model, wrap_raw_observation(raw, description), generator)[0]
        for action in actions:
            raw, _, success, _ = env.step(action.tolist())
            steps += 1
            if success or steps == 220:
                break
    return {
        "success": bool(success),
        "first_success_step": steps if success else None,
        "policy_actions_executed": steps,
        "env_seed": seed,
        "settling_steps": 10,
    }


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument(
        "--protocol", required=True, choices=["recipe-240", "standalone-220"]
    )
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 1 <= args.trials <= 50 or any(not 0 <= t < 10 for t in args.task_ids):
        parser.error("Expected Spatial task IDs 0..9 and 1..50 trials")
    if len(set(args.task_ids)) != len(args.task_ids):
        parser.error("Duplicate task IDs are not allowed")
    if args.output.exists():
        parser.error(
            "Output already exists; use a fresh attempt path to preserve evidence"
        )
    run = args.run.resolve()
    state = configure(run)
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from omegaconf import OmegaConf
    from rlinf.envs.sim.libero.libero_env import LiberoEnv

    seed = (
        args.seed
        if args.seed is not None
        else (0 if args.protocol == "recipe-240" else 7)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cfg = evaluation_config(state)
    model, model_audit = load_native_model(cfg.rollout.model, args.checkpoint)
    args.output.with_suffix(".model.json").write_text(
        json.dumps(model_audit, indent=2) + "\n"
    )
    sft = json.loads((run / "manifests/sft_files.json").read_text())
    weight_hash = next(
        f["sha256"] for f in sft["files"] if f["path"] == "model.safetensors"
    )
    if args.checkpoint:
        from prepare_run import sha256

        weight_hash = sha256(args.checkpoint)
    wrapper_hash = hashlib.sha256(
        Path(__file__).read_bytes()
        + Path(__file__).with_name("runtime.py").read_bytes()
    ).hexdigest()
    suite = benchmark.get_benchmark("libero_spatial")()
    native = None
    if args.protocol == "recipe-240":
        cfg.env.eval.specific_reset_id = args.task_ids[0] * 50
        cfg.env.eval.seed = seed
        cfg.env.eval.total_num_envs = 1
        native = LiberoEnv(cfg.env.eval, 1, 0, 1, None)
    args.output.with_suffix(".config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True)
    )
    errors = 0
    try:
        for task_id in args.task_ids:
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            standalone = None
            if args.protocol == "standalone-220":
                standalone = OffScreenRenderEnv(
                    bddl_file_name=str(
                        Path(get_libero_path("bddl_files"))
                        / task.problem_folder
                        / task.bddl_file
                    ),
                    camera_heights=256,
                    camera_widths=256,
                )
                standalone.seed(seed)
            try:
                for trial in range(args.trials):
                    derived_seed = noise_seed(seed, task_id, trial)
                    record = {
                        "protocol": args.protocol,
                        "seed": seed,
                        "noise_seed": derived_seed,
                        "task_id": task_id,
                        "task_name": task.name,
                        "init_state_id": trial,
                        "checkpoint_sha256": weight_hash,
                        "wrapper_sha256": wrapper_hash,
                        "num_steps": 3,
                        "action_chunk": 5,
                        "save_video": False,
                    }
                    started = time.monotonic()
                    generator = torch.Generator(device="cuda:0").manual_seed(
                        derived_seed
                    )
                    try:
                        if native is not None:
                            result = native_episode(
                                model, native, task_id, trial, seed, generator
                            )
                        else:
                            result = standalone_episode(
                                model,
                                standalone,
                                initial_states[trial],
                                task.language,
                                seed,
                                generator,
                            )
                        record.update(result)
                        record["status"] = "completed"
                    except Exception:
                        errors += 1
                        record.update(status="error", exception=traceback.format_exc())
                        append_jsonl(args.output, record)
                        raise
                    record["wall_seconds"] = time.monotonic() - started
                    append_jsonl(args.output, record)
                    print(
                        json.dumps(
                            {
                                k: record[k]
                                for k in [
                                    "protocol",
                                    "task_id",
                                    "init_state_id",
                                    "success",
                                    "wall_seconds",
                                ]
                            }
                        ),
                        flush=True,
                    )
            finally:
                if standalone is not None:
                    standalone.close()
    finally:
        if native is not None:
            native.env.close()
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
