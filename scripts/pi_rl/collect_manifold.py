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

"""Collect LIBERO-10 action hidden states and simultaneous visual/predicate evidence."""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from pathlib import Path

from evaluate_spatial import wrap_raw_observation
from manifold_tasks import TASKS, goal_values
from runtime import configure, load_native_model


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def collect(model, suite, trial, output, budget, task_id=3):
    """Save one attempt; never overwrite or select an attempt by its outcome."""
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    output.mkdir(parents=True, exist_ok=False)
    task = suite.get_task(task_id)
    expected_name = TASKS[task_id]["name"]
    if task.name != expected_name:
        raise ValueError(f"Expected task{task_id} {expected_name}, found {task.name}")
    initial = suite.get_task_init_states(task_id)[trial]
    np.save(output / "initial_state.npy", initial)
    seed = 7 + task_id * 50 + trial
    env = OffScreenRenderEnv(
        bddl_file_name=str(
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        ),
        camera_heights=256,
        camera_widths=256,
    )
    hidden_calls = []
    handle = model.action_out_proj.register_forward_pre_hook(
        lambda module, inputs: hidden_calls.append(inputs[0].detach().clone())
    )
    states, sim_states, actions, hidden, model_actions, indices = [], [], [], [], [], []
    frames = {"main": [], "wrist": []}
    record = {
        "task_id": task_id,
        "trial_id": trial,
        "episode_id": task_id * 50 + trial,
        "policy_seed": seed,
        "env_seed": seed,
        "settling_steps": 15,
        "reset_gripper_open": True,
        "budget": budget,
        "task_name": task.name,
        "attempt": output.name,
        "num_steps": 4,
        "action_chunk": 10,
    }
    started = time.monotonic()

    def save_state(raw, reward=None, done=None):
        goals = env.env.parsed_problem["goal_state"]
        predicates = [bool(env.env._eval_predicate(g)) for g in goals]
        i = len(states)
        states.append(
            {
                "state_index": i,
                "frame_index": i,
                "goals": goals,
                "predicates": predicates,
                "success": bool(env.check_success()),
                "reward": reward,
                "done": done,
            }
        )
        goal_values(states[-1], task_id)
        sim_states.append(env.get_sim_state().copy())
        for view, key in [
            ("main", "agentview_image"),
            ("wrist", "robot0_eye_in_hand_image"),
        ]:
            frame = np.ascontiguousarray(raw[key][::-1, ::-1])
            frames[view].append(frame)
            imageio.imwrite(output / view / f"{i:04d}.png", frame)

    try:
        for view in frames:
            (output / view).mkdir()
        env.seed(seed)
        env.reset()
        raw = env.set_init_state(initial)
        for _ in range(15):
            raw, _, _, _ = env.step([0.0] * 6 + [-1.0])
        save_state(raw)
        generator = torch.Generator(device="cuda:0").manual_seed(seed)
        success = False
        chunk = 0
        while len(actions) < budget and not success:
            hidden_calls.clear()
            obs = wrap_raw_observation(raw, task.language)
            noise = torch.randn(
                (1, model.action_horizon, model.action_dim),
                device="cuda:0",
                generator=generator,
            )
            predicted, aux = model.predict_action_batch(obs, mode="eval", noise=noise)
            if len(hidden_calls) != 4 or hidden_calls[-1].shape[-1] != 1024:
                raise ValueError(
                    f"Expected four [1,H,1024] hidden tensors, got {[tuple(h.shape) for h in hidden_calls]}"
                )
            h = hidden_calls[-1][0].cpu().numpy()
            a = predicted[0].detach().cpu().numpy()
            ma = aux["model_actions"][0].detach().cpu().numpy()
            if (
                a.shape != (10, 7)
                or not np.isfinite(a).all()
                or not np.isfinite(h).all()
            ):
                raise ValueError(
                    f"Expected finite (10,7) actions and hidden, got {a.shape}"
                )
            for slot, action in enumerate(a):
                raw, reward, done, _ = env.step(action.tolist())
                actions.append(action.copy())
                hidden.append(h[slot].copy())
                model_actions.append(ma[slot].copy())
                indices.append(
                    {
                        "action_index": len(actions) - 1,
                        "state_before": len(actions) - 1,
                        "state_after": len(actions),
                        "chunk_id": chunk,
                        "slot": slot,
                        "executed": True,
                        "denoise_step_index": 3,
                    }
                )
                save_state(raw, float(reward), bool(done))
                success = states[-1]["success"]
                if success or len(actions) == budget:
                    break
            chunk += 1
        np.savez(
            output / "trajectory.npz",
            actions=np.asarray(actions),
            hidden=np.asarray(hidden),
            model_actions=np.asarray(model_actions),
            sim_states=np.asarray(sim_states),
            execution_mask=np.ones(len(actions), dtype=bool),
            terminal_mask=np.asarray([s["success"] for s in states[1:]], dtype=bool),
        )
        write_json(output / "states.json", states)
        write_json(output / "indices.json", indices)
        for view, images in frames.items():
            with imageio.get_writer(
                output / f"{view}.mp4", fps=20, codec="libx264", quality=8
            ) as writer:
                for frame in images:
                    writer.append_data(frame)
        record.update(
            status="completed",
            T=len(actions),
            success=success,
            first_success_step=len(actions) if success else None,
            termination_reason="success" if success else "action_budget",
            final_state=states[-1],
            hidden_dtype=str(np.asarray(hidden).dtype),
        )
    except Exception:
        record.update(status="error", exception=traceback.format_exc())
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - started
        write_json(output / "summary.json", record)
        handle.remove()
        env.close()
    logging.info("Completed trial %s: T=%s success=%s", trial, len(actions), success)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    state = configure(args.run)
    task_id = state.get("task_id", 3)
    from hydra import compose, initialize_config_dir
    from libero.libero import benchmark
    from omegaconf import OmegaConf

    with initialize_config_dir(
        config_dir=str(Path(state["rlinf_repo"]) / "evaluations/libero"),
        version_base="1.1",
    ):
        cfg = compose(
            config_name="libero_10_openpi_pi05_eval",
            overrides=[
                "rollout.model.model_path="
                + json.dumps(state["sft_path"], ensure_ascii=False),
                "rollout.model.num_action_chunks=10",
                "rollout.model.num_steps=4",
                "env.eval.total_num_envs=1",
                "env.eval.auto_reset=false",
                "env.eval.ignore_terminations=false",
                "env.eval.max_episode_steps=480",
                "env.eval.max_steps_per_rollout_epoch=480",
            ],
        )
    OmegaConf.resolve(cfg)
    stage = "local_debug" if args.debug else "rollouts"
    target = args.run / stage
    target.mkdir(exist_ok=True)
    (target / f"rank{args.rank}.yaml").write_text(OmegaConf.to_yaml(cfg))
    write_json(
        target / f"rank{args.rank}_collection.json",
        {
            "checkpoint": state["sft_path"],
            "rank": args.rank,
            "world_size": args.world_size,
            "batch_size": 1,
            "budget": 32 if args.debug else 480,
            "settling_steps": 15,
            "seed_rule": "7 + task_id * 50 + trial",
            "task_id": task_id,
            "task_order_index": 0,
            "source": str(Path(__file__).resolve()),
            "hidden_source": "fourth action_out_proj forward input; native float32",
            "attempt_selection": "first technically complete attempt, regardless of success or metrics",
        },
    )
    model, audit = load_native_model(cfg.rollout.model)
    write_json(target / f"rank{args.rank}_model.json", audit)
    suite = benchmark.get_benchmark("libero_10")(task_order_index=0)
    trials = (
        [args.rank] if args.debug else list(range(50))[args.rank :: args.world_size]
    )
    for trial in trials:
        collect(
            model,
            suite,
            trial,
            target / f"trial{trial:02d}" / "attempt_001",
            32 if args.debug else 480,
            task_id=task_id,
        )


if __name__ == "__main__":
    main()
