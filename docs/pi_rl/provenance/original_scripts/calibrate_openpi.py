#!/usr/bin/env python3
"""Compare native and official standalone OpenPI on identical observations/noise."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from runtime import configure, evaluation_config, load_native_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--fp32-diagnostic",
        action="store_true",
        help="Isolated precision diagnostic; never used for production evaluation",
    )
    args = parser.parse_args()
    state = configure(args.run.resolve())
    import numpy as np
    import safetensors.torch
    import torch
    from openpi.models_pytorch import pi0_pytorch
    from toolkits.standalone_eval_scripts import openpi as standalone

    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    observation = fixture["env_obs"]
    noise = fixture["noise"].to("cuda:0")
    cfg = evaluation_config(state)
    if args.fp32_diagnostic:
        cfg.rollout.model.precision = "fp32"
        cfg.rollout.model.openpi.torch_compile = False
    native, audit = load_native_model(cfg.rollout.model)
    if args.fp32_diagnostic:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
    native_actions, _ = native.predict_action_batch(observation, noise=noise)
    external_loads = []

    def strict_external_load(config, weight_path):
        model_config = config.model
        if args.fp32_diagnostic:
            model_config = dataclasses.replace(model_config, dtype="float32")
        model = pi0_pytorch.PI0Pytorch(config=model_config)
        missing, unexpected = safetensors.torch.load_model(
            model, weight_path, strict=True
        )
        external_loads.append(
            {
                "missing": sorted(missing),
                "unexpected": sorted(unexpected),
                "strict": True,
            }
        )
        if args.fp32_diagnostic:
            convert = model.paligemma_with_expert.to_bfloat16_for_selected_params
            model.paligemma_with_expert.to_bfloat16_for_selected_params = lambda _: (
                convert("float32")
            )
            model.sample_actions = getattr(
                model.sample_actions, "_torchdynamo_orig_callable", model.sample_actions
            )
        return model

    original = standalone.load_pytorch
    standalone.load_pytorch = strict_external_load
    try:
        policy = standalone.create_trained_policy(
            standalone._CONFIGS_DICT["pi05_libero"],
            state["sft_path"],
            sample_kwargs={"num_steps": 3},
            pytorch_device="cuda:0",
        )
    finally:
        standalone.load_pytorch = original
    if args.fp32_diagnostic:
        # Both model constructors select TF32; override only in this isolated diagnostic.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
    example = {
        "observation/image": observation["main_images"][0].numpy(),
        "observation/wrist_image": observation["wrist_images"][0].numpy(),
        "observation/state": observation["states"][0].numpy(),
        "prompt": observation["task_descriptions"][0],
    }
    reference = policy.infer(example, noise=fixture["noise"][0].numpy())["actions"][
        :5, :7
    ]
    actual = native_actions[0].detach().cpu().numpy()
    difference = np.abs(reference - actual)
    passed = bool(
        np.isfinite(reference).all()
        and np.allclose(actual, reference, atol=1e-3, rtol=1e-3)
    )
    report = {
        "passed": passed,
        "fixture": str(args.fixture),
        "num_steps": 3,
        "action_chunk": 5,
        "atol": 1e-3,
        "rtol": 1e-3,
        "max_action_difference": float(difference.max()),
        "mean_action_difference": float(difference.mean()),
        "native_actions": actual.tolist(),
        "reference_actions": reference.tolist(),
        "native_load_audit": audit,
        "external_load_audit": external_loads,
        "save_video": False,
        "fp32_diagnostic": args.fp32_diagnostic,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ["passed", "max_action_difference", "mean_action_difference"]
            }
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
