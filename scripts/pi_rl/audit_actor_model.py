#!/usr/bin/env python3
"""Inspect the actual official RL actor on CPU while evaluation owns the GPUs."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from runtime import configure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    configure(run)
    import torch
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openpi import get_model

    cfg = OmegaConf.load(run / "configs/train.resolved.yaml")
    if (
        cfg.actor.model.openpi.task != "rl"
        or not cfg.actor.model.openpi.train_expert_only
    ):
        raise RuntimeError("Expected the official RL action-expert-only model")
    model = get_model(cfg.actor.model)
    frozen_ids = set()

    def include(module):
        if module is not None:
            frozen_ids.update(id(p) for p in module.parameters())

    include(model.img)
    include(model.llm.embedder)
    for layer in model.llm.layers:
        for module in [
            layer.pre_attention_norms[0],
            layer.pre_ffw_norms[0],
            layer.mlps[0],
        ]:
            include(module)
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            include(getattr(layer.attn, name)[0])
    include(model.llm.final_norms[0])
    rows = []
    violations = []
    dtypes = {"trainable": collections.Counter(), "frozen": collections.Counter()}
    for name, parameter in model.named_parameters():
        expected_trainable = id(parameter) not in frozen_ids
        if parameter.requires_grad != expected_trainable:
            violations.append(name)
        if "value_head" in name or "lora" in name.lower():
            violations.append(name)
        role = "trainable" if parameter.requires_grad else "frozen"
        dtypes[role][str(parameter.dtype)] += parameter.numel()
        rows.append(
            {
                "name": name,
                "numel": parameter.numel(),
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "requires_grad": parameter.requires_grad,
                "device": str(parameter.device),
            }
        )
    total = sum(p.numel() for p in model.parameters())
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable or len(trainable) == len(rows):
        violations.append(
            "Expected a nonempty action-expert subset of model parameters"
        )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.actor.optim.lr,
        betas=(cfg.actor.optim.adam_beta1, cfg.actor.optim.adam_beta2),
        eps=cfg.actor.optim.adam_eps,
        weight_decay=cfg.actor.optim.weight_decay,
    )
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if optimizer_ids != {id(p) for p in trainable} or optimizer_ids & frozen_ids:
        violations.append("Optimizer parameter topology includes frozen VLM parameters")
    report = {
        "passed": not violations,
        "scope": "Actual official model factory before FSDP wrapping. CPU optimizer topology only; fused CUDA optimizer and post-FSDP dtypes require lifecycle/preflight verification.",
        "model_class": type(model).__name__,
        "total_parameters": total,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "frozen_parameters": sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        ),
        "trainable_tensors": len(trainable),
        "frozen_tensors": len(rows) - len(trainable),
        "parameter_dtypes": {k: dict(v) for k, v in dtypes.items()},
        "optimizer_topology_exact": optimizer_ids == {id(p) for p in trainable},
        "optimizer": {
            k: v for k, v in optimizer.param_groups[0].items() if k != "params"
        },
        "violations": violations,
        "parameters": rows,
    }
    output = run / "manifests/actor_parameters.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "passed",
                    "total_parameters",
                    "trainable_parameters",
                    "frozen_parameters",
                    "trainable_tensors",
                ]
            }
        )
    )
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
