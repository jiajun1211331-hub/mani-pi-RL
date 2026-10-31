#!/usr/bin/env python3
"""Check every allocated GPU's NCCL, FlashAttention, and LIBERO EGL context."""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    from flash_attn import flash_attn_func
    from omegaconf import OmegaConf
    from rlinf.envs.sim.libero.libero_env import LiberoEnv

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    rank = dist.get_rank()
    world = dist.get_world_size()
    scalar = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
    dist.all_reduce(scalar)
    if scalar.item() != world * (world + 1) / 2:
        raise RuntimeError("NCCL all-reduce returned an incorrect sum")
    query = torch.randn(2, 16, 4, 64, device=f"cuda:{local_rank}", dtype=torch.bfloat16)
    attention = flash_attn_func(query, query, query)
    if attention.shape != query.shape or not torch.isfinite(attention).all():
        raise RuntimeError("FlashAttention GPU output is malformed or nonfinite")
    cfg = OmegaConf.load(args.run / f"configs/{args.phase}.yaml")
    cfg.env.eval.total_num_envs = 1
    cfg.env.eval.specific_reset_id = local_rank * 50
    env = LiberoEnv(cfg.env.eval, 1, 0, 1, None)
    try:
        observation, _ = env.reset()
        images = observation["main_images"]
        if images.shape[0] != 1 or images.min() == images.max():
            raise RuntimeError("EGL returned an empty or constant camera observation")
        report = {
            "passed": True,
            "rank": rank,
            "world_size": world,
            "gpu": torch.cuda.get_device_name(local_rank),
            "total_memory": torch.cuda.get_device_properties(local_rank).total_memory,
            "nccl_sum": scalar.item(),
            "flash_attention_shape": list(attention.shape),
            "main_image_shape": list(images.shape),
            "mujoco_egl_device_id": os.environ["MUJOCO_EGL_DEVICE_ID"],
        }
        target = args.run / f"train/{args.phase}/reports/collective_gpu{rank}.json"
        target.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.env.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
