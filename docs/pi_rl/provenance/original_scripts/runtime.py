"""Load the official native policy and retain its checkpoint-loading evidence."""

from __future__ import annotations

import collections
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path


def configure(run: Path) -> dict:
    state = json.loads((run / "manifests/run.json").read_text())
    repo = Path(state["rlinf_repo"])
    sys.path.insert(0, str(repo))
    os.environ["EMBODIED_PATH"] = str(repo / "examples/embodiment")
    os.environ["LIBERO_CONFIG_PATH"] = str(run / "runtime/libero_config")
    os.environ["ROBOT_PLATFORM"] = "LIBERO"
    os.environ["LIBERO_TYPE"] = "standard"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    temporary = run / "runtime/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temporary)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(run / "runtime/torch_extensions")
    device_label = os.environ.get("CUDA_VISIBLE_DEVICES", "all").replace(",", "_")
    compile_root = run / "runtime/compile" / device_label
    for variable, subdir in [
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
    ]:
        cache = compile_root / subdir
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache)
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[key] = "1"
    return state


def evaluation_config(state: dict):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    repo = Path(state["rlinf_repo"])
    with initialize_config_dir(
        config_dir=str(repo / "evaluations/libero"), version_base="1.1"
    ):
        cfg = compose(
            config_name="libero_spatial_openpi_pi05_eval",
            overrides=[
                "rollout.model.model_path="
                + json.dumps(state["sft_path"], ensure_ascii=False),
                "env.eval.video_cfg.save_video=false",
            ],
        )
    OmegaConf.resolve(cfg)
    return cfg


def load_native_model(model_cfg, checkpoint: Path | None = None):
    import torch
    from rlinf.models.embodiment.openpi import get_model
    from rlinf.models.embodiment.openpi.checkpoint import load_full_weights

    loads = []
    original = torch.nn.Module.load_state_dict

    def record_load(module, state_dict, *args, **kwargs):
        result = original(module, state_dict, *args, **kwargs)
        if type(module).__module__.startswith("rlinf.models.embodiment.openpi"):
            loads.append(
                {
                    "module": type(module).__name__,
                    "source_tensor_count": len(state_dict),
                    "matched_tensor_count": len(state_dict)
                    - len(result.unexpected_keys),
                    "missing_keys": list(result.missing_keys),
                    "unexpected_keys": list(result.unexpected_keys),
                }
            )
        return result

    torch.nn.Module.load_state_dict = record_load
    try:
        model = get_model(model_cfg)
        if checkpoint is not None:
            load_full_weights(model, checkpoint, expect_rlt=False)
    finally:
        torch.nn.Module.load_state_dict = original
    dtypes = collections.Counter()
    trainable = []
    for name, parameter in model.named_parameters():
        dtypes[str(parameter.dtype)] += parameter.numel()
        if parameter.requires_grad:
            trainable.append(
                {
                    "name": name,
                    "numel": parameter.numel(),
                    "dtype": str(parameter.dtype),
                }
            )
    audit = {
        "loads": loads,
        "model_class": type(model).__name__,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(x["numel"] for x in trainable),
        "trainable_parameter_names": trainable,
        "parameter_dtypes": dict(dtypes),
        "action_horizon": model.action_horizon,
        "action_dim": model.action_dim,
        "action_chunk": model.action_chunk,
        "num_steps": model.num_steps,
        "rl_checkpoint": str(checkpoint) if checkpoint else None,
    }
    model = model.to("cuda:0").eval()
    return model, audit


def environment_manifest() -> dict:
    import platform

    import torch

    packages = {}
    for name in [
        "torch",
        "torchvision",
        "transformers",
        "rlinf-transformer-openpi",
        "rlinf-openpi",
        "rlinf-libero",
        "ray",
        "numpy",
        "mujoco",
        "robosuite",
        "tokenizers",
        "jax",
        "jaxlib",
        "orbax-checkpoint",
        "flash-attn",
        "triton",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    modules = {}
    for name in [
        "torch",
        "transformers",
        "openpi",
        "libero.libero",
        "mujoco",
        "robosuite",
        "ray",
    ]:
        module = importlib.import_module(name)
        modules[name] = {
            "path": getattr(module, "__file__", None),
            "version": getattr(module, "__version__", None),
        }
    import openpi
    import transformers

    replacements = []
    source = Path(openpi.__file__).parent / "models_pytorch/transformers_replace"
    target = Path(transformers.__file__).parent
    for original in sorted(source.rglob("*.py")):
        installed = target / original.relative_to(source)
        if not installed.is_file():
            raise RuntimeError(
                f"Missing official Transformers replacement: {installed}"
            )
        source_hash = hashlib.sha256(original.read_bytes()).hexdigest()
        actual_hash = hashlib.sha256(installed.read_bytes()).hexdigest()
        if source_hash != actual_hash:
            raise RuntimeError(
                f"Official Transformers replacement differs: {installed}"
            )
        replacements.append(
            {"file": str(original.relative_to(source)), "sha256": actual_hash}
        )
    return {
        "python": sys.version,
        "prefix": sys.prefix,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        "libc": platform.libc_ver(),
        "packages": packages,
        "imported_modules": modules,
        "transformers_replacements": replacements,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
    }
