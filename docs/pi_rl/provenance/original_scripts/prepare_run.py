#!/usr/bin/env python3
"""Prepare and audit an isolated run of the fixed official Pi0.5 recipe."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.metadata
import json
import math
import os
import shlex
import shutil
import subprocess
from pathlib import Path

OFFICIAL_COMMIT = "30e0236981c6a1dc19202de64116834c12673828"
SFT_REVISION = "45ccfcc4e28634f1576ebf78cab0fbe2fd82432d"
ASSET_ROOT = Path("/vepfs-C区/jiajunlu/LIBERO/libero/libero")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prepare(run: Path, state: dict) -> None:
    repo = Path(state["rlinf_repo"])
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(
            f"Expected official commit {OFFICIAL_COMMIT}; found {commit}"
        )
    unchanged = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "examples/embodiment/config",
            "requirements/install.sh",
            "rlinf",
        ],
        cwd=repo,
        check=False,
    )
    if unchanged.returncode != 0:
        raise RuntimeError(
            "Official configs, installer, or RLinf source differ from the fixed commit"
        )
    config_source = repo / "examples/embodiment/config"
    config_target = run / "configs/official"
    if not config_target.exists():
        shutil.copytree(config_source, config_target)
    eval_config = repo / "evaluations/libero/libero_spatial_openpi_pi05_eval.yaml"
    shutil.copy2(eval_config, run / "configs/official_spatial_eval.yaml")
    script_path = repo / "requirements/install.sh"
    original = script_path.read_text()
    old = """install_libero_env() {
    uv pip install rlinf-libero
    materialize_package_files rlinf-libero
    retry_cmd libero-download-assets --skip-existing
    reset_libero_config
}"""
    helper = Path(__file__).resolve()
    new = f"""install_libero_env() {{
    uv pip install rlinf-libero
    materialize_package_files rlinf-libero
    python {shlex.quote(str(helper))} link-assets --run {shlex.quote(str(run))}
}}"""
    if original.count(old) != 1:
        raise RuntimeError("The fixed official LIBERO installer body did not match")
    original_path_line = 'SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"'
    if original.count(original_path_line) != 1:
        raise RuntimeError("The official SCRIPT_PATH assignment did not match")
    adapted = original.replace(old, new).replace(
        original_path_line, "SCRIPT_PATH=" + shlex.quote(str(script_path))
    )
    mirror_setup = "setup_mirror() {\n    remove_stale_github_mirror_rule"
    if adapted.count(mirror_setup) != 1:
        raise RuntimeError("The official global git cleanup call did not match")
    adapted = adapted.replace(
        mirror_setup,
        "setup_mirror() {\n    : # Preserve other experiments' global git mirror configuration",
    )
    adapted_path = run / "runtime/install_reuse_assets.sh"
    adapted_path.write_text(adapted)
    (run / "manifests/install_asset_reuse.patch").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                adapted.splitlines(keepends=True),
                fromfile=str(script_path),
                tofile=str(adapted_path),
            )
        )
    )
    write_json(
        run / "manifests/official_source.json",
        {
            "commit": commit,
            "repo": str(repo),
            "install_original_sha256": sha256(script_path),
            "install_adapted_sha256": sha256(adapted_path),
            "config_files": {
                str(p.relative_to(config_source)): sha256(p)
                for p in sorted(config_source.rglob("*.yaml"))
            },
        },
    )
    lines = {
        "PIRL_RUN": str(run),
        "PIRL_REPO": str(repo),
        "PIRL_ENV": state["environment"],
        "PIRL_SFT": state["sft_path"],
        "LIBERO_CONFIG_PATH": str(run / "runtime/libero_config"),
        "DOWNLOAD_DIR": str(run / "runtime/model_assets"),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "ROBOT_PLATFORM": "LIBERO",
        "LIBERO_TYPE": "standard",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MAX_JOBS": "8",
        "UV_LINK_MODE": "copy",
        "TMPDIR": str(run / "runtime/tmp"),
        "TORCH_EXTENSIONS_DIR": str(run / "runtime/torch_extensions"),
    }
    (run / "runtime/run.env").write_text(
        "\n".join(f"export {k}={shlex.quote(v)}" for k, v in lines.items()) + "\n"
    )
    tokenizer = Path.home() / ".cache/openpi/paligemma_tokenizer.model"
    if tokenizer.is_file():
        destination = (
            run / "runtime/model_assets/.cache/openpi/paligemma_tokenizer.model"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.symlink_to(tokenizer.resolve())
        write_json(
            run / "manifests/tokenizer_reuse.json",
            {
                "source": str(tokenizer.resolve()),
                "sha256": sha256(tokenizer),
            },
        )
    print(adapted_path)


def link_assets(run: Path, state: dict) -> None:
    import yaml
    from libero import libero

    config_dir = run / "runtime/libero_config"
    if Path(os.environ.get("LIBERO_CONFIG_PATH", "")).resolve() != config_dir.resolve():
        raise RuntimeError("LIBERO_CONFIG_PATH must point at this run's private config")
    package = Path(libero.__file__).resolve().parent
    site_packages = Path(
        importlib.metadata.distribution("rlinf-libero").locate_file("")
    )
    if not package.is_relative_to(Path(state["environment"]).resolve()):
        raise RuntimeError(
            f"Refusing to adapt a package outside the isolated venv: {package}"
        )
    comparisons = []
    for group in ["bddl_files", "init_files"]:
        installed = package / group / "libero_spatial"
        existing = ASSET_ROOT / group / "libero_spatial"
        for expected in sorted(installed.iterdir()):
            if expected.suffix not in [".bddl", ".pruned_init"]:
                continue
            reused = existing / expected.name
            match = reused.is_file() and sha256(expected) == sha256(reused)
            comparisons.append(
                {
                    "group": group,
                    "file": expected.name,
                    "installed_sha256": sha256(expected),
                    "reused_sha256": sha256(reused) if reused.is_file() else None,
                    "match": match,
                }
            )
    write_json(run / "manifests/packaged_spatial_asset_comparison.json", comparisons)
    if len(comparisons) != 20 or not all(x["match"] for x in comparisons):
        raise RuntimeError(
            "Spatial BDDL/init assets differ from the installed official package"
        )
    for directory in ["articulated_objects", "scenes", "stable_hope_objects"]:
        if not (ASSET_ROOT / "assets" / directory).is_dir():
            raise RuntimeError(f"Missing reusable LIBERO assets: {directory}")
    package_assets = package / "assets"
    if not package_assets.exists():
        package_assets.symlink_to(ASSET_ROOT / "assets", target_is_directory=True)
    elif package_assets.resolve() != (ASSET_ROOT / "assets").resolve():
        raise RuntimeError(
            f"Refusing to overwrite installed asset directory {package_assets}"
        )
    cfg = {
        "benchmark_root": str(package),
        "assets": str(ASSET_ROOT / "assets"),
        "bddl_files": str(ASSET_ROOT / "bddl_files"),
        "init_states": str(ASSET_ROOT / "init_files"),
        "datasets": str(run / "runtime/unused_datasets"),
    }
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    for key, expected in cfg.items():
        if libero.get_libero_path(key) != expected:
            raise RuntimeError(f"Private LIBERO path mismatch for {key}")
    asset_files = [p for p in sorted((ASSET_ROOT / "assets").rglob("*")) if p.is_file()]
    write_json(
        run / "manifests/libero_assets.json",
        {
            "installed_version": importlib.metadata.version("rlinf-libero"),
            "installed_package": str(package),
            "site_packages": str(site_packages),
            "private_config": cfg,
            "files": [
                {
                    "path": str(p.relative_to(ASSET_ROOT)),
                    "bytes": p.stat().st_size,
                    "sha256": sha256(p),
                }
                for p in asset_files
            ],
        },
    )
    print(
        f"Reused LIBERO assets: {len(asset_files)} files; 20/20 Spatial BDDL/init hashes match"
    )


def verify_sft(run: Path, state: dict) -> None:
    model_root = Path(state["sft_path"])
    metadata = json.loads((run / "manifests/hf_model_info.json").read_text())
    if metadata["sha"] != SFT_REVISION:
        raise RuntimeError("Unexpected SFT revision")
    files = []
    for entry in metadata["files"]:
        path = model_root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise RuntimeError(f"Missing or incomplete SFT file: {path}")
        digest = sha256(path)
        expected = (entry["lfs"] or {}).get("sha256")
        if expected and digest != expected:
            raise RuntimeError(f"SFT SHA256 mismatch: {path}")
        files.append(
            {"path": entry["path"], "bytes": path.stat().st_size, "sha256": digest}
        )
    stats = model_root / "physical-intelligence/libero/norm_stats.json"
    normalization = json.loads(stats.read_text())
    dimensions = {}
    for name in ["state", "actions"]:
        values = normalization["norm_stats"][name]
        for field in ["mean", "std", "q01", "q99"]:
            if len(values[field]) != 32 or not all(
                math.isfinite(x) for x in values[field]
            ):
                raise RuntimeError(
                    f"Invalid 32-dimensional normalization statistics: {name}.{field}"
                )
        if any(x < 0 for x in values["std"]) or any(
            lo > hi for lo, hi in zip(values["q01"], values["q99"], strict=True)
        ):
            raise RuntimeError(f"Invalid normalization ranges: {name}")
        dimensions[name] = 32
    write_json(
        run / "manifests/sft_files.json",
        {
            "revision": SFT_REVISION,
            "model_root": str(model_root),
            "files": files,
            "norm_stats_keys": list(normalization),
            "norm_stats_dimensions": dimensions,
        },
    )
    print(
        json.dumps(
            {"verified_files": len(files), "norm_stats_keys": list(normalization)}
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "link-assets", "verify-sft"])
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    state = json.loads((run / "manifests/run.json").read_text())
    {"prepare": prepare, "link-assets": link_assets, "verify-sft": verify_sft}[
        args.command
    ](run, state)


if __name__ == "__main__":
    main()
