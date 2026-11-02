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

"""Freeze actual mani_v1 source and resolve its config without launching jobs."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rlinf.algorithms.manifold.snapshot import file_identity, sha256, source_files

SOURCE_ROOTS = (
    "rlinf/",
    "examples/",
    "scripts/",
    "requirements/",
    "ray_utils/",
    "toolkits/",
    "evaluations/",
)
SOURCE_FILES = {"pyproject.toml", "uv.lock", "LICENSE", "README.md"}
REQUIRED = (
    "rlinf/algorithms/manifold/metrics.py",
    "rlinf/algorithms/manifold/groups.py",
    "rlinf/algorithms/manifold/advantages.py",
    "rlinf/algorithms/manifold/audit.py",
    "rlinf/algorithms/manifold/snapshot.py",
    "rlinf/workers/actor/embodied_fsdp_actor_worker.py",
    "scripts/pi_rl/mani_v1/launch_training.py",
)


def git(repo: Path, *args: str) -> bytes:
    """Run a read-only Git command in the selected source repository."""
    return subprocess.check_output(["git", "-C", str(repo), *args])


def write_json(path: Path, value: dict) -> None:
    """Create a new evidence file without replacing an earlier attempt."""
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")


def freeze_source(repo: Path, run: Path, *, allow_dirty: bool = False) -> dict:
    """Copy tracked and explicit development files from one stable worktree."""
    repo, run = repo.resolve(), run.resolve()
    if Path(git(repo, "rev-parse", "--show-toplevel").decode().strip()) != repo:
        raise ValueError(f"Expected a Git repository root, found {repo}.")
    status = git(repo, "status", "--porcelain=v1", "--untracked-files=all").decode()
    if status and not allow_dirty:
        raise RuntimeError(
            "Expected clean committed source; use --allow-dirty for development."
        )
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    names = [
        os.fsdecode(name)
        for name in git(
            repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard"
        ).split(b"\0")
        if name
    ]
    selected = sorted(
        {
            name
            for name in names
            if name.startswith(SOURCE_ROOTS) or name in SOURCE_FILES
        }
    )
    identities = {}
    for name in selected:
        path = repo / name
        if path.is_symlink():
            target = path.resolve()
            if not target.is_relative_to(repo) or os.path.isabs(os.readlink(path)):
                raise ValueError(
                    f"Expected an internal relative source symlink: {path}"
                )
        elif not path.exists():
            continue
        elif not path.is_file():
            raise ValueError(f"Expected a source file, found {path}.")
        identities[name] = file_identity(path)
    missing = set(REQUIRED) - identities.keys()
    if missing:
        raise ValueError(f"Missing mani_v1 implementation files: {sorted(missing)}")
    if run.is_relative_to(repo) and str(run.relative_to(repo)).startswith(SOURCE_ROOTS):
        raise ValueError("Run directory must be outside the snapshotted source roots.")
    run.mkdir(parents=True, exist_ok=False)
    (run / "manifests").mkdir()
    (run / "configs").mkdir()
    root = run / "source"
    for name in identities:
        source, target = repo / name, root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
        else:
            shutil.copy2(source, target)
    actual = source_files(root)
    if actual != identities or any(
        file_identity(repo / name) != value for name, value in identities.items()
    ):
        raise RuntimeError(
            "Source changed while freezing; prepare a new run directory."
        )
    if (
        git(repo, "rev-parse", "HEAD").decode().strip() != head
        or git(repo, "status", "--porcelain=v1", "--untracked-files=all").decode()
        != status
    ):
        raise RuntimeError(
            "Git state changed while freezing; prepare a new run directory."
        )
    (run / "manifests/worktree.diff").write_bytes(git(repo, "diff", "--binary", "HEAD"))
    return {
        "schema_version": 1,
        "source_repo": str(repo),
        "head": head,
        "commits": git(repo, "log", "--reverse", "--format=%H %s")
        .decode()
        .splitlines(),
        "development_snapshot": bool(status),
        "allow_dirty": allow_dirty,
        "git_status": status,
        "worktree_diff_sha256": sha256(run / "manifests/worktree.diff"),
        "source_roots": list(SOURCE_ROOTS),
        "excluded_nonruntime_paths": sorted(set(names) - set(selected)),
        "files": actual,
        "tree_sha256": hashlib.sha256(
            json.dumps(actual, sort_keys=True).encode()
        ).hexdigest(),
    }


def prepare(
    repo: Path,
    run: Path,
    config_name: str,
    overrides: list[str],
    model: Path,
    *,
    allow_dirty: bool = False,
) -> dict:
    """Freeze source, resolve with that source, and record the external model."""
    run, model = run.resolve(), model.resolve()
    if not model.is_dir():
        raise ValueError(f"Expected a local model directory, found {model}.")
    owned = {
        "actor.model.model_path": str(model),
        "rollout.model.model_path": str(model),
        "runner.logger.log_path": str(run / "train/logs"),
        "runner.logger.experiment_name": run.name,
    }
    for override in overrides:
        if override.split("=", 1)[0].lstrip("+~") in owned:
            raise ValueError(f"Override is owned by --run/--model: {override}")
    manifest = freeze_source(repo, run, allow_dirty=allow_dirty)
    model_files = {
        str(path.relative_to(model)): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(model.rglob("*"))
        if path.is_file() and ".cache" not in path.relative_to(model).parts
    }
    if not any(name.endswith((".safetensors", ".pt", ".bin")) for name in model_files):
        raise ValueError(f"Expected model weights in {model}.")
    write_json(run / "manifests/model.json", {"path": str(model), "files": model_files})
    selected = [
        *overrides,
        *(
            f"{key}={json.dumps(value, ensure_ascii=False)}"
            for key, value in owned.items()
        ),
    ]
    command = [
        sys.executable,
        str(run / "source/scripts/pi_rl/mani_v1/resolve_config.py"),
        "--run",
        str(run),
        "--config-name",
        config_name,
    ]
    for value in selected:
        command.extend(["--override", value])
    environment = os.environ.copy()
    environment.update(
        {"PYTHONPATH": str(run / "source"), "PYTHONDONTWRITEBYTECODE": "1"}
    )
    subprocess.run(command, cwd=run, env=environment, check=True)
    manifest["config_name"] = config_name
    manifest["overrides"] = selected
    manifest["config_sha256"] = sha256(run / "configs/train.yaml")
    manifest["model_manifest_sha256"] = sha256(run / "manifests/model.json")
    manifest["environment_sha256"] = sha256(run / "manifests/environment.json")
    write_json(run / "manifests/source.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-repo", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    manifest = prepare(
        args.source_repo,
        args.run,
        args.config_name,
        args.override,
        args.model,
        allow_dirty=args.allow_dirty,
    )
    from rlinf.utils.logging import get_logger

    get_logger().info(
        "Prepared mani_v1 source %s in %s", manifest["tree_sha256"], args.run
    )


if __name__ == "__main__":
    main()
