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

"""Verify a prepared mani_v1 snapshot and enter native synchronous training."""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from rlinf.algorithms.manifold.snapshot import (
    sha256,
    verify_runtime_source,
    verify_snapshot,
)
from scripts.pi_rl.mani_v1.resolve_config import environment_identity


def verify_run(run: Path) -> dict:
    """Check immutable source/config, external weights, and interpreter packages."""
    manifest = verify_snapshot(run)
    for name, key in (
        ("model", "model_manifest_sha256"),
        ("environment", "environment_sha256"),
    ):
        path = run / f"manifests/{name}.json"
        if sha256(path) != manifest[key]:
            raise RuntimeError(f"Frozen mani_v1 {name} identity changed.")
    expected_environment = json.loads((run / "manifests/environment.json").read_text())
    if environment_identity() != expected_environment:
        raise RuntimeError(
            "Expected the prepared mani_v1 interpreter and dependency versions."
        )
    model = json.loads((run / "manifests/model.json").read_text())
    root = Path(model["path"])
    actual = {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    }
    if actual != model["files"]:
        raise RuntimeError(f"Prepared mani_v1 model changed: {root}")
    return manifest


def training_arguments(run: Path) -> list[str]:
    """Build native Hydra arguments with literal Unicode paths preserved."""
    return [
        str(run / "source/examples/embodiment/train_embodied_agent.py"),
        "--config-path",
        str(run / "configs"),
        "--config-name",
        "train",
        "hydra.run.dir=" + json.dumps(str(run / "runtime/work"), ensure_ascii=False),
        "hydra.output_subdir=null",
        "hydra.job.chdir=false",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    source = run / "source"
    if Path(__file__).resolve() != source / "scripts/pi_rl/mani_v1/launch_training.py":
        raise RuntimeError(
            "Launch the frozen scripts/pi_rl/mani_v1/launch_training.py."
        )
    manifest = verify_run(run)
    os.environ["RLINF_MANIFOLD_RUN"] = str(run)
    report = verify_runtime_source("launch")
    if args.check_only:
        from rlinf.utils.logging import get_logger

        get_logger().info(
            "Verified mani_v1 source %s; %d loaded modules",
            manifest["tree_sha256"],
            len(report["modules"]),
        )
        return
    attempt = run / "runtime/launch.json"
    with attempt.open("x") as stream:
        json.dump(
            {"pid": os.getpid(), "source_tree_sha256": manifest["tree_sha256"]}, stream
        )
    os.environ.update(
        {
            "PYTHONPATH": str(source),
            "PYTHONDONTWRITEBYTECODE": "1",
            "EMBODIED_PATH": str(source / "examples/embodiment"),
            "RLINF_CODE_WORKING_DIR": "0",
            "RLINF_PATH_ENV_MERGE_MODE": "override",
            "RAY_ADDRESS": "local",
        }
    )
    entry = source / "examples/embodiment/train_embodied_agent.py"
    spec = importlib.util.spec_from_file_location("mani_v1_training_entry", entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    verify_runtime_source("driver")
    sys.argv = training_arguments(run)
    module.main()


if __name__ == "__main__":
    main()
