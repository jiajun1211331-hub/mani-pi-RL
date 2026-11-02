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

"""Resolve a mani_v1 recipe in an isolated frozen-source process."""

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.config import validate_manifold_cfg


def environment_identity() -> dict:
    """Record the selected interpreter and its installed distributions."""
    return {
        "python": sys.executable,
        "prefix": sys.prefix,
        "version": sys.version,
        "packages": dict(
            sorted(
                (distribution.metadata["Name"].lower(), distribution.version)
                for distribution in importlib.metadata.distributions()
                if distribution.metadata["Name"]
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    run = args.run.resolve()
    root = run / "source"
    os.environ["EMBODIED_PATH"] = str(root / "examples/embodiment")
    with initialize_config_dir(
        config_dir=str(root / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(config_name=args.config_name, overrides=args.override)
        validate_manifold_cfg(cfg)
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        with (run / "configs/train.yaml").open("x") as stream:
            stream.write(OmegaConf.to_yaml(cfg, resolve=True))
    with (run / "manifests/environment.json").open("x") as stream:
        json.dump(environment_identity(), stream, sort_keys=True, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
