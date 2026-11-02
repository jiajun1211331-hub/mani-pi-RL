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

"""Run one prepared snapshot with scalar-only native lifecycle observations."""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    root = run / "source"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from rlinf.algorithms.manifold.snapshot import sha256, verify_snapshot

    verify_snapshot(run)
    observer = "scripts.pi_rl.mani_v1.lifecycle_observer"
    os.environ.update(
        {
            "RLINF_EXT_MODULE": observer,
            "PIRL_OBSERVER_DIR": str(run / "runtime/observer"),
        }
    )
    __import__(observer, fromlist=["register"]).register()
    entry = root / "scripts/pi_rl/mani_v1/launch_training.py"
    spec = importlib.util.spec_from_file_location("frozen_mani_launcher", entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (run / "runtime").mkdir(exist_ok=True)
    with (run / "runtime/test_harness.json").open("x") as stream:
        json.dump(
            {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
            stream,
        )
    sys.argv = [str(entry), "--run", str(run)]
    returncode = 0
    try:
        module.main()
    except BaseException as error:
        returncode = error.code if isinstance(error, SystemExit) else 1
        raise
    finally:
        with (run / "runtime/training_exit.json").open("x") as stream:
            json.dump({"pid": os.getpid(), "returncode": returncode}, stream)


if __name__ == "__main__":
    main()
