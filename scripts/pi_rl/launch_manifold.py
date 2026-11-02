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

"""Launch eight independent batch-one inference shards and retain every exit code."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    state = json.loads((args.run / "manifests/run.json").read_text())
    validation_path = Path(
        state.get(
            "collection_validation_path", args.run / "local_debug/validation.json"
        )
    )
    validation = json.loads(validation_path.read_text())
    if validation["passed"] is not True:
        raise RuntimeError(
            "Local two-GPU collection validation must pass before formal evaluation"
        )
    workers = []
    for rank in range(8):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(rank), MUJOCO_EGL_DEVICE_ID=str(rank))
        log = (args.run / "volc" / f"rank{rank}.log").open("x")
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(Path(__file__).with_name("collect_manifold.py")),
                "--run",
                str(args.run),
                "--rank",
                str(rank),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        workers.append((rank, process, log))
    (args.run / "volc/pids.json").write_text(
        json.dumps({rank: p.pid for rank, p, _ in workers})
    )
    codes = {}
    for rank, process, log in workers:
        codes[rank] = process.wait()
        log.close()
        (args.run / "volc/exit_codes.json").write_text(json.dumps(codes, indent=2))
    if any(codes.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
