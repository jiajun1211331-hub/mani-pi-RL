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

"""Check the two real debug rollouts, including partial chunk and k10 geometry."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.run / "runtime/geometry_packages"))
    import imageio.v2 as imageio
    from manifold_geometry import metrics

    checks = []
    for trial in range(2):
        path = args.run / "local_debug" / f"trial{trial:02d}" / "attempt_001"
        summary = json.loads((path / "summary.json").read_text())
        assert summary["status"] == "completed", summary
        data = np.load(path / "trajectory.npz")
        states = json.loads((path / "states.json").read_text())
        indices = json.loads((path / "indices.json").read_text())
        T = summary["T"]
        assert T == 32 or summary["success"], summary
        assert data["hidden"].shape == (T, 1024), data["hidden"].shape
        assert data["actions"].shape == (T, 7), data["actions"].shape
        assert len(data["model_actions"]) == T == len(indices)
        assert len(states) == len(data["sim_states"]) == T + 1
        assert [s["state_index"] for s in states] == list(range(T + 1))
        assert [s["frame_index"] for s in states] == list(range(T + 1))
        for i, row in enumerate(indices):
            assert row == {
                "action_index": i,
                "state_before": i,
                "state_after": i + 1,
                "chunk_id": i // 10,
                "slot": i % 10,
                "executed": True,
                "denoise_step_index": 3,
            }, row
        expected = [
            ["in", "akita_black_bowl_1", "white_cabinet_1_bottom_region"],
            ["close", "white_cabinet_1_bottom_region"],
        ]
        for state in states:
            assert sorted(state["goals"]) == sorted(expected), state
            assert state["success"] == all(state["predicates"]), state
        video_counts = {}
        for view in ["main", "wrist"]:
            assert len(list((path / view).glob("*.png"))) == T + 1
            with imageio.get_reader(path / f"{view}.mp4") as reader:
                decoded = list(reader.iter_data())
            assert len(decoded) == T + 1
            assert decoded[-1].shape == (256, 256, 3)
            video_counts[view] = len(decoded)
        geometry = metrics(data["hidden"])
        checks.append(
            {
                "trial": trial,
                "T": T,
                "shape": list(data["hidden"].shape),
                "video_frames": video_counts,
                "geometry": geometry,
            }
        )
    report = {"passed": True, "checks": checks}
    (args.run / "local_debug/validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
