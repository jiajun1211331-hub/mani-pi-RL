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

"""Prepare metric-blind LIBERO-10 predicate candidates and dual-view review sheets."""

import argparse
import json
from pathlib import Path

from manifold_tasks import TASKS, goal_values, outcome
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    task_id = json.loads((args.run / "manifests/run.json").read_text()).get(
        "task_id", 3
    )
    output = args.run / "annotations"
    if (output / "annotations_frozen.json").exists():
        raise RuntimeError(
            "Labels are already frozen; preserve the reviewed candidates"
        )
    sheets = output / "review_sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    candidates = []
    for trial in range(50):
        attempts = sorted(
            (args.run / "rollouts" / f"trial{trial:02d}").glob("attempt_*/summary.json")
        )
        complete = [
            p for p in attempts if json.loads(p.read_text())["status"] == "completed"
        ]
        if not complete:
            continue
        path = complete[0].parent
        summary = json.loads(complete[0].read_text())
        states = json.loads((path / "states.json").read_text())
        T = summary["T"]
        values = [goal_values(state, task_id) for state in states]
        final_values = values[-1]
        success = summary["success"]
        outcome_class, label = outcome(task_id, final_values, success)
        events = []
        for k in range(2):
            events.append(
                {
                    "goal": list(TASKS[task_id]["goals"][k]),
                    "first_true_state": next(
                        (i for i, v in enumerate(values) if v[k]), None
                    ),
                    "loss_states": [
                        i
                        for i in range(1, len(values))
                        if values[i - 1][k] and not values[i][k]
                    ],
                }
            )
        row = {
            "trial_id": trial,
            "episode_id": task_id * 50 + trial,
            "task_id": task_id,
            "candidate_class": outcome_class,
            "attempt_path": str(path.resolve()),
            "T": T,
            "success": success,
            "candidate_label": label,
            "final_goal_values": final_values,
            "goal_events": events,
            "state200": states[200] if T >= 200 else None,
            "review_status": "pending",
            "terminal_frame": T,
        }
        candidates.append(row)
    (output / "candidates.json").write_text(json.dumps(candidates, indent=2) + "\n")
    for offset in range(0, len(candidates), 5):
        batch = candidates[offset : offset + 5]
        canvas = Image.new("RGB", (1024, len(batch) * 288), "white")
        draw = ImageDraw.Draw(canvas)
        for j, row in enumerate(batch):
            root = Path(row["attempt_path"])
            draw.text(
                (4, j * 288 + 4),
                f"trial {row['trial_id']:02d} | T={row['T']} | goals={row['final_goal_values']} success={row['success']} | sT-20(main,wrist), sT(main,wrist)",
                fill="black",
            )
            for column, (step, view) in enumerate(
                [
                    (max(0, row["T"] - 20), "main"),
                    (max(0, row["T"] - 20), "wrist"),
                    (row["T"], "main"),
                    (row["T"], "wrist"),
                ]
            ):
                with Image.open(root / view / f"{step:04d}.png") as frame:
                    canvas.paste(frame, (column * 256, j * 288 + 28))
        canvas.save(sheets / f"terminal_{offset:02d}_{offset + len(batch) - 1:02d}.png")


if __name__ == "__main__":
    main()
