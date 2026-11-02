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

"""Freeze explicit visual reviews and create dual-view classified video copies."""

import argparse
import csv
import datetime
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    root = args.run / "annotations"
    destination = root / "annotations_frozen.json"
    if destination.exists():
        raise RuntimeError("Frozen annotations already exist")
    candidates = json.loads((root / "candidates.json").read_text())
    reviews = json.loads((root / "visual_reviews.json").read_text())
    assert sorted(r["trial_id"] for r in candidates) == list(range(50))
    assert sorted(r["trial_id"] for r in reviews) == list(range(50))
    task_id = json.loads((args.run / "manifests/run.json").read_text()).get(
        "task_id", 3
    )
    from manifold_tasks import TASKS

    review_map = {r["trial_id"]: r for r in reviews}
    records = []
    for candidate in candidates:
        review = review_map[candidate["trial_id"]]
        assert review["label"] in [
            "level_0",
            "level_1",
            "level_2",
            "unknown",
            "disputed",
            "invalid",
        ]
        assert review["reviewer"] == "Codex visual review (model; no human review)"
        assert review["visual_evidence"] and review["frames_reviewed"]
        assert candidate["T"] in review["frames_reviewed"]
        assert review["views_reviewed"] == ["main", "wrist"]
        if task_id != 3:
            assert review["outcome_class"] in TASKS[task_id]["classes"] + [
                "unknown",
                "disputed",
                "invalid",
            ]
        records.append({**candidate, **review, "review_status": "reviewed"})
    frozen = {
        "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": "same-rollout official predicates and original lossless dual-view frames",
        "formal_geometry_viewed_before_freeze": False,
        "human_review": False,
        "records": records,
    }
    destination.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n")
    fields = sorted(set().union(*(record.keys() for record in records)))
    with (root / "annotations_frozen.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    k: json.dumps(record.get(k), ensure_ascii=False)
                    if isinstance(record.get(k), (list, dict))
                    else record.get(k)
                    for k in fields
                }
            )
    index = []
    for record in records:
        folder = args.run / "videos" / record.get("outcome_class", record["label"])
        folder.mkdir(exist_ok=True)
        row = {
            "trial_id": record["trial_id"],
            "episode_id": record["episode_id"],
            "label": record["label"],
            "outcome_class": record.get("outcome_class", record["label"]),
            "T": record["T"],
        }
        for view in ["main", "wrist"]:
            source = Path(record["attempt_path"]) / f"{view}.mp4"
            target = folder / f"episode_{record['episode_id']}_{view}.mp4"
            shutil.copy2(source, target)
            row[view + "_video"] = str(target.resolve())
            row[view + "_source"] = str(source.resolve())
        index.append(row)
    with (args.run / "videos/video_index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    (root / "freeze_record.json").write_text(
        json.dumps(
            {
                "frozen_utc": frozen["frozen_utc"],
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "count": 50,
                "classified_videos": 100,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
