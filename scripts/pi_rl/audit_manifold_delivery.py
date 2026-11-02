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

"""Audit the collected action/state/frame contract and the final offline joins once."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from manifold_tasks import goal_values, outcome


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run
    task_id = json.loads((run / "manifests/run.json").read_text()).get("task_id", 3)
    labels_path = run / "annotations/annotations_frozen.json"
    frozen = json.loads(labels_path.read_text())
    labels = frozen["records"]
    assert sorted(r["trial_id"] for r in labels) == list(range(50))
    freeze = json.loads((run / "annotations/freeze_record.json").read_text())
    label_hash = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    assert label_hash == freeze["sha256"]
    assert (
        json.loads((run / "geometry/provenance.json").read_text())["labels_sha256"]
        == label_hash
    )
    videos = read_csv(run / "videos/video_index.csv")
    assert sorted(int(v["trial_id"]) for v in videos) == list(range(50))
    video_map = {int(v["trial_id"]): v for v in videos}
    result = []
    anomalies = []
    for label in labels:
        trial = label["trial_id"]
        path = Path(label["attempt_path"])
        attempts = sorted(path.parent.glob("attempt_*/summary.json"))
        completed = []
        for candidate in attempts:
            summary = json.loads(candidate.read_text())
            if summary["status"] == "completed":
                completed.append(candidate.parent)
            else:
                anomalies.append(
                    {
                        "trial_id": trial,
                        "path": str(candidate),
                        "status": summary["status"],
                    }
                )
        assert path == completed[0], (trial, "attempt selection")
        summary = json.loads((path / "summary.json").read_text())
        states = json.loads((path / "states.json").read_text())
        indices = json.loads((path / "indices.json").read_text())
        T = summary["T"]
        assert 1 <= T <= 480
        assert summary["policy_seed"] == summary["env_seed"] == 7 + task_id * 50 + trial
        assert summary["episode_id"] == task_id * 50 + trial
        assert summary["settling_steps"] == 15 and summary["reset_gripper_open"]
        assert summary["num_steps"] == 4 and summary["action_chunk"] == 10
        assert len(states) == T + 1 and len(indices) == T
        assert [s["state_index"] for s in states] == list(range(T + 1))
        assert [s["frame_index"] for s in states] == list(range(T + 1))
        success_states = [s["state_index"] for s in states if s["success"]]
        assert success_states == ([T] if summary["success"] else []), (
            trial,
            success_states,
        )
        assert summary["first_success_step"] == (T if summary["success"] else None)
        assert summary["termination_reason"] == (
            "success" if summary["success"] else "action_budget"
        )
        assert summary["success"] or T == 480
        assert all(s["success"] == all(s["predicates"]) for s in states)
        assert label["T"] == T and label["success"] == summary["success"]
        assert summary["task_id"] == task_id
        expected_class, expected_level = outcome(
            task_id, goal_values(states[-1], task_id), summary["success"]
        )
        if label["label"] not in ["unknown", "disputed", "invalid"]:
            assert label["label"] == expected_level
            assert label.get("outcome_class", expected_class) == expected_class
        assert label["review_status"] == "reviewed" and label["visual_evidence"]
        for i, row in enumerate(indices):
            assert row == {
                "action_index": i,
                "state_before": i,
                "state_after": i + 1,
                "chunk_id": i // 10,
                "slot": i % 10,
                "executed": True,
                "denoise_step_index": 3,
            }
        with np.load(path / "trajectory.npz") as data:
            assert data["actions"].shape == (T, 7)
            assert data["hidden"].shape == (T, 1024)
            assert data["hidden"].dtype == np.float32
            assert data["model_actions"].shape == (T, 32)
            assert len(data["sim_states"]) == T + 1
            assert data["execution_mask"].shape == (T,) and data["execution_mask"].all()
            assert np.array_equal(
                data["terminal_mask"], np.array([s["success"] for s in states[1:]])
            )
            assert all(
                np.isfinite(data[key]).all()
                for key in ["actions", "hidden", "model_actions", "sim_states"]
            )
        video = video_map[trial]
        assert video["label"] == label["label"]
        for view in ["main", "wrist"]:
            assert sorted(p.name for p in (path / view).glob("*.png")) == [
                f"{i:04d}.png" for i in range(T + 1)
            ]
            source = Path(video[view + "_source"])
            target = Path(video[view + "_video"])
            assert source == path / f"{view}.mp4"
            assert source.stat().st_size == target.stat().st_size > 0
            with imageio.get_reader(target) as reader:
                assert reader.count_frames() == T + 1, (trial, view)
        result.append(
            {
                "trial_id": trial,
                "episode_id": task_id * 50 + trial,
                "T": T,
                "success": summary["success"],
                "label": label["label"],
                "attempt_path": str(path),
            }
        )
    modes = {}
    for mode in ["prefix200_k10", "full_k10"]:
        rows = read_csv(run / "geometry" / mode / "trajectory_metrics.csv")
        assert sorted(int(r["trial_id"]) for r in rows) == list(range(50))
        for r in rows:
            original = next(s for s in result if s["trial_id"] == int(r["trial_id"]))
            T = original["T"]
            assert int(r["T"]) == T and r["label"] == original["label"]
            if mode == "prefix200_k10" and T < 200:
                assert r["status"] == "excluded" and r["reason"] == "T_less_than_200"
                assert all(r[k] == "" for k in ["D", "V", "R", "G"])
            else:
                assert int(r["N"]) == (200 if mode == "prefix200_k10" else T)
                if r["status"] == "valid":
                    D, V, R, G = [float(r[k]) for k in ["D", "V", "R", "G"]]
                    assert all(math.isfinite(x) for x in [D, V, R, G])
                    assert math.isclose(R, D / V) and math.isclose(
                        G, V * math.exp(-0.1 * D)
                    )
                else:
                    assert r["reason"]
        stats = read_csv(run / "geometry" / mode / "group_statistics.csv")
        level_path = run / "geometry" / mode / "level_statistics.csv"
        if level_path.exists():
            stats += read_csv(level_path)
        for s in stats:
            group = s["group"]
            selected = [
                r
                for r in rows
                if (
                    r["success"] == "True"
                    if group == "success"
                    else r["success"] == "False"
                    if group == "failure"
                    else r["label"] == group or r.get("outcome_class") == group
                )
            ]
            vals = np.array(
                [float(r[s["metric"]]) for r in selected if r["status"] == "valid"]
            )
            assert int(s["total_n"]) == len(selected) and int(s["valid_n"]) == len(vals)
            for key, value in [
                ("mean", np.mean(vals) if len(vals) else None),
                ("std", np.std(vals, ddof=1) if len(vals) > 1 else None),
                ("median", np.median(vals) if len(vals) else None),
                ("q25", np.quantile(vals, 0.25) if len(vals) else None),
                ("q75", np.quantile(vals, 0.75) if len(vals) else None),
            ]:
                assert (
                    s[key] == ""
                    if value is None
                    else math.isclose(float(s[key]), value)
                )
        modes[mode] = {
            "valid": sum(r["status"] == "valid" for r in rows),
            "excluded": sum(r["status"] == "excluded" for r in rows),
            "invalid": sum(r["status"] == "invalid" for r in rows),
        }
    summary = {
        "total": 50,
        "successes": sum(r["success"] for r in result),
        "trajectories": result,
        "technical_anomalies": anomalies,
    }
    (run / "reports/rollout_summary.json").write_text(json.dumps(summary, indent=2))
    audit = {
        "passed": True,
        "coverage": list(range(50)),
        "videos_checked": 100,
        "geometry": modes,
        "labels_sha256": label_hash,
    }
    (run / "reports/delivery_audit.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
