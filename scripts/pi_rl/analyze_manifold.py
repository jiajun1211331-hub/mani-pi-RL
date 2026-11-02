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

"""Compute predeclared k10 summaries only after visual labels are frozen."""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"

import numpy as np  # noqa: E402 -- BLAS thread limits must precede NumPy import.


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows, name):
    if name == "success":
        return [r for r in rows if r["success"]]
    if name == "failure":
        return [r for r in rows if not r["success"]]
    return [r for r in rows if r["label"] == name or r.get("outcome_class") == name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.run / "runtime/geometry_packages"))
    import matplotlib
    from manifold_geometry import metrics

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frozen = args.run / "annotations/annotations_frozen.json"
    annotations = json.loads(frozen.read_text())
    records = annotations["records"]
    assert sorted(r["trial_id"] for r in records) == list(range(50))
    assert all(r["review_status"] == "reviewed" for r in records)
    plan = json.loads((args.run / "config/analysis_plan.json").read_text())
    primary = plan.get("primary_groups", ["level_0", "level_1", "level_2"])
    groups = list(
        dict.fromkeys(primary + ["level_0", "level_1", "level_2", "success", "failure"])
    )
    metric_names = ["D", "V", "R", "G"]
    provenance = {
        "labels_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
        "analysis_plan": plan,
        "R_definition": "D/V",
        "source": str(Path(__file__).resolve()),
    }
    (args.run / "geometry/provenance.json").write_text(json.dumps(provenance, indent=2))
    for mode in ["prefix200_k10", "full_k10"]:
        rows = []
        details = []
        for record in records:
            with np.load(Path(record["attempt_path"]) / "trajectory.npz") as data:
                X = data["hidden"]
                T = len(X)
                if mode == "prefix200_k10" and T < 200:
                    m = {
                        "N": T,
                        "D": None,
                        "V": None,
                        "R": None,
                        "G": None,
                        "status": "excluded",
                        "reason": "T_less_than_200",
                        "warnings": [],
                    }
                else:
                    m = metrics(X[:200] if mode == "prefix200_k10" else X)
            row = {
                "trial_id": record["trial_id"],
                "episode_id": record["episode_id"],
                "T": T,
                "N": m["N"],
                "label": record["label"],
                "outcome_class": record.get("outcome_class", record["label"]),
                "success": record["success"],
                "status": m["status"],
                "reason": m.get("reason", ""),
                **{k: m[k] for k in metric_names},
                "warnings": json.dumps(m["warnings"]),
            }
            rows.append(row)
            details.append({"trial_id": record["trial_id"], **m})
        output = args.run / "geometry" / mode
        output.mkdir(exist_ok=True)
        write_csv(output / "trajectory_metrics.csv", rows)
        (output / "numerical_details.json").write_text(
            json.dumps(details, indent=2, allow_nan=False)
        )
        stats = []
        for group in groups:
            selected = group_rows(rows, group)
            valid = [r for r in selected if r["status"] == "valid"]
            for metric in metric_names:
                values = np.asarray([r[metric] for r in valid])
                n = len(values)
                stats.append(
                    {
                        "group": group,
                        "metric": metric,
                        "total_n": len(selected),
                        "valid_n": n,
                        "mean": float(values.mean()) if n else None,
                        "std": float(values.std(ddof=1)) if n > 1 else None,
                        "median": float(np.median(values)) if n else None,
                        "q25": float(np.quantile(values, 0.25)) if n else None,
                        "q75": float(np.quantile(values, 0.75)) if n else None,
                    }
                )
        write_csv(
            output / "group_statistics.csv",
            [s for s in stats if s["group"] in primary + ["success", "failure"]],
        )
        write_csv(
            output / "level_statistics.csv",
            [s for s in stats if s["group"] in ["level_0", "level_1", "level_2"]],
        )
        effects = []
        for a, b in plan["comparisons"]:
            for metric in metric_names:
                x = np.array(
                    [r[metric] for r in group_rows(rows, a) if r["status"] == "valid"]
                )
                y = np.array(
                    [r[metric] for r in group_rows(rows, b) if r["status"] == "valid"]
                )
                auc = delta = d = None
                if len(x) >= 2 and len(y) >= 2:
                    diff = y[:, None] - x[None, :]
                    auc = float(np.mean((diff > 0) + 0.5 * (diff == 0)))
                    delta = 2 * auc - 1
                    pooled = np.sqrt(
                        ((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1))
                        / (len(x) + len(y) - 2)
                    )
                    d = float((y.mean() - x.mean()) / pooled) if pooled > 0 else None
                effects.append(
                    {
                        "group_a": a,
                        "group_b": b,
                        "metric": metric,
                        "n_a": len(x),
                        "n_b": len(y),
                        "auc_b_greater_than_a": auc,
                        "cliffs_delta": delta,
                        "cohens_d": d,
                    }
                )
        write_csv(output / "effect_sizes.csv", effects)
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        for ax, metric in zip(axes, metric_names):
            for j, g in enumerate(primary):
                vals = [
                    r[metric] for r in group_rows(rows, g) if r["status"] == "valid"
                ]
                if vals:
                    ax.boxplot([vals], positions=[j], widths=0.5)
                    ax.scatter([j] * len(vals), vals, s=12, alpha=0.5)
            ax.set_xticks(range(len(primary)), primary)
            ax.tick_params(axis="x", labelsize=7, labelrotation=20)
            ax.set_title("D/V" if metric == "R" else metric)
        fig.suptitle(mode)
        fig.tight_layout()
        fig.savefig(args.run / f"figures/{mode}_distributions.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for g in primary:
            selected = [r for r in group_rows(rows, g) if r["status"] == "valid"]
            for ax, x, y in [
                (axes[0], "D", "V"),
                (axes[1], "T", "V"),
                (axes[2], "T", "G"),
            ]:
                ax.scatter(
                    [r[x] for r in selected],
                    [r[y] for r in selected],
                    label=g,
                    alpha=0.7,
                )
                ax.set_xlabel(x)
                ax.set_ylabel(y)
        axes[0].legend()
        fig.suptitle(mode)
        fig.tight_layout()
        fig.savefig(args.run / f"figures/{mode}_scatter_length.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, g in enumerate(primary):
        ts = [r["T"] for r in records if r.get("outcome_class", r["label"]) == g]
        included = sum(t >= 200 for t in ts)
        ax.bar(j, included, color="tab:blue", label="T >= 200" if j == 0 else None)
        ax.bar(
            j,
            len(ts) - included,
            bottom=included,
            color="tab:orange",
            label="T < 200" if j == 0 else None,
        )
    ax.set_xticks(range(len(primary)), primary)
    ax.tick_params(axis="x", labelsize=8, labelrotation=15)
    ax.set_ylabel("trajectories")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.run / "figures/prefix200_coverage.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
