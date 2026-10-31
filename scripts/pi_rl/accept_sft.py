#!/usr/bin/env python3
"""Recompute SFT coverage and provenance before allowing official GRPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from prepare_run import asset_root, OFFICIAL_COMMIT, SFT_REVISION, sha256
from summarize_eval import summarize


def inspect(run: Path, *, verify_weights: bool = True) -> dict:
    state = json.loads((run / "manifests/run.json").read_text())
    failures = []
    pending = []
    results = {}
    snapshot = json.loads(
        (run / "manifests/evaluation_source_snapshot.json").read_text()
    )
    scripts = Path(snapshot["directory"])
    for filename, expected in snapshot["files"].items():
        if sha256(scripts / filename) != expected:
            failures.append(f"Frozen evaluation script changed: {filename}")
    wrapper_hash = hashlib.sha256(
        (scripts / "evaluate_spatial.py").read_bytes()
        + (scripts / "runtime.py").read_bytes()
    ).hexdigest()
    weights = json.loads((run / "manifests/sft_files.json").read_text())
    hub = json.loads((run / "manifests/hf_model_info.json").read_text())
    if weights["revision"] != SFT_REVISION or hub["sha"] != SFT_REVISION:
        failures.append("The official SFT revision changed")
    model_hash = next(
        x["sha256"] for x in weights["files"] if x["path"] == "model.safetensors"
    )
    published_hash = next(
        x["lfs"]["sha256"] for x in hub["files"] if x["path"] == "model.safetensors"
    )
    if model_hash != published_hash:
        failures.append("SFT model hash differs from the published revision")
    if verify_weights:
        for entry in weights["files"]:
            file = Path(state["sft_path"]) / entry["path"]
            if file.stat().st_size != entry["bytes"] or sha256(file) != entry["sha256"]:
                failures.append(f"SFT file changed: {entry['path']}")
        assets = json.loads((run / "manifests/libero_assets.json").read_text())
        for entry in assets["files"]:
            file = asset_root(state) / entry["path"]
            if not file.is_file() or sha256(file) != entry["sha256"]:
                failures.append(f"LIBERO input asset changed: {entry['path']}")
    repo = Path(state["rlinf_repo"])
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "rlinf", "examples", "evaluations"],
        cwd=repo,
        check=False,
    )
    if head != OFFICIAL_COMMIT or clean.returncode:
        failures.append("The actual official model/environment source changed")
    for gpu in [0, 1]:
        probe = json.loads((run / f"reports/runtime_gpu{gpu}.json").read_text())
        if not probe["passed"] or any(
            x["missing_keys"] or x["unexpected_keys"] for x in probe["model"]["loads"]
        ):
            failures.append(f"GPU {gpu} rendering/loading/inference gate did not pass")
    ready = json.loads((run / "reports/native_eval_readiness.json").read_text())
    if ready.get("ready_for_native_measurement") is not True:
        failures.append("Native evaluation readiness review did not pass")
    for protocol, seed, budget, settling in [
        ("recipe-240", 0, 240, 15),
        ("standalone-220", 7, 220, 10),
    ]:
        root = run / "eval/sft/full" / protocol
        saved_path = root / "summary.json"
        if not saved_path.exists():
            pending.append(f"{protocol}: worker completion/summary pending")
            continue
        saved = json.loads(saved_path.read_text())
        files = [root / f"episodes_gpu{gpu}.jsonl" for gpu in [0, 1]]
        rows = [
            json.loads(line)
            for file in files
            for line in file.read_text().splitlines()
            if line
        ]
        try:
            result = summarize(rows, list(range(10)), 50, protocol, seed)
            for field in [
                "complete",
                "n",
                "successes",
                "success_rate",
                "checkpoint_sha256",
                "wrapper_sha256",
            ]:
                if result[field] != saved[field]:
                    failures.append(
                        f"{protocol}: saved summary differs from raw episodes: {field}"
                    )
            if not result["complete"] or result["n"] != 500:
                failures.append(
                    f"{protocol}: required 500-episode coverage is incomplete"
                )
            if (
                result["checkpoint_sha256"] != model_hash
                or result["wrapper_sha256"] != wrapper_hash
            ):
                failures.append(
                    f"{protocol}: evaluation checkpoint or frozen wrapper identity mismatch"
                )
            for row in rows:
                task, trial = row["task_id"], row["init_state_id"]
                expected_env_seed = (
                    (seed + (trial * 10 + task) % 2) * 250 + (trial * 10 + task) // 2
                    if protocol == "recipe-240"
                    else seed
                )
                valid = (
                    row["num_steps"] == 3
                    and row["action_chunk"] == 5
                    and row["noise_seed"] == seed * 1_000_003 + task * 50 + trial
                    and row["env_seed"] == expected_env_seed
                    and row["settling_steps"] == settling
                    and 1 <= row["policy_actions_executed"] <= budget
                    and (
                        protocol != "recipe-240"
                        or row["policy_actions_executed"] == budget
                    )
                    and (row["first_success_step"] is not None) == row["success"]
                    and (
                        row["first_success_step"] is None
                        or 1 <= row["first_success_step"] <= budget
                    )
                )
                if not valid:
                    failures.append(
                        f"{protocol}: episode protocol violation at task={task}, init={trial}"
                    )
            result["source_episode_files"] = list(map(str, files))
            result["source_episode_sha256"] = {
                str(file): sha256(file) for file in files
            }
            results[protocol] = result
        except (KeyError, ValueError) as error:
            failures.append(f"{protocol}: {error}")
    videos = [
        str(p.relative_to(run))
        for suffix in ["*.mp4", "*.avi", "*.webm", "*.gif"]
        for p in (run / "eval").rglob(suffix)
    ]
    if videos:
        failures.append(
            "Evaluation contains video artifacts despite the no-video requirement"
        )
    discrepancies = [
        f"{name}: {row['difference_from_paper_sft_pp']:+.2f} pp"
        for name, row in results.items()
        if row["difference_from_paper_sft_pp"] is not None
        and abs(row["difference_from_paper_sft_pp"]) > 3
    ]
    return {
        "passed": not failures
        and not pending
        and not discrepancies
        and len(results) == 2,
        "status": "pending"
        if pending
        else "needs_diagnosis"
        if failures or discrepancies
        else "passed",
        "official_commit": head,
        "sft_revision": SFT_REVISION,
        "checkpoint_sha256": model_hash,
        "wrapper_sha256": wrapper_hash,
        "hard_failures": failures,
        "pending": pending,
        "metric_discrepancies": discrepancies,
        "results": results,
        "legacy_production_action_equality": ready[
            "legacy_production_actions_passed_original_1e3_test"
        ],
        "scope": "Current-official-native SFT baseline acceptance. Wilson intervals do not establish full paper software equivalence. Any >3pp difference triggers diagnosis rather than automatic tolerance relaxation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument(
        "--skip-weight-rehash",
        action="store_true",
        help="Read-only progress check only; never use for training admission",
    )
    args = parser.parse_args()
    run = args.run.resolve()
    report = inspect(run, verify_weights=not args.skip_weight_rehash)
    if not args.skip_weight_rehash:
        (run / "reports/sft_acceptance.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        lines = [
            "# Spatial SFT 验收",
            "",
            f"状态：{report['status']}；通过：{report['passed']}。",
            "",
            "| 协议 | N | SR | Wilson 95% | 与论文 84.6% 的差 |",
            "|---|---:|---:|---|---:|",
        ]
        for name, row in report["results"].items():
            ci = row["wilson95"]
            rate = row["success_rate"]
            delta = row["difference_from_paper_sft_pp"]
            rate_text = f"{rate:.2%}" if rate is not None else "待测"
            ci_text = f"[{ci[0]:.2%}, {ci[1]:.2%}]" if ci is not None else "待测"
            delta_text = f"{delta:+.2f} pp" if delta is not None else "待测"
            lines.append(
                f"| {name} | {row['n']} | {rate_text} | {ci_text} | {delta_text} |"
            )
        lines += ["", "待处理项：", ""] + [
            f"- {x}"
            for x in report["pending"]
            + report["hard_failures"]
            + report["metric_discrepancies"]
        ]
        lines += [
            "",
            "旧 standalone 与当前原生路径的生产精度动作不满足 1e-3 完全一致；该差异保留在数值诊断中。本验收固定使用同一原生 batch-1 封装进行 SFT/RL 对比，不声明论文软件逐项复原。",
            "",
        ]
        (run / "reports/sft_acceptance.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "passed",
                    "status",
                    "pending",
                    "hard_failures",
                    "metric_discrepancies",
                ]
            },
            ensure_ascii=False,
        )
    )
    if not report["passed"]:
        raise SystemExit(2 if report["pending"] else 1)


if __name__ == "__main__":
    main()
