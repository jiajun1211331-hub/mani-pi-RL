#!/usr/bin/env python3
"""Compare matched SFT/RL episodes without mixing protocols or checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_run import sha256, write_json
from summarize_eval import summarize

PAIR_FIELDS = (
    "task_name",
    "noise_seed",
    "env_seed",
    "settling_steps",
    "num_steps",
    "action_chunk",
)


def compare(
    sft: list[dict],
    rl: list[dict],
    protocol: str,
    seed: int,
    sft_hash: str,
    rl_hash: str,
    *,
    task_ids: list[int] | None = None,
    trials: int = 50,
) -> dict:
    tasks = list(range(10)) if task_ids is None else task_ids
    before = summarize(sft, tasks, trials, protocol, seed)
    after = summarize(rl, tasks, trials, protocol, seed)
    if not before["complete"] or not after["complete"]:
        raise ValueError("Both SFT and RL need complete error-free episode coverage")
    if before["checkpoint_sha256"] != sft_hash or after["checkpoint_sha256"] != rl_hash:
        raise ValueError(
            "SFT or RL checkpoint identity differs from the admitted checkpoint"
        )
    if before["wrapper_sha256"] != after["wrapper_sha256"]:
        raise ValueError("SFT and RL evaluation implementations differ")
    left = {(row["task_id"], row["init_state_id"]): row for row in sft}
    right = {(row["task_id"], row["init_state_id"]): row for row in rl}
    pairs = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if any(a[field] != b[field] for field in PAIR_FIELDS):
            raise ValueError(
                f"Paired episode settings differ: task={key[0]}, init={key[1]}"
            )
        task, trial = key
        if protocol not in ["recipe-240", "standalone-220"]:
            raise ValueError(f"Unknown evaluation protocol: {protocol}")
        expected_env_seed = (
            (seed + (trial * 10 + task) % 2) * 250 + (trial * 10 + task) // 2
            if protocol == "recipe-240"
            else seed
        )
        if (
            a["noise_seed"] != seed * 1_000_003 + task * 50 + trial
            or a["env_seed"] != expected_env_seed
            or a["num_steps"] != 3
            or a["action_chunk"] != 5
            or a["settling_steps"] != (15 if protocol == "recipe-240" else 10)
        ):
            raise ValueError(
                f"Paired metadata disagrees with the registered protocol: {key}"
            )
        pairs.append(
            {
                "task_id": key[0],
                "init_state_id": key[1],
                "task_name": a["task_name"],
                "sft_success": a["success"],
                "rl_success": b["success"],
                "change": int(b["success"]) - int(a["success"]),
            }
        )

    def transitions(rows):
        return {
            "both_success": sum(x["sft_success"] and x["rl_success"] for x in rows),
            "gained_success": sum(
                not x["sft_success"] and x["rl_success"] for x in rows
            ),
            "regressed": sum(x["sft_success"] and not x["rl_success"] for x in rows),
            "both_failure": sum(
                not x["sft_success"] and not x["rl_success"] for x in rows
            ),
        }

    per_task = []
    for task, a, b in zip(tasks, before["per_task"], after["per_task"], strict=True):
        rows = [row for row in pairs if row["task_id"] == task]
        per_task.append(
            {
                "task_id": task,
                "task_name": rows[0]["task_name"],
                "n": len(rows),
                "sft": a,
                "rl": b,
                "improvement_pp": 100 * (b["success_rate"] - a["success_rate"]),
                "paired_transitions": transitions(rows),
            }
        )
    return {
        "complete": True,
        "protocol": protocol,
        "seed": seed,
        "n": len(pairs),
        "sft": before,
        "rl": after,
        "improvement_pp": 100 * (after["success_rate"] - before["success_rate"]),
        "rl_difference_from_paper_grpo_pp": 100 * (after["success_rate"] - 0.974),
        "paired_transitions": transitions(pairs),
        "per_task": per_task,
        "pairs": pairs,
        "scope": "Matched initial states and per-episode noise seeds for this fixed evaluation implementation. Wilson intervals describe each marginal SR; no statistical equivalence to the paper or independent seed replication is claimed.",
    }


def read_episodes(root: Path) -> list[dict]:
    return [
        json.loads(line)
        for gpu in [0, 1]
        for line in (root / f"episodes_gpu{gpu}.jsonl").read_text().splitlines()
        if line
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(run / "configs/formal.yaml")
    acceptance = json.loads((run / "reports/sft_acceptance.json").read_text())
    lifecycle = json.loads((run / "train/formal/reports/lifecycle.json").read_text())
    if not acceptance["passed"] or not lifecycle["passed"]:
        raise RuntimeError(
            "SFT acceptance and real formal training lifecycle must pass"
        )
    sft_hash = acceptance["checkpoint_sha256"]
    checkpoint = lifecycle["checkpoint"]
    if sha256(Path(checkpoint["path"])) != checkpoint["sha256"]:
        raise RuntimeError("The verified final RL checkpoint changed")
    results = {}
    for protocol, seed in [("recipe-240", 0), ("standalone-220", 7)]:
        for filename, expected in acceptance["results"][protocol][
            "source_episode_sha256"
        ].items():
            if sha256(Path(filename)) != expected:
                raise RuntimeError(f"Accepted SFT episode log changed: {filename}")
        results[protocol] = compare(
            read_episodes(run / "eval/sft/full" / protocol),
            read_episodes(run / "eval/rl/full" / protocol),
            protocol,
            seed,
            sft_hash,
            checkpoint["sha256"],
        )
        if results[protocol]["sft"]["wrapper_sha256"] != acceptance["wrapper_sha256"]:
            raise RuntimeError("Paired wrapper identity differs from accepted SFT")
    report = {
        "complete": True,
        "results": results,
        "official_commit": acceptance["official_commit"],
        "source_tree_sha256": lifecycle["source_tree_sha256"],
        "formal_config_sha256": lifecycle["config_sha256"],
        "actual_outer_steps": cfg.runner.max_epochs,
        "paper_reference": {
            "sft_spatial": 0.846,
            "grpo_spatial": 0.974,
            "paper_epochs": 500,
            "current_official_epochs": 1000,
        },
        "limitations": [
            f"Actual frozen run uses {cfg.runner.max_epochs} outer steps; the official default is 1000 and the paper table lists 500.",
            "Actual train/eval hardware must be read from their runtime manifests; the paper used H100.",
            "Legacy external OpenPI BF16 action equality did not pass the initial 1e-3 test; all main SFT/RL results use the same frozen native implementation.",
            "The 220 and 240 protocols also differ in seed and settling; their difference is not a causal estimate of the horizon effect.",
        ],
    }
    write_json(run / "reports/final_comparison.json", report)
    lines = [
        "# Spatial SFT / GRPO 配对评测",
        "",
        "| 协议 | checkpoint | N | SR | Wilson 95% | 相对本地 SFT |",
        "|---|---|---:|---:|---|---:|",
    ]
    for name, result in results.items():
        for kind in ["sft", "rl"]:
            row = result[kind]
            ci = row["wilson95"]
            delta = "—" if kind == "sft" else f"{result['improvement_pp']:+.2f} pp"
            lines.append(
                f"| {name} | {kind} | {row['n']} | {row['success_rate']:.2%} | [{ci[0]:.2%}, {ci[1]:.2%}] | {delta} |"
            )
    lines += [
        "",
        f"论文参考为 SFT 84.6%、GRPO/Flow-SDE 97.4%。本次冻结配置为 {cfg.runner.max_epochs} 外层步；官方默认 1000，论文通用表为 500 epoch。",
        "",
    ]
    for name, result in results.items():
        lines += [
            f"{name}：新增成功 {result['paired_transitions']['gained_success']} 条，退化 {result['paired_transitions']['regressed']} 条。",
            "",
            "| task | SFT | GRPO | 提升 pp | 新增成功 | 退化 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for task in result["per_task"]:
            lines.append(
                f"| {task['task_id']} | {task['sft']['success_rate']:.2%} | {task['rl']['success_rate']:.2%} | {task['improvement_pp']:+.2f} | {task['paired_transitions']['gained_success']} | {task['paired_transitions']['regressed']} |"
            )
        lines.append("")
    lines += ["实现、硬件与预算差异：", ""] + [
        f"- {item}" for item in report["limitations"]
    ]
    (run / "reports/final_comparison.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps({"complete": True, "protocols": list(results)}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
