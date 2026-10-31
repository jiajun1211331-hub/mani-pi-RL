#!/usr/bin/env python3
"""Check lifecycle observation through real Ray workers and CPU AdamW."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "manifests/training_source_snapshot.json").read_text())
    official = Path(manifest["directory"]) / "official"
    harness = Path(__file__).resolve().parent
    sys.path.insert(0, str(official))
    from launch_training import private_ray_environment, short_temporary_directory
    from prepare_run import sha256, write_json

    root = run / "runtime/observer_transport_probe"
    root.mkdir(exist_ok=False)
    short = Path(tempfile.mkdtemp(prefix="pirl-obs-", dir="/dev/shm"))
    os.environ.update(private_ray_environment(root))
    os.environ.pop("RAY_AUTH_TOKEN", None)
    os.environ.update(
        {
            "RAY_ADDRESS": "local",
            "RAY_TMPDIR": str(short),
            "TMPDIR": str(short_temporary_directory(root, short)),
            "RAY_USAGE_STATS_ENABLED": "0",
            "RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES": str(256 << 20),
            "RLINF_CODE_WORKING_DIR": "0",
            "RLINF_EXT_MODULE": "training_observer",
            "PIRL_OBSERVER_DIR": str(root / "records"),
            "PYTHONPATH": os.pathsep.join([str(harness), str(official)]),
        }
    )
    tempfile.tempdir = None
    import ray
    import torch
    from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
    from training_observer import TrainingObservation, register

    register()

    class CPUModelManager:
        def setup_model_and_optimizer(self):
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    "gloo", rank=self._rank, world_size=self._world_size
                )
            torch.manual_seed(43)
            self.model = torch.nn.ModuleDict(
                {"vlm": torch.nn.Linear(2, 2), "actor": torch.nn.Linear(2, 1)}
            )
            self.model["vlm"].requires_grad_(False)
            self.before = {
                key: value.clone() for key, value in self.model.state_dict().items()
            }
            self.optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=5e-6,
                betas=(0.9, 0.95),
            )
            self.optimizer_steps = 0

        def optimizer_step(self):
            norm = torch.nn.utils.clip_grad_norm_(self.model["actor"].parameters(), 1.0)
            self.optimizer.step()
            self.optimizer_steps += 1
            return norm, [g["lr"] for g in self.optimizer.param_groups]

        def run_training(self):
            inputs = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 8
            loss = (self.model["actor"](self.model["vlm"](inputs)) - 1).square().mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer_step()
            return {"loss": loss.item()}

        def save_checkpoint(self, save_path, step=0):
            torch.save(self.model.state_dict(), save_path)

        def verify_weights(self):
            return {
                "frozen_unchanged": all(
                    torch.equal(value, self.before[name])
                    for name, value in self.model.state_dict().items()
                    if name.startswith("vlm.")
                ),
                "actor_changed": not torch.equal(
                    self.model["actor"].weight, self.before["actor.weight"]
                ),
            }

    class ObservedCPUWorker(TrainingObservation, CPUModelManager, Worker):
        def __init__(self):
            Worker.__init__(self)

    try:
        cluster = Cluster(num_nodes=1)
        workers = ObservedCPUWorker.create_group().launch(
            cluster=cluster,
            name="observer_probe",
            placement_strategy=NodePlacementStrategy([0]),
        )
        workers.setup_model_and_optimizer().wait()
        metrics = workers.run_training().wait()
        workers.save_checkpoint(str(root / "weights.pt"), 1).wait()
        weights = workers.verify_weights().wait()[0]
        rows = [
            json.loads(line)
            for line in (root / "records/actor_rank0.jsonl").read_text().splitlines()
            if line
        ]
        expected = [
            "initialized",
            "optimizer_step",
            "training_completed",
            "checkpoint_saved",
        ]
        passed = (
            [row["event"] for row in rows] == expected
            and rows[0]["optimizer_topology_exact"]
            and rows[1]["finite"]
            and rows[1]["grad_norm"] > 0
            and rows[2]["finite"]
            and weights["frozen_unchanged"]
            and weights["actor_changed"]
        )
        report = {
            "passed": passed,
            "events": rows,
            "metrics": metrics,
            "weights": weights,
            "observer_module_sha256": sha256(harness / "training_observer.py"),
            "official_commit": manifest["official_commit"],
            "scope": "Real RLinf Ray worker, CPU autograd/AdamW, and checkpoint write verify observation dispatch. This small CPU model does not substitute for the official Pi05 GPU lifecycle.",
        }
        write_json(run / "reports/observer_transport_probe.json", report)
        print(json.dumps({"passed": passed, "events": expected, "weights": weights}))
        if not passed:
            raise RuntimeError("Actual Ray observation dispatch did not pass")
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
