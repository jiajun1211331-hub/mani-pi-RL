"""Delegate to the official actor and observe its completed lifecycle calls."""

from __future__ import annotations

import collections
import json
import math
import os
import time
from pathlib import Path

from rlinf.workers.actor.embodied_fsdp_actor_worker import (
    EmbodiedFSDPActor as OfficialFSDPActor,
)


class TrainingObservation:
    """Keep observations on methods that survive Ray's actor-class wrapping."""

    def _record(self, event: str, **values) -> None:
        import torch

        root = Path(os.environ["PIRL_OBSERVER_DIR"])
        root.mkdir(parents=True, exist_ok=True)
        rank = torch.distributed.get_rank()

        def finite_json(value):
            if isinstance(value, float) and not math.isfinite(value):
                return str(value)
            if isinstance(value, dict):
                return {key: finite_json(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [finite_json(item) for item in value]
            return value

        row = {"event": event, "rank": rank, "time": time.time(), **values}
        with (root / f"actor_rank{rank}.jsonl").open("a") as stream:
            stream.write(
                json.dumps(finite_json(row), ensure_ascii=False, allow_nan=False) + "\n"
            )

    def setup_model_and_optimizer(self):
        import torch

        result = super().setup_model_and_optimizer()
        rows = [
            {
                "name": name,
                "numel": p.numel(),
                "dtype": str(p.dtype),
                "requires_grad": p.requires_grad,
                "device": str(p.device),
            }
            for name, p in self.model.named_parameters()
        ]
        trainable = {id(p) for p in self.model.parameters() if p.requires_grad}
        optimized = {id(p) for g in self.optimizer.param_groups for p in g["params"]}
        state_dtypes = collections.Counter(
            str(value.dtype)
            for state in self.optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        )
        self._record(
            "initialized",
            parameters=rows,
            optimizer_class=type(self.optimizer).__name__,
            optimizer_groups=[
                {k: v for k, v in g.items() if k != "params"}
                for g in self.optimizer.param_groups
            ],
            optimizer_state_dtypes=dict(state_dtypes),
            optimizer_topology_exact=optimized == trainable,
        )
        return result

    def optimizer_step(self, *args, **kwargs):
        result = super().optimizer_step(*args, **kwargs)
        norm, rates = result
        scalar = float(norm)
        self._record(
            "optimizer_step",
            grad_norm=scalar,
            finite=math.isfinite(scalar)
            and all(math.isfinite(float(x)) for x in rates),
            learning_rates=[float(x) for x in rates],
            optimizer_steps=self.optimizer_steps,
        )
        return result

    def run_training(self, *args, **kwargs):
        result = super().run_training(*args, **kwargs)
        metrics = {key: float(value) for key, value in result.items()}
        self._record(
            "training_completed",
            metrics=metrics,
            finite=all(math.isfinite(value) for value in metrics.values()),
        )
        return result

    def save_checkpoint(self, save_path, step=0):
        result = super().save_checkpoint(save_path, step)
        self._record("checkpoint_saved", path=str(save_path), global_step=step)
        return result


class ObservedFSDPActor(TrainingObservation, OfficialFSDPActor):
    """Call the official actor unchanged, with lifecycle return-value records."""


def register() -> None:
    import rlinf.workers.actor.embodied_fsdp_actor_worker as actor_module

    if os.environ.get("PIRL_OBSERVER_DIR"):
        actor_module.EmbodiedFSDPActor = ObservedFSDPActor
