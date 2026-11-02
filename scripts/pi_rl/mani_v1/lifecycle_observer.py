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

"""Observe bounded native GPU lifecycle tests without changing their numerics."""

import json
import os
import resource
import time
from pathlib import Path

import psutil
import torch

from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
from scripts.pi_rl.training_observer import TrainingObservation


def record(worker, role: str, event: str, **values) -> None:
    """Append observations in the current run's per-role/rank file."""
    root = Path(os.environ["PIRL_OBSERVER_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{role}_rank{worker._rank}.jsonl").open("a") as stream:
        json.dump(
            {"event": event, "rank": worker._rank, "time": time.time(), **values},
            stream,
            allow_nan=False,
        )
        stream.write("\n")


def memory(worker) -> dict:
    """Measure current RSS and process/device high-water marks."""
    return {
        "rss_bytes": psutil.Process().memory_info().rss,
        "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "device_allocated_bytes": worker.torch_platform.memory_allocated(),
        "device_peak_allocated_bytes": worker.torch_platform.max_memory_allocated(),
        "device_reserved_bytes": worker.torch_platform.memory_reserved(),
        "device_peak_reserved_bytes": worker.torch_platform.max_memory_reserved(),
    }


def assert_no_hidden(value) -> None:
    """Fail if a training input still carries high-dimensional manifold fields."""
    if isinstance(value, dict):
        if "manifold_hidden" in value or "manifold_context" in value:
            raise RuntimeError(
                "Manifold collection fields leaked into a training minibatch."
            )
        for item in value.values():
            assert_no_hidden(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_no_hidden(item)


class LifecycleActor(TrainingObservation, EmbodiedFSDPActor):
    """Use native actor methods and observe scoring/training memory boundaries."""

    def compute_advantages_and_returns(self):
        before = memory(self)
        result = super().compute_advantages_and_returns()
        assert_no_hidden(self.rollout_batch)
        record(
            self,
            "actor",
            "advantages_ready",
            before=before,
            after=memory(self),
            metrics={key: float(value) for key, value in result.items()},
        )
        return result

    def train_micro_batch(self, micro_batch, *args, **kwargs):
        assert_no_hidden(micro_batch)
        return super().train_micro_batch(micro_batch, *args, **kwargs)

    def run_training(self, *args, **kwargs):
        result = super().run_training(*args, **kwargs)
        record(
            self,
            "actor",
            "training_memory",
            memory=memory(self),
            optimizer_steps=self.optimizer_steps,
        )
        return result


class LifecycleRollout(MultiStepRolloutWorker):
    """Save one real observation and measure the actual collection outputs."""

    def predict(self, env_obs, mode="train"):
        root = Path(os.environ["PIRL_OBSERVER_DIR"])
        sample = root / f"observation_rank{self._rank}.pt"
        if mode == "train" and not sample.exists():
            root.mkdir(parents=True, exist_ok=True)
            torch.save(env_obs, sample)
        before = memory(self)
        result = super().predict(env_obs, mode=mode)
        if not hasattr(self, "_lifecycle_observed_modes"):
            self._lifecycle_observed_modes = set()
        if mode not in self._lifecycle_observed_modes:
            _, extras = result
            hidden = extras.get("manifold_hidden")
            record(
                self,
                "rollout",
                "prediction",
                mode=mode,
                before=before,
                after=memory(self),
                hidden_shape=list(hidden.shape) if hidden is not None else None,
                hidden_bytes=hidden.numel() * hidden.element_size()
                if hidden is not None
                else 0,
                hidden_detached=hidden is None or not hidden.requires_grad,
            )
            self._lifecycle_observed_modes.add(mode)
        return result


_REGISTERED = False


def register() -> None:
    """Select observation subclasses only for the explicit lifecycle test."""
    global _REGISTERED
    if _REGISTERED:
        return
    import rlinf.workers.actor.embodied_fsdp_actor_worker as actor_module
    import rlinf.workers.rollout.hf.huggingface_worker as rollout_module

    actor_module.EmbodiedFSDPActor = LifecycleActor
    rollout_module.MultiStepRolloutWorker = LifecycleRollout
    _REGISTERED = True
