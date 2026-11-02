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

"""Compatibility report adapter for the shared FP64 manifold metrics."""

from rlinf.algorithms.manifold.metrics import compute_metrics, require_tle


def metrics(features):
    """Return the original pi0.5 offline report fields using the shared core."""
    import numpy as np

    require_tle()
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 1024:
        raise ValueError(f"Expected [N,1024] features, found {x.shape}")
    values = compute_metrics(x)
    result = {
        "N": values.n,
        "k": values.k,
        "D": values.dimension.value,
        "V": values.volume.value,
        "R": values.ratio.value,
        "G": values.score.value,
        "status": "valid" if values.score.valid else "invalid",
        **values.diagnostics,
    }
    if "computed_volume" in result:
        result["V"] = result.pop("computed_volume")
    if not values.score.valid:
        # The offline report historically prioritizes TLE errors after volume
        # computation, while the training core keeps each metric's own reason.
        result["reason"] = (
            values.diagnostics.get("tle_error")
            or values.dimension.reason
            or values.volume.reason
            or values.score.reason
        )
    return result
