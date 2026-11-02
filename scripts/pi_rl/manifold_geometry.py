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

"""FP64 action-level TLE k10 and centered Gram log-volume."""

import importlib.metadata
import math
import warnings

import numpy as np


def metrics(features):
    """Retain numerical warnings and distinguish undefined metrics from zero."""
    from skdim.id import TLE

    if importlib.metadata.version("scikit-dimension") != "0.3.7":
        raise RuntimeError("Expected scikit-dimension==0.3.7")
    X = np.asarray(features, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 1024:
        raise ValueError(f"Expected [N,1024] features, found {X.shape}")
    result = {
        "N": len(X),
        "k": 10,
        "D": None,
        "V": None,
        "R": None,
        "G": None,
        "warnings": [],
        "status": "invalid",
    }
    if len(X) <= 10 or not np.isfinite(X).all():
        result["reason"] = "insufficient_points" if len(X) <= 10 else "nonfinite_input"
        return result
    centered = X - X.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered @ centered.T)
    tolerance = 1e-12 + 1e-10 * max(float(eigenvalues[-1]), 0.0)
    result.update(
        gram_min_eigenvalue=float(eigenvalues[0]),
        gram_negative_tolerance=tolerance,
        gram_negative_count=int((eigenvalues < 0).sum()),
    )
    if eigenvalues[0] < -tolerance:
        result["reason"] = "negative_spectrum_outside_tolerance"
        return result
    result["V"] = float(
        0.5 * np.log1p((1024 / len(X)) * np.maximum(eigenvalues, 0)).sum()
    )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator = TLE(epsilon=1e-4).fit(X, n_neighbors=10, n_jobs=1)
        result["warnings"] = sorted({str(w.message) for w in caught})
        local = np.asarray(estimator.dimension_pw_)
        result["tle_local_valid_count"] = int(np.isfinite(local).sum())
        dimension = float(np.mean(local))
        if (
            not np.isfinite(local).all()
            or not math.isfinite(dimension)
            or dimension <= 0
        ):
            result["reason"] = "invalid_local_dimensions"
            return result
        result["D"] = dimension
        if result["V"] <= 0 or not math.isfinite(result["V"]):
            result["reason"] = "invalid_volume"
            return result
        result.update(
            R=dimension / result["V"],
            G=result["V"] * math.exp(-0.1 * dimension),
            status="valid",
        )
    except ValueError as error:
        result["reason"] = str(error)
    return result
