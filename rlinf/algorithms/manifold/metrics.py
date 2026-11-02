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

"""FP64 centered Gram volume and scikit-dimension 0.3.7 TLE metrics."""

import importlib.metadata
import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Metric:
    """A scalar and its independent validity; invalid values are absent."""

    value: float | None = None
    reason: str | None = None

    @property
    def valid(self) -> bool:
        """Whether this scalar can be used for scoring or aggregation."""
        return self.value is not None and self.reason is None


@dataclass(frozen=True)
class ManifoldMetrics:
    """Metrics for one matched action prefix, without group normalization."""

    n: int
    k: int
    dimension: Metric
    volume: Metric
    score: Metric
    ratio: Metric
    diagnostics: dict[str, Any] = field(default_factory=dict)


def require_tle() -> type:
    """Load the pinned optional estimator or explain the required installation."""
    try:
        version = importlib.metadata.version("scikit-dimension")
        if version != "0.3.7":
            raise ImportError(f"found scikit-dimension=={version}")
        from skdim.id import TLE
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise ImportError(
            "Manifold G scoring requires scikit-dimension==0.3.7; "
            "install -r requirements/manifold.txt."
        ) from error
    return TLE


def compute_metrics(
    features: np.ndarray,
    *,
    k: int = 10,
    epsilon: float = 1e-4,
    alpha: float = 0.1,
    compute_dimension: bool = True,
) -> ManifoldMetrics:
    """Compute independent D, V, G and D/V states from an unscaled [N,d] array.

    Args:
        features: One action point per row; converted to FP64 without rescaling.
        k: TLE neighbors and minimum-prefix gate for both scoring branches.
        epsilon: TLE distance tolerance.
        alpha: Nonnegative dimension penalty used only for G.
        compute_dimension: Whether to attempt TLE; V never requires it. Missing
            TLE is reported on D/G/D:V while V retains its own state. G callers
            must call require_tle at startup to reject missing dependencies.

    Returns:
        Independent metric validity, reasons, and numerical diagnostics. Shape
        and parameter errors raise ValueError; invalid numerical data does not.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError(f"Expected nonempty feature dimension [N,d], found {x.shape}")
    if (
        not isinstance(k, int)
        or k < 1
        or not math.isfinite(epsilon)
        or epsilon <= 0
        or not math.isfinite(alpha)
        or alpha < 0
    ):
        raise ValueError(
            f"Expected k>=1, epsilon>0 and alpha>=0, found {k=}, {epsilon=}, {alpha=}"
        )
    n, d = x.shape
    diagnostics: dict[str, Any] = {"warnings": []}

    def invalid(reason: str) -> ManifoldMetrics:
        metric = Metric(reason=reason)
        return ManifoldMetrics(n, k, metric, metric, metric, metric, diagnostics)

    if n <= k:
        return invalid("insufficient_points")
    if not np.isfinite(x).all():
        return invalid("nonfinite_input")
    with np.errstate(over="ignore", invalid="ignore"):
        centered = x - x.mean(axis=0)
        gram = centered @ centered.T
    if not np.isfinite(gram).all():
        return invalid("nonfinite_gram")
    try:
        eigenvalues = np.linalg.eigvalsh(gram)
    except np.linalg.LinAlgError:
        return invalid("eigendecomposition_failed")
    tolerance = 1e-12 + 1e-10 * max(float(eigenvalues[-1]), 0.0)
    diagnostics.update(
        gram_min_eigenvalue=float(eigenvalues[0]),
        gram_negative_tolerance=tolerance,
        gram_negative_count=int((eigenvalues < 0).sum()),
    )
    if eigenvalues[0] < -tolerance:
        return invalid("negative_spectrum_outside_tolerance")
    with np.errstate(over="ignore", invalid="ignore"):
        v = float(0.5 * np.log1p((d / n) * np.maximum(eigenvalues, 0)).sum())
    volume = (
        Metric(v) if math.isfinite(v) and v > 0 else Metric(reason="invalid_volume")
    )
    diagnostics["computed_volume"] = v if math.isfinite(v) else None
    dimension = Metric(reason="dimension_disabled")
    if compute_dimension:
        try:
            estimator_type = require_tle()
        except ImportError as error:
            dimension = Metric(reason="tle_dependency_unavailable")
            diagnostics["tle_dependency_error"] = str(error)
        else:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    estimator = estimator_type(epsilon=epsilon).fit(
                        x, n_neighbors=k, n_jobs=1
                    )
                diagnostics["warnings"] = sorted({str(w.message) for w in caught})
                local = np.asarray(estimator.dimension_pw_)
                diagnostics["tle_local_valid_count"] = int(np.isfinite(local).sum())
                dimension = Metric(reason="invalid_local_dimensions")
                if local.shape == (n,) and np.isfinite(local).all():
                    mean = float(np.mean(local))
                    if math.isfinite(mean) and mean > 0:
                        dimension = Metric(mean)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                dimension = Metric(reason="tle_fit_failed")
                diagnostics["tle_error"] = str(error)

    reason = volume.reason or dimension.reason
    score = ratio = Metric(reason=reason)
    if volume.valid and dimension.valid:
        g = volume.value * math.exp(-alpha * dimension.value)
        r = dimension.value / volume.value
        score = Metric(g) if math.isfinite(g) else Metric(reason="invalid_score")
        ratio = Metric(r) if math.isfinite(r) else Metric(reason="invalid_ratio")
    return ManifoldMetrics(n, k, dimension, volume, score, ratio, diagnostics)
