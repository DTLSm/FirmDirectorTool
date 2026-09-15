"""Whitening transform for Mahalanobis distance.

"Whitening" is the standard term for a linear map that takes correlated
variables with arbitrary variances to variables with identity covariance —
unit variance in every direction, no correlation. The name is borrowed from
signal processing, where white noise has a flat spectrum and no structure.
Mahalanobis distance is simply Euclidean distance after whitening, which is
why the transform is the natural unit of work here: map each row once, then
plain sums of squares give the right metric for every pair.

A :class:`WhiteningTransform` is fitted once on a reference population and then
applied consistently, so that distances computed at different times remain
comparable. It is a versioned artifact in its own right: the metric is only
meaningful relative to the covariance it was fitted on, and a change in that
covariance is itself a signal worth monitoring.

Mathematically, given feature matrix ``X`` with covariance ``S``, we compute a
matrix ``L`` such that ``L @ L.T`` is the Moore-Penrose pseudo-inverse of ``S``.
Then for any two rows ``x_i`` and ``x_j``::

    ||(x_i - x_j) @ L||  ==  sqrt((x_i - x_j) @ pinv(S) @ (x_i - x_j).T)

which is the Mahalanobis distance. Using the pseudo-inverse means directions
with negligible variance (collinear dummies, constant columns) are dropped
rather than inverted, so rank-deficient feature sets do not blow up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class WhiteningTransform:
    """Fitted whitening transform. Immutable; create via :meth:`fit` or :meth:`load`."""

    feature_names: tuple[str, ...]
    mean: FloatArray  # shape (p,)
    matrix: FloatArray  # shape (p, k), k = rank
    n_fitted: int
    rank: int
    rel_tol: float
    fitted_at: str
    version: str

    # ------------------------------------------------------------------ fit

    @classmethod
    def fit(
        cls,
        X: FloatArray,
        feature_names: tuple[str, ...] | list[str],
        *,
        rel_tol: float = 1e-10,
        version: str = "unversioned",
    ) -> WhiteningTransform:
        """Fit on a reference population.

        Parameters
        ----------
        X
            Feature matrix, shape ``(n, p)``. Must have at least two rows.
        feature_names
            Names for the ``p`` columns, in order. Stored so that later callers
            can verify they are passing features in the same order.
        rel_tol
            Eigenvalues below ``rel_tol * max(eigenvalue)`` are treated as zero
            and dropped. Relative to the largest eigenvalue so the threshold is
            scale-aware.
        version
            Free-form label. Use it: distances are only comparable within a
            version.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        n, p = X.shape
        if n < 2:
            raise ValueError(f"need at least 2 rows to estimate covariance, got {n}")
        names = tuple(feature_names)
        if len(names) != p:
            raise ValueError(f"{len(names)} feature names for {p} columns")

        mean = X.mean(axis=0)
        cov = np.atleast_2d(np.cov(X, rowvar=False))
        eigvals, eigvecs = np.linalg.eigh(cov)

        # eigh returns ascending eigenvalues; numerical noise can make tiny ones
        # slightly negative. Clip before thresholding.
        eigvals = np.clip(eigvals, 0.0, None)
        largest = float(eigvals.max())
        if largest <= 0.0:
            raise ValueError("all features are constant; covariance has rank 0")

        keep = eigvals > rel_tol * largest
        matrix = eigvecs[:, keep] / np.sqrt(eigvals[keep])

        return cls(
            feature_names=names,
            mean=mean,
            matrix=matrix,
            n_fitted=n,
            rank=int(keep.sum()),
            rel_tol=rel_tol,
            fitted_at=datetime.now(UTC).isoformat(timespec="seconds"),
            version=version,
        )

    # ------------------------------------------------------------ transform

    def transform(self, X: FloatArray) -> FloatArray:
        """Map raw features to whitened coordinates, shape ``(n, rank)``.

        Euclidean distance between rows of the result equals Mahalanobis
        distance between the corresponding rows of ``X``.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError(f"expected shape (n, {len(self.feature_names)}), got {X.shape}")
        return (X - self.mean) @ self.matrix

    @property
    def pseudo_inverse_covariance(self) -> FloatArray:
        """``L @ L.T`` — the pseudo-inverse of the fitted covariance. For tests and inspection."""
        return self.matrix @ self.matrix.T

    # ------------------------------------------------------------ persist

    def save(self, path: str | Path) -> Path:
        """Persist to a single ``.npz`` file. Returns the path written."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        meta = {
            "feature_names": list(self.feature_names),
            "n_fitted": self.n_fitted,
            "rank": self.rank,
            "rel_tol": self.rel_tol,
            "fitted_at": self.fitted_at,
            "version": self.version,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, matrix=self.matrix, meta=np.array(json.dumps(meta)))
        return path

    @classmethod
    def load(cls, path: str | Path) -> WhiteningTransform:
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            return cls(
                feature_names=tuple(meta["feature_names"]),
                mean=np.asarray(data["mean"], dtype=np.float64),
                matrix=np.asarray(data["matrix"], dtype=np.float64),
                n_fitted=int(meta["n_fitted"]),
                rank=int(meta["rank"]),
                rel_tol=float(meta["rel_tol"]),
                fitted_at=str(meta["fitted_at"]),
                version=str(meta["version"]),
            )
