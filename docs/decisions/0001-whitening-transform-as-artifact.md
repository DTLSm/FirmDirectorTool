# 0001. The whitening transform is a versioned artifact, not a per-run computation

**Date:** 2026-09-15
**Status:** Accepted

## Context

The BoardEx research tool estimated the covariance matrix from whatever data was
in the current run, inverted it with `np.linalg.inv`, and only then filtered
near-zero eigenvalues. Two problems:

1. `inv` on a rank-deficient covariance (collinear dummies, a constant column)
   fails or returns garbage before the filter can act. With 17 mostly-binary
   features this is a live risk.
2. Re-estimating the metric on every run means distances from two runs are not
   comparable — the yardstick moved. For a tool that recomputes distances after
   every daily ingestion, that is unacceptable.

## Options considered

1. **Keep per-run estimation, fix the inversion.** Solves (1), not (2).
2. **Fit once, persist, apply consistently.** Solves both. Costs: a fitting step
   in the pipeline, a storage location, and a versioning discipline.
3. **Fix the covariance per board-year.** Statistically odd for ~10-row boards;
   rejected.

## Decision

Option 2. `WhiteningTransform.fit()` eigendecomposes the covariance directly and
builds `L` from the pseudo-inverse, dropping eigen-directions below a relative
tolerance. The fitted transform is saved as an `.npz` with metadata (feature
names, rank, fit date, version label) and loaded by the engine at run time.

## Consequences

- Distances carry a transform version and are only comparable within it.
- Drift in the underlying covariance becomes an explicit monitoring signal
  (Slice 6): refit on a recent window, compare to the deployed transform.
- The transform will be registered in MLflow alongside the prediction model
  (Slice 5), since the model's "hypothetical distance" feature depends on it.
- Feature order is enforced: the engine refuses a transform whose feature names
  do not match the config in order.
