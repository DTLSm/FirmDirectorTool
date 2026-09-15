"""Within-board Mahalanobis distance engine."""

from .engine import (
    EngineConfig,
    focal_distances,
    group_summary,
    pairwise_distances,
    prepare,
    run_focal,
    run_group,
)
from .whitening import WhiteningTransform

__all__ = [
    "EngineConfig",
    "WhiteningTransform",
    "focal_distances",
    "group_summary",
    "pairwise_distances",
    "prepare",
    "run_focal",
    "run_group",
]
