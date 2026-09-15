"""Within-board distance engine.

Ported from the BoardEx research tool with three deliberate changes:

* DataFrame in, DataFrame out. No file I/O here.
* Mahalanobis only, via a fitted :class:`WhiteningTransform`. The metric is an
  input to the engine, not something it estimates on the fly.
* Ambiguous focal directors raise instead of being silently resolved.

Terminology: a *board-period* is one issuer at one snapshot date; a *pair* is
two distinct persons on the same board-period.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl

from .whitening import WhiteningTransform

Imputation = Literal["drop", "median", "mean", "zero"]


@dataclass(frozen=True)
class EngineConfig:
    """Column names and preprocessing choices.

    Defaults match the EDGAR-sourced schema. The private BoardEx tool can pass
    its own names (``BoardID``, ``DirectorID``, ``AnnualReportDate``).
    """

    features: tuple[str, ...]
    board_col: str = "issuer_cik"
    person_col: str = "person_cik"
    period_col: str = "snapshot_date"
    imputation: Imputation = "drop"
    #: Columns carried through unchanged into the output alongside the ids.
    passthrough: tuple[str, ...] = field(default_factory=tuple)

    @property
    def id_cols(self) -> list[str]:
        return [self.board_col, self.period_col, self.person_col]

    @property
    def group_cols(self) -> list[str]:
        return [self.board_col, self.period_col]


# --------------------------------------------------------------- preparation


def prepare(df: pl.DataFrame, cfg: EngineConfig) -> pl.DataFrame:
    """Validate columns, apply imputation, cast features to float64."""
    missing = [c for c in cfg.id_cols + list(cfg.features) if c not in df.columns]
    if missing:
        raise ValueError(f"input is missing columns: {missing}")

    feats = list(cfg.features)
    if cfg.imputation == "drop":
        df = df.drop_nulls(subset=feats)
    elif cfg.imputation == "median":
        df = df.with_columns([pl.col(c).fill_null(pl.col(c).median()) for c in feats])
    elif cfg.imputation == "mean":
        df = df.with_columns([pl.col(c).fill_null(pl.col(c).mean()) for c in feats])
    elif cfg.imputation == "zero":
        df = df.with_columns([pl.col(c).fill_null(0) for c in feats])
    else:  # pragma: no cover - Literal guards this at type-check time
        raise ValueError(f"unknown imputation: {cfg.imputation!r}")

    return df.with_columns([pl.col(c).cast(pl.Float64) for c in feats])


def feature_matrix(df: pl.DataFrame, cfg: EngineConfig) -> npt.NDArray[np.float64]:
    """Feature columns in config order as a float64 ndarray."""
    return df.select(list(cfg.features)).to_numpy().astype(np.float64)


# ----------------------------------------------------------------- pairwise


def _whitened_frame(
    df: pl.DataFrame, cfg: EngineConfig, wt: WhiteningTransform
) -> tuple[pl.DataFrame, list[str]]:
    if tuple(wt.feature_names) != tuple(cfg.features):
        raise ValueError(
            "feature order mismatch between config and whitening transform: "
            f"{cfg.features} vs {wt.feature_names}"
        )
    Z = wt.transform(feature_matrix(df, cfg))
    z_names = [f"_w{i}" for i in range(Z.shape[1])]
    keep = cfg.id_cols + list(cfg.passthrough)
    out = df.select(keep).with_columns(
        [pl.Series(name=n, values=Z[:, i]) for i, n in enumerate(z_names)]
    )
    return out, z_names


def pairwise_distances(df: pl.DataFrame, cfg: EngineConfig, wt: WhiteningTransform) -> pl.DataFrame:
    """All ordered pairs ``(i, j)``, ``i != j``, within each board-period.

    Output columns: ``board_col, period_col, person_i, person_j, distance``,
    plus ``<passthrough>_j`` for every passthrough column.
    """
    work, z_names = _whitened_frame(df, cfg, wt)
    pairs = work.join(work, on=cfg.group_cols, suffix="_j").filter(
        pl.col(cfg.person_col) != pl.col(f"{cfg.person_col}_j")
    )

    if z_names:
        sq = pl.sum_horizontal([(pl.col(z) - pl.col(f"{z}_j")) ** 2 for z in z_names])
        dist = sq.sqrt()
    else:
        dist = pl.lit(0.0)

    select: list[pl.Expr | str] = [
        *cfg.group_cols,
        pl.col(cfg.person_col).alias("person_i"),
        pl.col(f"{cfg.person_col}_j").alias("person_j"),
        dist.alias("distance"),
    ]
    select.extend(pl.col(f"{c}_j") for c in cfg.passthrough)
    return pairs.select(select)


# ------------------------------------------------------------- aggregations


def group_summary(pairwise: pl.DataFrame, cfg: EngineConfig) -> pl.DataFrame:
    """Per person: distance statistics to every other person on the board-period."""
    return (
        pairwise.group_by([*cfg.group_cols, "person_i"])
        .agg(
            pl.col("distance").mean().alias("dist_mean"),
            pl.col("distance").median().alias("dist_median"),
            pl.col("distance").std().alias("dist_std"),
            pl.col("distance").min().alias("dist_min"),
            pl.col("distance").max().alias("dist_max"),
            pl.col("distance").count().alias("n_others"),
        )
        .rename({"person_i": cfg.person_col})
    )


def focal_distances(
    df: pl.DataFrame, pairwise: pl.DataFrame, cfg: EngineConfig, focal_col: str
) -> pl.DataFrame:
    """Per person: distance to the single focal person on their board-period.

    The focal person's own distance is 0. Board-periods with no focal person
    yield null. Board-periods with more than one focal person raise — pick a
    rule upstream rather than let the engine guess.
    """
    if focal_col not in df.columns:
        raise ValueError(f"focal column {focal_col!r} not in input")
    if focal_col not in cfg.passthrough:
        raise ValueError(f"focal column {focal_col!r} must be listed in cfg.passthrough")

    counts = (
        df.filter(pl.col(focal_col) == 1)
        .group_by(cfg.group_cols)
        .len(name="n_focal")
        .filter(pl.col("n_focal") > 1)
    )
    if counts.height:
        raise ValueError(
            f"{counts.height} board-period(s) have more than one focal person:\n{counts}"
        )

    to_focal = (
        pairwise.filter(pl.col(f"{focal_col}_j") == 1)
        .select([*cfg.group_cols, "person_i", pl.col("distance").alias("dist_focal")])
        .rename({"person_i": cfg.person_col})
    )
    focal_self = df.filter(pl.col(focal_col) == 1).select(
        [*cfg.id_cols, pl.lit(0.0).alias("dist_focal")]
    )
    return pl.concat([to_focal, focal_self], how="vertical_relaxed")


# ------------------------------------------------------------- convenience


def run_group(df: pl.DataFrame, cfg: EngineConfig, wt: WhiteningTransform) -> pl.DataFrame:
    """Prepare → pairwise → group summary, joined back onto the input ids and features."""
    df = prepare(df, cfg)
    pw = pairwise_distances(df, cfg, wt)
    summary = group_summary(pw, cfg)
    base = df.select(cfg.id_cols + list(cfg.passthrough) + list(cfg.features))
    return base.join(summary, on=cfg.id_cols, how="left")


def run_focal(
    df: pl.DataFrame, cfg: EngineConfig, wt: WhiteningTransform, focal_col: str
) -> pl.DataFrame:
    """Prepare → pairwise → focal distances, joined back onto the input ids and features."""
    df = prepare(df, cfg)
    pw = pairwise_distances(df, cfg, wt)
    fd = focal_distances(df, pw, cfg, focal_col)
    base = df.select(cfg.id_cols + list(cfg.passthrough) + list(cfg.features))
    return base.join(fd, on=cfg.id_cols, how="left")
