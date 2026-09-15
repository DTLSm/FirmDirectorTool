import numpy as np
import polars as pl
import pytest

from firmdirectortool.distance import (
    EngineConfig,
    WhiteningTransform,
    pairwise_distances,
    run_focal,
    run_group,
)


def identity_transform(names: tuple[str, ...]) -> WhiteningTransform:
    """A whitening transform that is the identity, so distances are plain Euclidean.

    Lets the aggregation logic be tested by hand, independently of the covariance
    estimation that test_whitening.py covers.
    """
    p = len(names)
    return WhiteningTransform(
        feature_names=names,
        mean=np.zeros(p),
        matrix=np.eye(p),
        n_fitted=0,
        rank=p,
        rel_tol=0.0,
        fitted_at="test",
        version="identity",
    )


FEATS = ("x", "y")
CFG = EngineConfig(features=FEATS, passthrough=("is_chair",))
ID = identity_transform(FEATS)


@pytest.fixture
def three_person_board() -> pl.DataFrame:
    # Points at (0,0), (3,0), (0,4): a 3-4-5 triangle, so distances are 3, 4, 5.
    return pl.DataFrame(
        {
            "issuer_cik": ["A", "A", "A"],
            "snapshot_date": ["2025-12-31"] * 3,
            "person_cik": ["p1", "p2", "p3"],
            "x": [0.0, 3.0, 0.0],
            "y": [0.0, 0.0, 4.0],
            "is_chair": [1, 0, 0],
        }
    )


def test_pairwise_hand_computed(three_person_board: pl.DataFrame) -> None:
    pw = pairwise_distances(three_person_board, CFG, ID)
    assert pw.height == 6  # 3 people, ordered pairs, no self-pairs

    d = {(r["person_i"], r["person_j"]): r["distance"] for r in pw.iter_rows(named=True)}
    assert d[("p1", "p2")] == pytest.approx(3.0)
    assert d[("p1", "p3")] == pytest.approx(4.0)
    assert d[("p2", "p3")] == pytest.approx(5.0)
    # symmetric
    assert d[("p2", "p1")] == pytest.approx(3.0)
    assert d[("p3", "p2")] == pytest.approx(5.0)


def test_group_summary_hand_computed(three_person_board: pl.DataFrame) -> None:
    out = run_group(three_person_board, CFG, ID).sort("person_cik")
    rows = {r["person_cik"]: r for r in out.iter_rows(named=True)}

    # p1 sees 3 and 4 → mean 3.5, min 3, max 4
    assert rows["p1"]["dist_mean"] == pytest.approx(3.5)
    assert rows["p1"]["dist_min"] == pytest.approx(3.0)
    assert rows["p1"]["dist_max"] == pytest.approx(4.0)
    assert rows["p1"]["n_others"] == 2

    # p2 sees 3 and 5 → mean 4
    assert rows["p2"]["dist_mean"] == pytest.approx(4.0)
    # p3 sees 4 and 5 → mean 4.5
    assert rows["p3"]["dist_mean"] == pytest.approx(4.5)

    # ids, passthrough and features come back
    assert set(out.columns) >= {"issuer_cik", "snapshot_date", "person_cik", "is_chair", "x", "y"}


def test_boards_do_not_mix(three_person_board: pl.DataFrame) -> None:
    other = three_person_board.with_columns(pl.lit("B").alias("issuer_cik"))
    both = pl.concat([three_person_board, other])
    pw = pairwise_distances(both, CFG, ID)
    assert pw.height == 12  # 6 per board, none across
    assert set(pw["issuer_cik"].unique()) == {"A", "B"}


def test_periods_do_not_mix(three_person_board: pl.DataFrame) -> None:
    later = three_person_board.with_columns(pl.lit("2026-12-31").alias("snapshot_date"))
    both = pl.concat([three_person_board, later])
    pw = pairwise_distances(both, CFG, ID)
    assert pw.height == 12


def test_focal_distances(three_person_board: pl.DataFrame) -> None:
    out = run_focal(three_person_board, CFG, ID, focal_col="is_chair").sort("person_cik")
    rows = {r["person_cik"]: r["dist_focal"] for r in out.iter_rows(named=True)}
    assert rows["p1"] == pytest.approx(0.0)  # the chair
    assert rows["p2"] == pytest.approx(3.0)
    assert rows["p3"] == pytest.approx(4.0)


def test_focal_raises_when_ambiguous(three_person_board: pl.DataFrame) -> None:
    two_chairs = three_person_board.with_columns(pl.Series("is_chair", [1, 1, 0]))
    with pytest.raises(ValueError, match="more than one focal"):
        run_focal(two_chairs, CFG, ID, focal_col="is_chair")


def test_focal_null_when_no_focal_on_board(three_person_board: pl.DataFrame) -> None:
    no_chair = three_person_board.with_columns(pl.Series("is_chair", [0, 0, 0]))
    out = run_focal(no_chair, CFG, ID, focal_col="is_chair")
    assert out["dist_focal"].null_count() == 3


def test_focal_requires_passthrough(three_person_board: pl.DataFrame) -> None:
    cfg = EngineConfig(features=FEATS)  # is_chair not in passthrough
    with pytest.raises(ValueError, match="passthrough"):
        run_focal(three_person_board, cfg, ID, focal_col="is_chair")


def test_imputation_drop_removes_rows(three_person_board: pl.DataFrame) -> None:
    with_null = three_person_board.with_columns(pl.Series("x", [0.0, None, 0.0]))
    out = run_group(with_null, CFG, ID)
    assert out.height == 2
    assert set(out["person_cik"]) == {"p1", "p3"}


def test_imputation_median(three_person_board: pl.DataFrame) -> None:
    cfg = EngineConfig(features=FEATS, passthrough=("is_chair",), imputation="median")
    with_null = three_person_board.with_columns(pl.Series("x", [0.0, None, 0.0]))
    out = run_group(with_null, cfg, ID)
    assert out.height == 3
    assert out.filter(pl.col("person_cik") == "p2")["x"][0] == pytest.approx(0.0)


def test_feature_order_mismatch_raises(three_person_board: pl.DataFrame) -> None:
    wrong = identity_transform(("y", "x"))
    with pytest.raises(ValueError, match="feature order mismatch"):
        run_group(three_person_board, CFG, wrong)


def test_missing_column_raises(three_person_board: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        run_group(three_person_board.drop("y"), CFG, ID)


def test_end_to_end_with_fitted_transform() -> None:
    """Fit on a population, then run the engine — the realistic path."""
    rng = np.random.default_rng(0)
    n_boards, size = 40, 9
    n = n_boards * size
    df = pl.DataFrame(
        {
            "issuer_cik": [f"B{k // size:03d}" for k in range(n)],
            "snapshot_date": ["2025-12-31"] * n,
            "person_cik": [f"P{k:04d}" for k in range(n)],
            "tenure": rng.normal(8, 4, n).clip(0),
            "board_count": rng.poisson(2, n).astype(float),
            "is_independent": rng.binomial(1, 0.7, n).astype(float),
            "has_ceo_exp": rng.binomial(1, 0.3, n).astype(float),
        }
    )
    feats = ("tenure", "board_count", "is_independent", "has_ceo_exp")
    cfg = EngineConfig(features=feats)
    wt = WhiteningTransform.fit(df.select(list(feats)).to_numpy(), feats, version="v1")

    out = run_group(df, cfg, wt)
    assert out.height == n
    assert out["n_others"].unique().to_list() == [size - 1]
    assert out["dist_mean"].null_count() == 0
    assert (out["dist_mean"] > 0).all()
