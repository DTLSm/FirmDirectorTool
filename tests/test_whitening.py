import numpy as np
import pytest
from scipy.spatial.distance import mahalanobis

from firmdirectortool.distance import WhiteningTransform

RNG = np.random.default_rng(12345)


def _pairwise(Z: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    return np.linalg.norm(Z[i] - Z[j], axis=1)


@pytest.fixture
def well_conditioned() -> np.ndarray:
    # Mixed-scale continuous + binary features, like the real feature vector.
    n = 500
    age = RNG.normal(60, 8, n)
    tenure = RNG.normal(8, 5, n).clip(0)
    boards = RNG.poisson(2, n).astype(float)
    dummies = RNG.binomial(1, 0.4, (n, 5)).astype(float)
    return np.column_stack([age, tenure, boards, dummies])


def test_known_answer_matches_scipy(well_conditioned: np.ndarray) -> None:
    X = well_conditioned
    names = [f"f{k}" for k in range(X.shape[1])]
    wt = WhiteningTransform.fit(X, names)
    Z = wt.transform(X)

    VI = np.linalg.pinv(np.cov(X, rowvar=False))
    idx = RNG.integers(0, len(X), (50, 2))
    for i, j in idx:
        if i == j:
            continue
        expected = mahalanobis(X[i], X[j], VI)
        got = float(np.linalg.norm(Z[i] - Z[j]))
        assert got == pytest.approx(expected, rel=1e-9)


def test_pseudo_inverse_property(well_conditioned: np.ndarray) -> None:
    X = well_conditioned
    wt = WhiteningTransform.fit(X, [f"f{k}" for k in range(X.shape[1])])
    expected = np.linalg.pinv(np.cov(X, rowvar=False))
    np.testing.assert_allclose(wt.pseudo_inverse_covariance, expected, rtol=1e-8, atol=1e-10)
    assert wt.rank == X.shape[1]


def test_duplicate_column_does_not_raise_and_preserves_distances(
    well_conditioned: np.ndarray,
) -> None:
    X = well_conditioned
    X_dup = np.column_stack([X, X[:, 0]])  # exact collinearity

    wt = WhiteningTransform.fit(X, [f"f{k}" for k in range(X.shape[1])])
    wt_dup = WhiteningTransform.fit(X_dup, [f"f{k}" for k in range(X_dup.shape[1])])

    assert wt_dup.rank == wt.rank  # the duplicate adds no rank

    i = RNG.integers(0, len(X), 200)
    j = RNG.integers(0, len(X), 200)
    d = _pairwise(wt.transform(X), i, j)
    d_dup = _pairwise(wt_dup.transform(X_dup), i, j)
    np.testing.assert_allclose(d_dup, d, rtol=1e-8)


def test_constant_column_does_not_raise_and_preserves_distances(
    well_conditioned: np.ndarray,
) -> None:
    X = well_conditioned
    X_const = np.column_stack([X, np.full(len(X), 7.0)])

    wt = WhiteningTransform.fit(X, [f"f{k}" for k in range(X.shape[1])])
    wt_c = WhiteningTransform.fit(X_const, [f"f{k}" for k in range(X_const.shape[1])])

    assert wt_c.rank == wt.rank

    i = RNG.integers(0, len(X), 200)
    j = RNG.integers(0, len(X), 200)
    np.testing.assert_allclose(
        _pairwise(wt_c.transform(X_const), i, j),
        _pairwise(wt.transform(X), i, j),
        rtol=1e-8,
    )


def test_affine_invariance(well_conditioned: np.ndarray) -> None:
    """Mahalanobis distance is invariant to invertible affine maps of the features."""
    X = well_conditioned
    p = X.shape[1]
    A = RNG.normal(size=(p, p))
    while abs(np.linalg.det(A)) < 1e-3:  # ensure invertible
        A = RNG.normal(size=(p, p))
    b = RNG.normal(size=p) * 100
    Y = X @ A + b

    names = [f"f{k}" for k in range(p)]
    wt_x = WhiteningTransform.fit(X, names)
    wt_y = WhiteningTransform.fit(Y, names)

    i = RNG.integers(0, len(X), 200)
    j = RNG.integers(0, len(X), 200)
    np.testing.assert_allclose(
        _pairwise(wt_y.transform(Y), i, j),
        _pairwise(wt_x.transform(X), i, j),
        rtol=1e-7,
    )


def test_save_load_roundtrip(tmp_path, well_conditioned: np.ndarray) -> None:  # type: ignore[no-untyped-def]
    X = well_conditioned
    wt = WhiteningTransform.fit(X, [f"f{k}" for k in range(X.shape[1])], version="v-test")
    path = wt.save(tmp_path / "wt")
    assert path.suffix == ".npz"

    loaded = WhiteningTransform.load(path)
    assert loaded.feature_names == wt.feature_names
    assert loaded.version == "v-test"
    assert loaded.rank == wt.rank
    assert loaded.n_fitted == wt.n_fitted
    np.testing.assert_array_equal(loaded.mean, wt.mean)
    np.testing.assert_array_equal(loaded.matrix, wt.matrix)
    np.testing.assert_array_equal(loaded.transform(X), wt.transform(X))


def test_rejects_single_row() -> None:
    with pytest.raises(ValueError, match="at least 2 rows"):
        WhiteningTransform.fit(np.array([[1.0, 2.0]]), ["a", "b"])


def test_rejects_all_constant() -> None:
    with pytest.raises(ValueError, match="rank 0"):
        WhiteningTransform.fit(np.ones((10, 3)), ["a", "b", "c"])


def test_rejects_wrong_feature_count_on_transform(well_conditioned: np.ndarray) -> None:
    X = well_conditioned
    wt = WhiteningTransform.fit(X, [f"f{k}" for k in range(X.shape[1])])
    with pytest.raises(ValueError, match="expected shape"):
        wt.transform(X[:, :-1])


def test_single_feature_works() -> None:
    X = RNG.normal(5, 2, (100, 1))
    wt = WhiteningTransform.fit(X, ["only"])
    assert wt.rank == 1
    Z = wt.transform(X)
    # In 1-D, Mahalanobis distance is |x_i - x_j| / sd
    sd = X.std(ddof=1)
    np.testing.assert_allclose(abs(Z[3, 0] - Z[7, 0]), abs(X[3, 0] - X[7, 0]) / sd, rtol=1e-10)
