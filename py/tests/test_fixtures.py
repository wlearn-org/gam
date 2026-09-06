import json
from pathlib import Path

import wlearn_gam
from wlearn.bundle import decode_bundle
from wlearn_gam import GAMModel


def _check_close(actual, expected, tol, label):
    if len(actual) != len(expected):
        raise AssertionError(f"{label}: length {len(actual)} != {len(expected)}")
    for i, (a, e) in enumerate(zip(actual, expected)):
        if abs(a - e) > tol:
            raise AssertionError(f"{label}[{i}]: got {a}, expected {e}, tol={tol}")


def test_gaussian_linear_fixture(tmp_path):
    assert not hasattr(wlearn_gam, "get_lib")

    fixture_path = Path(__file__).with_name("fixtures") / "gaussian_linear.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    X = fixture["X"]
    y = fixture["y"]
    pred_X = fixture["predict_X"]

    model = GAMModel({
        "family": "gaussian",
        "link": "identity",
        "penalty": "none",
        "nLambda": 1,
        "standardize": 0,
        "fitIntercept": 1,
        "tol": 1e-10,
        "maxIter": 10000,
        "screening": 0,
        "seed": 42,
    }).fit(X, y)

    try:
        assert model.is_fitted
        assert model.n_fits >= 1
        _check_close(model.get_coefs(0), fixture["expected_coef"], 1e-6, "coef")
        _check_close(model.predict(pred_X, 0), fixture["expected_pred"], 1e-6, "prediction")

        bundle = model.save()
        manifest, _, _ = decode_bundle(bundle)
        assert manifest["typeId"] == "wlearn.gam.regressor@1"

        loaded = GAMModel.load(bundle)
        try:
            _check_close(loaded.predict(pred_X, 0), fixture["expected_pred"], 1e-6, "loaded prediction")
        finally:
            loaded.dispose()

        path = tmp_path / "gam.wlrn"
        path_bundle = model.save(path)
        assert path.read_bytes() == path_bundle

        loaded_path = GAMModel.load(path)
        try:
            _check_close(loaded_path.predict(pred_X, 0), fixture["expected_pred"], 1e-6, "path loaded prediction")
        finally:
            loaded_path.dispose()

        loaded_str_path = GAMModel.load(str(path))
        try:
            _check_close(
                loaded_str_path.predict(pred_X, 0),
                fixture["expected_pred"],
                1e-6,
                "string path loaded prediction",
            )
        finally:
            loaded_str_path.dispose()
    finally:
        model.dispose()


def test_relaxed_roundtrip_and_missing_state():
    import numpy as np
    import pytest
    from wlearn.registry import load

    X = (np.arange(80) - 39.5).reshape(-1, 1) / 10
    y = 1.25 + 3 * X[:, 0]
    model = GAMModel({'family': 'gaussian', 'penalty': 'lasso', 'relax': 1,
                      'nLambda': 3, 'lambdaMinRatio': 0.3}).fit(X, y)
    try:
        assert model.has_relaxed
        assert abs(model.get_coefs(2)[1] - 3) > 0.1
        np.testing.assert_allclose(model.get_relaxed_coefs(2), [1.25, 3], atol=1e-5)
        np.testing.assert_allclose(model.predict_relaxed(X, 2), y, atol=1e-5)
        bundle = model.save()
        assert decode_bundle(bundle)[0]['typeId'] == 'wlearn.gam.regressor@2'
        loaded = load(bundle)
        try:
            np.testing.assert_array_equal(loaded.predict(X, 2), model.predict(X, 2))
            np.testing.assert_array_equal(loaded.predict_relaxed(X, 2), model.predict_relaxed(X, 2))
            assert loaded.save() == bundle
        finally:
            loaded.dispose()
    finally:
        model.dispose()
    ordinary = GAMModel({'nLambda': 1}).fit(X, y)
    try:
        assert not ordinary.has_relaxed
        assert decode_bundle(ordinary.save())[0]['typeId'] == 'wlearn.gam.regressor@1'
        with pytest.raises(RuntimeError, match='relaxed'):
            ordinary.predict_relaxed(X)
        with pytest.raises(RuntimeError, match='relaxed'):
            ordinary.get_relaxed_coefs()
    finally:
        ordinary.dispose()


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_gaussian_linear_fixture(Path(td))
        test_relaxed_roundtrip_and_missing_state()
    print("wlearn_gam fixture tests passed")
