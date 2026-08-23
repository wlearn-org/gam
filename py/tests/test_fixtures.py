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


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_gaussian_linear_fixture(Path(td))
    print("wlearn_gam fixture tests passed")
