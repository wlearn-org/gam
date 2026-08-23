# wlearn-gam Python

Python estimator wrapper for the wlearn C11 GLM/GAM core.

## Install

```bash
pip install wlearn-gam
```

## Example

```python
from wlearn_gam import GAMModel

model = GAMModel({
    "family": "gaussian",
    "penalty": "elasticnet",
    "alpha": 0.5,
    "seed": 42,
})
model.fit(X, y)
pred = model.predict(X_test)
score = model.score(X_test, y_test)

model.save("gam.wlrn")
restored = GAMModel.load("gam.wlrn")
```

## API

- `GAMModel(params=None)` or `GAMModel.create(params)`.
- `fit(X, y)` trains the regularization path.
- `predict(X, fit_idx=None)`, `predict_eta(...)`, `predict_proba(...)`.
- `score(X, y, fit_idx=None)` returns accuracy for classifiers, R2 otherwise.
- `get_coefs`, `get_lambda`, `get_deviance`, `get_df`, `get_cv_mean`,
  `get_cv_se` inspect fitted path entries.
- `save(path=None)` returns WLRN bytes and writes them when given a `str` or `Path`.
- `GAMModel.load(bytes_or_path)` accepts WLRN bytes, `str`, or `Path`.
- `get_params()` / `set_params(...)` support estimator cloning/search.
- `default_search_space()` returns the AutoML search-space IR.
- `dispose()` releases native memory early in long-running processes; context-manager use is supported.

Properties: `is_fitted`, `n_features`, `n_fits`, `idx_min`, `idx_1se`,
`capabilities`.

Common params: `family`, `penalty`, `alpha`, `nLambda`, `lambdaMinRatio`,
`nFolds`, `standardize`, `seed`. Families include `gaussian`, `binomial`,
`poisson`, `gamma`, `multinomial`, `cox`, `huber`, and `quantile`.

`save()` returns a WLRN bundle. Native GAM bytes are an internal artifact inside
the bundle, matching JavaScript `@wlearn/gam`.

## Development

The canonical native source is repository root `src/`; `py/csrc/` is generated
for Python builds. `make test-py` uses fixtures and has no sklearn/scipy/
statsmodels dependency. Use `make test-py-ref` for optional external parity
tests.
