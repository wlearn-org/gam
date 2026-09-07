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

## Relaxed paths

Set `relax: 1` in the params dictionary to refit each active set without a penalty.
Ordinary `predict()` and `get_coefs()` keep their penalized-path behavior. Use
`has_relaxed`, `predict_relaxed(X, fit_idx=None)`, and
`get_relaxed_coefs(fit_idx=None)` for the relaxed path. Both accessors default to
the selected CV fit (or the last fit when CV is absent). Missing relaxed state
raises an error. Grouped fitting rejects `relax` because that ABI does not carry it.

Ordinary models keep GAM1 / `wlearn.gam.{classifier,regressor}@1` artifacts.
Relaxed models use GAM2 / `@2` and persist both paths, including coefficients and
diagnostics. Current loaders accept both formats; older loaders cannot load `@2`.

## Classifier prediction contract

For binomial and multinomial models, `predict()` returns int32 class labels.
`predictProba()` (Python: `predict_proba()`) returns a flat row-major matrix
with one column per entry in `classes`, including both columns for binary
classification. Arbitrary int32 labels are encoded for fitting and retained in
WLRN metadata. Existing artifacts without class metadata use ordinal labels.

`task: 'classification'` chooses binomial or multinomial from the fitted labels
when no family is specified. Explicit families remain authoritative. The former
scalar response is available as `predictResponse()` / `predict_response()`;
relaxed response accessors retain their numerical meaning. This corrects the
unreleased estimator contract for Pipeline, AutoML scoring, and ensembles.
