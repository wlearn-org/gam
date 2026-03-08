# @wlearn/gam

Generalized linear models (GLM), generalized additive models (GAM), and penalized regression, written in C11 from scratch. No upstream dependencies.

Runs as native C, WASM (browser + Node), with JS and Python wrappers.

## Features

### Families

- Gaussian (identity, log)
- Binomial (logit, probit, cloglog)
- Poisson (log)
- Gamma (log, inverse)
- Inverse Gaussian (inverse squared)
- Negative Binomial (log, with dispersion)
- Tweedie (log, configurable power)
- Multinomial (softmax, K-class)
- Cox proportional hazards (partial likelihood)
- Huber (robust regression)
- Quantile (check loss)
- GAMLSS distributional regression (Normal, Gamma, Beta)

### Penalties

- Lasso (L1)
- Ridge (L2)
- Elastic net (L1 + L2)
- MCP (minimax concave penalty)
- SCAD (smoothly clipped absolute deviation)
- SLOPE (sorted L1 with FDR control)
- Group lasso (block coordinate descent)
- Sparse group lasso (group + within-group L1)
- Fused lasso / trend filtering (ADMM)
- Multi-task lasso (shared support across responses)

### GAM Smooths

- B-spline basis functions (de Boor recursion)
- Quantile-spaced knots
- Smoothness penalties (Gauss-Legendre quadrature)
- Tensor product smooths (row-wise Kronecker product of marginal bases)
- Multiple penalties per tensor (one per marginal direction)

### Optimization

- Pathwise coordinate descent with warm starts
- Proximal Newton (IRLS outer + CD inner) for GLM families
- Anderson acceleration (2-10x speedup)
- GAP Safe screening rules
- Strong screening rules
- K-fold cross-validation (lambda.min / lambda.1se)
- Relaxed fits (Meinshausen 2007)
- Per-feature penalty factors

### Other

- Deterministic (seed-controlled)
- Binary serialization (GAM1 format, save/load)
- 33 C tests, 50 JS tests, 31 Python tests

## Installation

### JavaScript (npm)

```
npm install @wlearn/gam
```

### Python

```bash
# Build the C library
mkdir build && cd build && cmake .. && make
```

### C

```bash
mkdir build && cd build
cmake .. -DBUILD_TESTING=ON
make
./test_gam  # run tests
```

## Quick Start

### JavaScript

```js
const { GAMModel } = require('@wlearn/gam')

// Linear regression with elastic net
const model = await GAMModel.create({
  family: 'gaussian',
  penalty: 'elasticnet',
  alpha: 0.5,
  nLambda: 100
})
model.fit(X_train, y_train)
const preds = model.predict(X_test)
const r2 = model.score(X_test, y_test)

// Save / load
const bundle = model.save()
const loaded = await GAMModel.load(bundle)

model.dispose()
```

### Logistic Regression

```js
const clf = await GAMModel.create({
  family: 'binomial',
  penalty: 'lasso'
})
clf.fit(X, y)
const proba = clf.predictProba(X_test)
const accuracy = clf.score(X_test, y_test)
clf.dispose()
```

### Multinomial (Multi-class)

```js
const model = await GAMModel.create({
  family: 'multinomial',
  penalty: 'elasticnet',
  alpha: 0.5
})
model.fitMultinomial(X, y, nClasses)

// Probabilities: Float64Array of length n * nClasses (row-major)
const probs = model.predictMultinomial(X_test)

// Accuracy via score()
const acc = model.score(X_test, y_test)

model.dispose()
```

### Cox Proportional Hazards

```js
const cox = await GAMModel.create({
  penalty: 'elasticnet',
  alpha: 0.5
})
cox.fitCox(X, time, status)
const hazard = cox.predictEta(X_test)
cox.dispose()
```

### Multi-task Lasso

```js
const model = await GAMModel.create({
  penalty: 'lasso',
  alpha: 1.0
})
// Y is a flat Float64Array of n * nTasks values (row-major)
model.fitMulti(X, Y, nTasks)
const preds = model.predictMulti(X_test)
model.dispose()
```

### Robust Regression (Huber / Quantile)

```js
// Huber loss -- robust to outliers
const huber = await GAMModel.create({
  family: 'huber',
  penalty: 'elasticnet',
  huberGamma: 1.345
})
huber.fit(X, y)

// Quantile regression -- model conditional quantiles
const q90 = await GAMModel.create({
  family: 'quantile',
  penalty: 'elasticnet',
  quantileTau: 0.9
})
q90.fit(X, y)
```

### GAMLSS (Distributional Regression)

Models both location (mu) and scale (sigma) as functions of covariates using the RS algorithm (Rigby and Stasinopoulos 2005). Supports Normal, Gamma, and Beta distributions.

```js
const model = await GAMModel.create({
  penalty: 'elasticnet',
  alpha: 0.5
})
model.fitGamlss(X, y, 'normal')  // or 'gamma', 'beta'

// Returns [mu_0, sigma_0, mu_1, sigma_1, ...] interleaved
const preds = model.predictGamlss(X_test)
for (let i = 0; i < n; i++) {
  console.log(`mu=${preds[i*2]}, sigma=${preds[i*2+1]}`)
}

model.dispose()
```

### Python (via ctypes)

```python
import ctypes
import numpy as np

lib = ctypes.CDLL('build/libgam.so')

# Set up function signatures (see test/test_python.py for full example)
# ...

# Fit Gaussian lasso
X = np.random.randn(100, 5)
y = X @ [1, -2, 0, 0, 3] + np.random.randn(100) * 0.1

path = lib.wl_gam_fit(
    X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 100, 5,
    y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    0, -1, 1,  # family=gaussian, link=canonical, penalty=lasso
    1.0, 100, 0.0,  # alpha, n_lambda, lambda_min_ratio
    3.0, 3.7,  # gamma_mcp, gamma_scad
    1e-7, 10000, 25, 1,  # tol, max_iter, max_inner, screening
    0, 1, 1, 0,  # n_folds, standardize, fit_intercept, relax
    1.5, 0.0,  # tweedie_power, neg_binom_theta
    0.1, 42,   # slope_q, seed
    1.345, 0.5  # huber_gamma, quantile_tau
)
```

### C

```c
#include "gam.h"

gam_params_t params;
gam_params_init(&params);
params.family = GAM_FAMILY_GAUSSIAN;
params.penalty = GAM_PENALTY_LASSO;
params.alpha = 1.0;
params.n_lambda = 100;

gam_path_t *path = gam_fit(X, nrow, ncol, y, &params);

// Predict at best CV lambda
int idx = path->idx_min >= 0 ? path->idx_min : path->n_fits - 1;
double *preds = malloc(nrow * sizeof(double));
gam_predict(path, idx, X, nrow, ncol, preds);

// Inspect path
for (int k = 0; k < path->n_fits; k++) {
    printf("lambda=%.4f df=%d deviance=%.4f\n",
           path->fits[k].lambda, path->fits[k].df,
           path->fits[k].deviance);
}

// Save / load
char *buf; int32_t len;
gam_save(path, &buf, &len);
gam_path_t *loaded = gam_load(buf, len);
gam_free_buffer(buf);

gam_free(path);
free(preds);
```

## API Reference

### Parameters

| Parameter | JS name | C field | Default | Description |
|-----------|---------|---------|---------|-------------|
| Family | `family` | `family` | `'gaussian'` (0) | Distribution family |
| Link | `link` | `link` | `'canonical'` (-1) | Link function (auto-selected if canonical) |
| Penalty | `penalty` | `penalty` | `'elasticnet'` (3) | Regularization type |
| Alpha | `alpha` | `alpha` | 1.0 | L1/L2 mix (1=lasso, 0=ridge) |
| N lambda | `nLambda` | `n_lambda` | 100 | Number of lambda values in path |
| Lambda min ratio | `lambdaMinRatio` | `lambda_min_ratio` | 0.0 (auto) | Ratio of smallest to largest lambda |
| Tolerance | `tol` | `tol` | 1e-7 | Convergence tolerance |
| Max iterations | `maxIter` | `max_iter` | 10000 | Max outer iterations |
| Max inner | `maxInner` | `max_inner` | 25 | Max CD passes per IRLS step |
| Screening | `screening` | `screening` | 1 | Enable strong screening rules |
| N folds | `nFolds` | `n_folds` | 0 | K-fold CV (0 = no CV) |
| Standardize | `standardize` | `standardize` | 1 | Standardize features |
| Fit intercept | `fitIntercept` | `fit_intercept` | 1 | Include intercept |
| Relax | `relax` | `relax` | 0 | Relaxed (unpenalized) refit on active set |
| Seed | `seed` | `seed` | 42 | Random seed |
| MCP gamma | `gammaMcp` | `gamma_mcp` | 3.0 | MCP penalty parameter |
| SCAD gamma | `gammaScad` | `gamma_scad` | 3.7 | SCAD penalty parameter |
| Tweedie power | `tweedieP` | `tweedie_power` | 1.5 | Tweedie variance power |
| NB theta | `nbTheta` | `neg_binom_theta` | 0.0 (auto) | Negative binomial dispersion |
| SLOPE q | `slopeQ` | `slope_q` | 0.1 | SLOPE FDR target |
| Huber gamma | `huberGamma` | `huber_gamma` | 1.345 | Huber loss threshold |
| Quantile tau | `quantileTau` | `quantile_tau` | 0.5 | Quantile regression level |

### Methods

#### JavaScript (`GAMModel`)

```js
// Construction
const model = await GAMModel.create(params)

// Training
model.fit(X, y)                      // Standard GLM/GAM fit
model.fitCox(X, time, status)        // Cox proportional hazards
model.fitMulti(X, Y, nTasks)         // Multi-task (multi-response)
model.fitMultinomial(X, y, nClasses) // Multinomial logistic
model.fitGamlss(X, y, distribution)  // GAMLSS distributional

// Prediction
model.predict(X, fitIdx?)            // Response predictions
model.predictEta(X, fitIdx?)         // Linear predictor
model.predictProba(X, fitIdx?)       // Probabilities (binomial/multinomial)
model.predictMulti(X, fitIdx?)       // Multi-task predictions
model.predictMultinomial(X, fitIdx?) // Multinomial probabilities
model.predictGamlss(X, fitIdx?)      // GAMLSS mu + sigma predictions
model.score(X, y, fitIdx?)           // R2 (regression), accuracy (classification)

// Path inspection
model.nFits                          // Number of lambda values fitted
model.nFeatures                      // Number of input features
model.nTasks                         // Number of tasks/classes (multi-task/multinomial)
model.familyGamlss                   // GAMLSS distribution (0=none)
model.idxMin                         // Index of lambda.min (best CV)
model.idx1se                         // Index of lambda.1se (1 SE rule)
model.getLambda(fitIdx)              // Lambda value at index
model.getDeviance(fitIdx)            // Deviance at index
model.getDf(fitIdx)                  // Degrees of freedom at index
model.getCvMean(fitIdx)              // CV mean error at index
model.getCvSe(fitIdx)               // CV standard error at index
model.getCoefs(fitIdx?)             // Coefficient vector (intercept at [0])

// Persistence
const bundle = model.save()          // Returns Uint8Array (wlearn bundle)
const loaded = await GAMModel.load(bundle)

// Cleanup (required)
model.dispose()

// AutoML
GAMModel.defaultSearchSpace()

// Params
model.getParams()
model.setParams({ alpha: 0.5 })
model.capabilities                   // { classifier, regressor, predictProba, ... }
```

#### C API

```c
// Initialize params with defaults
void gam_params_init(gam_params_t *params);

// Fit a regularization path
gam_path_t *gam_fit(const double *X, int32_t nrow, int32_t ncol,
                     const double *y, const gam_params_t *params);

// Specialized fit functions
gam_path_t *gam_fit_cox(const double *X, int32_t nrow, int32_t ncol,
                         const double *time, const double *status,
                         const gam_params_t *params);
gam_path_t *gam_fit_multi(const double *X, int32_t nrow, int32_t ncol,
                           const double *Y, int32_t n_tasks,
                           const gam_params_t *params);
gam_path_t *gam_fit_multinomial(const double *X, int32_t nrow, int32_t ncol,
                                 const double *y, int32_t n_classes,
                                 const gam_params_t *params);

// Predict
int gam_predict(const gam_path_t *path, int32_t fit_idx,
                const double *X, int32_t nrow, int32_t ncol, double *out);
int gam_predict_eta(const gam_path_t *path, int32_t fit_idx,
                    const double *X, int32_t nrow, int32_t ncol, double *out);
int gam_predict_proba(const gam_path_t *path, int32_t fit_idx,
                      const double *X, int32_t nrow, int32_t ncol, double *out);
int gam_predict_multi(const gam_path_t *path, int32_t fit_idx,
                      const double *X, int32_t nrow, int32_t ncol, double *out);
int gam_predict_multinomial(const gam_path_t *path, int32_t fit_idx,
                            const double *X, int32_t nrow, int32_t ncol, double *out);

// GAMLSS distributional regression
gam_path_t *gam_fit_gamlss(const double *X, int32_t nrow, int32_t ncol,
                            const double *y, int32_t distribution,
                            const gam_params_t *params);
int gam_predict_gamlss(const gam_path_t *path, int32_t fit_idx,
                        const double *X, int32_t nrow, int32_t ncol, double *out);

// Serialize / deserialize
int gam_save(const gam_path_t *path, char **out_buf, int32_t *out_len);
gam_path_t *gam_load(const char *buf, int32_t len);

// Free resources
void gam_free(gam_path_t *path);
void gam_free_buffer(void *ptr);

// B-spline utility
int gam_bspline_basis(const double *x, int32_t n, const double *knots,
                      int32_t n_knots, int32_t degree, double *out);

// Error message
const char *gam_get_error(void);
```

## Building

### Native (C)

```bash
mkdir build && cd build
cmake .. -DBUILD_TESTING=ON
make
./test_gam                    # 33 tests, 3904 assertions
```

### WASM

Requires Emscripten.

```bash
bash scripts/build-wasm.sh    # outputs wasm/gam.js
bash scripts/verify-exports.sh # verifies 34 exports
```

### JS Tests

```bash
node test/test.js             # 50 tests
```

### Python Tests

```bash
python test/test_python.py    # 31 tests (needs build/libgam.so)
```

## References

- Friedman, Hastie, Tibshirani (2010). "Regularization Paths for GLMs via Coordinate Descent." J. Stat. Software, 33(1).
- Zhang (2010). "Nearly Unbiased Variable Selection under MCP." Annals of Statistics, 38(2).
- Fan, Li (2001). "Variable Selection via Nonconcave Penalized Likelihood." JASA, 96(456).
- Bogdan et al. (2015). "SLOPE -- Adaptive Variable Selection via Convex Optimization." Annals of Applied Statistics, 9(3).
- Yuan, Lin (2006). "Model Selection and Estimation in Regression with Grouped Variables." JRSS-B, 68(1).
- Simon et al. (2013). "A Sparse-Group Lasso." J. Comp. Graph. Stat., 22(2).
- Tibshirani (2005). "Sparsity and Smoothness via the Fused Lasso." JRSS-B, 67(1).
- Simon et al. (2011). "Regularization Paths for Cox's PH Model via Coordinate Descent." J. Stat. Software, 39(5).
- Bertrand, Massias (2021). "Anderson Acceleration of Coordinate Descent." AISTATS.
- Fercoq et al. (2015). "Mind the Duality Gap: Safer Rules for the Lasso." ICML.
- Yi, Huang (2017). "Semismooth Newton CD for Elastic-Net Penalized Huber Loss and Quantile Regression." J. Comp. Graph. Stat.
- Rigby, Stasinopoulos (2005). "GAMLSS." J. Royal Stat. Soc. C, 54(3).
- Wood (2006). "Generalized Additive Models: An Introduction with R." Chapman & Hall/CRC.

## License

Apache-2.0
