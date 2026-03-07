# Changelog

## 0.1.0 (2026-03-07)

Initial release.

### Families

- Gaussian, Binomial, Poisson, Gamma, Inverse Gaussian, Negative Binomial, Tweedie
- Multinomial logistic regression (K-class softmax with full regularization path)
- Cox proportional hazards (partial likelihood)
- Huber robust regression
- Quantile regression (check loss)
- GAMLSS distributional regression: Normal(mu, sigma), Gamma(mu, sigma), Beta(mu, phi)

### Penalties

- Lasso (L1), Ridge (L2), Elastic net
- MCP (minimax concave penalty)
- SCAD (smoothly clipped absolute deviation)
- SLOPE (sorted L1 with FDR control)
- Group lasso, Sparse group lasso
- Fused lasso / trend filtering (ADMM solver)
- Multi-task lasso (shared support across responses)
- Per-feature penalty factors

### GAM

- B-spline basis functions (de Boor recursion)
- Quantile-spaced knots
- Smoothness penalties (Gauss-Legendre quadrature)
- Tensor product smooths (row-wise Kronecker product)

### Optimization

- Pathwise coordinate descent with warm starts
- Proximal Newton (IRLS outer + CD inner)
- Anderson acceleration
- GAP Safe screening rules
- Strong screening rules
- K-fold cross-validation (lambda.min / lambda.1se)
- Relaxed fits

### Infrastructure

- C11 core (~5000 lines), no dependencies beyond libc/libm
- WASM build (Emscripten, single-file)
- JS wrapper with wlearn Estimator API
- Python ctypes bindings
- Binary serialization (GAM1 format)
- 33 C tests (3904 assertions), 50 JS tests, 31 Python tests
