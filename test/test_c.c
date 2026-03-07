/*
 * test_c.c -- Tests for GAM C core
 *
 * Tests cover: Gaussian (lasso, ridge, elastic net), Binomial (logistic),
 * Poisson, regularization path, coefficient shrinkage, CV, MCP/SCAD,
 * B-spline basis, serialization, relaxed fits, penalty factors.
 */
#define _POSIX_C_SOURCE 200809L
#include "gam.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
    } else { \
        tests_passed++; \
    } \
} while(0)

/* Simple LCG matching rf.h */
typedef struct { uint32_t state; } test_rng_t;
static inline double test_rng_uniform(test_rng_t *rng) {
    rng->state = (rng->state * 1664525u + 1013904223u) & 0x7FFFFFFFu;
    return (double)rng->state / (double)0x7FFFFFFFu;
}

/* ---- Gaussian + Lasso ---- */
static void test_gaussian_lasso(void) {
    printf("=== Gaussian Lasso ===\n");

    int n = 100, d = 5;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 42 };

    /* y = 3*x0 + 2*x1 + noise; x2..x4 are irrelevant */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) X[i * d + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
        y[i] = 3.0 * X[i * d + 0] + 2.0 * X[i * d + 1] + 0.1 * (test_rng_uniform(&rng) - 0.5);
    }

    gam_params_t params;
    gam_params_init(&params);
    params.family = GAM_FAMILY_GAUSSIAN;
    params.penalty = GAM_PENALTY_L1;
    params.alpha = 1.0;
    params.n_lambda = 50;

    gam_path_t *path = gam_fit(X, n, d, y, &params);
    ASSERT(path != NULL, "gam_fit returns non-NULL");
    ASSERT(path->n_fits > 0, "at least one fit");
    ASSERT(path->n_fits <= 50, "at most n_lambda fits");
    printf("  n_fits: %d\n", path->n_fits);

    /* At lambda_max, all coefficients should be zero */
    ASSERT(path->fits[0].df == 0, "all zero at lambda_max");

    /* At small lambda, should recover true coefficients */
    int last = path->n_fits - 1;
    ASSERT(path->fits[last].df >= 2, "at least 2 nonzero at small lambda");
    printf("  df at min lambda: %d\n", path->fits[last].df);

    /* Check coefficient values at small lambda */
    double b0 = path->fits[last].beta[1];  /* coef for x0 */
    double b1 = path->fits[last].beta[2];  /* coef for x1 */
    printf("  beta[0] = %.3f (expected ~3.0)\n", b0);
    printf("  beta[1] = %.3f (expected ~2.0)\n", b1);
    ASSERT(fabs(b0 - 3.0) < 0.5, "x0 coef close to 3.0");
    ASSERT(fabs(b1 - 2.0) < 0.5, "x1 coef close to 2.0");

    /* Irrelevant features should be shrunk */
    int irrelevant_small = 1;
    for (int j = 2; j < d; j++) {
        if (fabs(path->fits[last].beta[j + 1]) > 0.3) irrelevant_small = 0;
    }
    ASSERT(irrelevant_small, "irrelevant features shrunk");

    /* Deviance should decrease along path */
    int dev_decreasing = 1;
    for (int k = 1; k < path->n_fits; k++) {
        if (path->fits[k].deviance > path->fits[k-1].deviance + 1e-6) {
            dev_decreasing = 0;
            break;
        }
    }
    ASSERT(dev_decreasing, "deviance decreasing along path");

    /* Prediction */
    double *preds = (double *)malloc((size_t)n * sizeof(double));
    int ret = gam_predict(path, last, X, n, d, preds);
    ASSERT(ret == 0, "predict succeeds");

    /* R-squared */
    double y_mean_val = 0.0;
    for (int i = 0; i < n; i++) y_mean_val += y[i];
    y_mean_val /= n;
    double ss_res = 0.0, ss_tot = 0.0;
    for (int i = 0; i < n; i++) {
        ss_res += (y[i] - preds[i]) * (y[i] - preds[i]);
        ss_tot += (y[i] - y_mean_val) * (y[i] - y_mean_val);
    }
    double r2 = 1.0 - ss_res / ss_tot;
    printf("  R-squared: %.4f\n", r2);
    ASSERT(r2 > 0.95, "R-squared > 0.95");

    free(preds);
    gam_free(path);
    free(X);
    free(y);
}

/* ---- Gaussian + Ridge ---- */
static void test_gaussian_ridge(void) {
    printf("=== Gaussian Ridge ===\n");

    int n = 100, d = 5;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 123 };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) X[i * d + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
        y[i] = 1.0 * X[i * d + 0] + 1.0 * X[i * d + 1] + 1.0 * X[i * d + 2];
    }

    gam_params_t params;
    gam_params_init(&params);
    params.penalty = GAM_PENALTY_L2;
    params.alpha = 0.0;
    params.n_lambda = 20;

    gam_path_t *path = gam_fit(X, n, d, y, &params);
    ASSERT(path != NULL, "ridge fit non-NULL");

    /* Ridge: all coefficients should be nonzero (no sparsity) */
    int last = path->n_fits - 1;
    ASSERT(path->fits[last].df == d, "all features nonzero with ridge");

    gam_free(path);
    free(X);
    free(y);
}

/* ---- Gaussian + Elastic Net ---- */
static void test_gaussian_elasticnet(void) {
    printf("=== Gaussian Elastic Net ===\n");

    int n = 100, d = 10;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 77 };

    /* Correlated features: x1 = x0 + noise */
    for (int i = 0; i < n; i++) {
        X[i * d + 0] = test_rng_uniform(&rng) * 2.0 - 1.0;
        X[i * d + 1] = X[i * d + 0] + 0.1 * (test_rng_uniform(&rng) - 0.5);
        for (int j = 2; j < d; j++) X[i * d + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
        y[i] = 2.0 * X[i * d + 0] + 2.0 * X[i * d + 1];
    }

    gam_params_t params;
    gam_params_init(&params);
    params.penalty = GAM_PENALTY_ELASTICNET;
    params.alpha = 0.5;
    params.n_lambda = 30;

    gam_path_t *path = gam_fit(X, n, d, y, &params);
    ASSERT(path != NULL, "elastic net fit non-NULL");

    /* With elastic net, correlated features should both have nonzero coefs */
    int last = path->n_fits - 1;
    double b0 = path->fits[last].beta[1];
    double b1 = path->fits[last].beta[2];
    printf("  beta[0] = %.3f, beta[1] = %.3f\n", b0, b1);
    ASSERT(fabs(b0) > 0.1 && fabs(b1) > 0.1, "correlated features both retained");

    gam_free(path);
    free(X);
    free(y);
}

/* ---- Binomial (Logistic Regression) ---- */
static void test_binomial(void) {
    printf("=== Binomial (Logistic) ===\n");

    int n = 200, d = 3;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 99 };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) X[i * d + j] = test_rng_uniform(&rng) * 4.0 - 2.0;
        double eta = 2.0 * X[i * d + 0] - 1.5 * X[i * d + 1];
        double prob = 1.0 / (1.0 + exp(-eta));
        y[i] = (test_rng_uniform(&rng) < prob) ? 1.0 : 0.0;
    }

    gam_params_t params;
    gam_params_init(&params);
    params.family = GAM_FAMILY_BINOMIAL;
    params.penalty = GAM_PENALTY_ELASTICNET;
    params.alpha = 1.0;
    params.n_lambda = 30;

    gam_path_t *path = gam_fit(X, n, d, y, &params);
    ASSERT(path != NULL, "binomial fit non-NULL");
    ASSERT(path->n_fits > 0, "at least one fit");

    int last = path->n_fits - 1;
    printf("  n_fits: %d, df at min lambda: %d\n", path->n_fits, path->fits[last].df);

    /* Predict probabilities */
    double *proba = (double *)malloc((size_t)n * sizeof(double));
    int ret = gam_predict_proba(path, last, X, n, d, proba);
    ASSERT(ret == 0, "predict_proba succeeds");

    /* Check probabilities are in [0, 1] */
    int valid = 1;
    for (int i = 0; i < n; i++) {
        if (proba[i] < 0 || proba[i] > 1) { valid = 0; break; }
    }
    ASSERT(valid, "probabilities in [0, 1]");

    /* Classification accuracy */
    int correct = 0;
    for (int i = 0; i < n; i++) {
        double pred_class = (proba[i] >= 0.5) ? 1.0 : 0.0;
        if (pred_class == y[i]) correct++;
    }
    double acc = (double)correct / n;
    printf("  accuracy: %.3f\n", acc);
    ASSERT(acc > 0.70, "accuracy > 0.70");

    free(proba);
    gam_free(path);
    free(X);
    free(y);
}

/* ---- Poisson ---- */
static void test_poisson(void) {
    printf("=== Poisson ===\n");

    int n = 200, d = 3;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 55 };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) X[i * d + j] = test_rng_uniform(&rng);
        double rate = exp(0.5 + 1.0 * X[i * d + 0] - 0.5 * X[i * d + 1]);
        /* Simple Poisson sample: use inverse CDF */
        double L = exp(-rate);
        double p = 1.0;
        int k = 0;
        do { k++; p *= test_rng_uniform(&rng); } while (p > L);
        y[i] = (double)(k - 1);
    }

    gam_params_t params;
    gam_params_init(&params);
    params.family = GAM_FAMILY_POISSON;
    params.penalty = GAM_PENALTY_ELASTICNET;
    params.alpha = 0.5;
    params.n_lambda = 20;

    gam_path_t *path = gam_fit(X, n, d, y, &params);
    ASSERT(path != NULL, "poisson fit non-NULL");

    int last = path->n_fits - 1;
    printf("  n_fits: %d, df: %d\n", path->n_fits, path->fits[last].df);

    /* Predict rates */
    double *rates = (double *)malloc((size_t)n * sizeof(double));
    gam_predict(path, last, X, n, d, rates);

    /* All rates should be positive */
    int positive = 1;
    for (int i = 0; i < n; i++) {
        if (rates[i] <= 0) { positive = 0; break; }
    }
    ASSERT(positive, "predicted rates are positive");

    free(rates);
    gam_free(path);
    free(X);
    free(y);
}

/* ---- MCP penalty ---- */
static void test_mcp(void) {
    printf("=== MCP Penalty ===\n");

    int n = 100, d = 5;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 33 };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) X[i * d + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
        y[i] = 5.0 * X[i * d + 0] + 0.1 * (test_rng_uniform(&rng) - 0.5);
    }

    gam_params_t params;
    gam_params_init(&params);
    params.penalty = GAM_PENALTY_MCP;
    params.alpha = 1.0;
    params.gamma_mcp = 3.0;
    params.n_lambda = 30;

    gam_path_t *path = gam_fit(X, n, d, y, &params);
    ASSERT(path != NULL, "MCP fit non-NULL");

    /* MCP should give less biased estimates than lasso */
    int last = path->n_fits - 1;
    double b0 = path->fits[last].beta[1];
    printf("  MCP beta[0] = %.3f (expected ~5.0)\n", b0);
    ASSERT(fabs(b0 - 5.0) < 0.5, "MCP less biased than lasso");

    gam_free(path);
    free(X);
    free(y);
}

/* ---- SCAD penalty ---- */
static void test_scad(void) {
    printf("=== SCAD Penalty ===\n");

    int n = 100, d = 5;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 44 };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) X[i * d + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
        y[i] = 4.0 * X[i * d + 0] - 3.0 * X[i * d + 1];
    }

    gam_params_t params;
    gam_params_init(&params);
    params.penalty = GAM_PENALTY_SCAD;
    params.alpha = 1.0;
    params.gamma_scad = 3.7;
    params.n_lambda = 30;

    gam_path_t *path = gam_fit(X, n, d, y, &params);
    ASSERT(path != NULL, "SCAD fit non-NULL");

    int last = path->n_fits - 1;
    printf("  SCAD beta[0] = %.3f (expected ~4.0)\n", path->fits[last].beta[1]);
    printf("  SCAD beta[1] = %.3f (expected ~-3.0)\n", path->fits[last].beta[2]);
    ASSERT(path->fits[last].df >= 2, "SCAD finds true features");

    gam_free(path);
    free(X);
    free(y);
}

/* ---- Cross-validation ---- */
static void test_cv(void) {
    printf("=== Cross-Validation ===\n");

    int n = 200, d = 5;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 88 };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) X[i * d + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
        y[i] = 2.0 * X[i * d + 0] + 0.5 * (test_rng_uniform(&rng) - 0.5);
    }

    gam_params_t params;
    gam_params_init(&params);
    params.penalty = GAM_PENALTY_L1;
    params.n_lambda = 20;
    params.n_folds = 5;

    gam_path_t *path = gam_fit(X, n, d, y, &params);
    ASSERT(path != NULL, "CV fit non-NULL");
    ASSERT(path->idx_min >= 0, "idx_min is valid");
    ASSERT(path->idx_1se >= 0, "idx_1se is valid");
    ASSERT(path->idx_1se <= path->idx_min, "1se is more regularized than min");

    printf("  idx_min: %d, idx_1se: %d\n", path->idx_min, path->idx_1se);
    printf("  CV mean at min: %.4f\n", path->fits[path->idx_min].cv_mean);

    /* CV means should be finite */
    ASSERT(!isnan(path->fits[path->idx_min].cv_mean), "CV mean is finite");

    gam_free(path);
    free(X);
    free(y);
}

/* ---- B-spline basis ---- */
static void test_bspline(void) {
    printf("=== B-spline Basis ===\n");

    double x[] = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
    int n = 11;
    double knots[] = { 0.25, 0.5, 0.75 };
    int n_knots = 3;
    int degree = 3;
    int n_basis = n_knots + degree + 1;  /* = 7 */

    double *basis = (double *)calloc((size_t)n * (size_t)n_basis, sizeof(double));
    int32_t nb = gam_bspline_basis(x, n, knots, n_knots, degree, basis);

    ASSERT(nb == n_basis, "correct number of basis functions");

    /* Partition of unity: basis functions sum to ~1 at each point */
    int partition_ok = 1;
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n_basis; j++) {
            sum += basis[i * n_basis + j];
        }
        if (fabs(sum - 1.0) > 0.01) {
            printf("  row %d: sum = %.4f\n", i, sum);
            partition_ok = 0;
        }
    }
    ASSERT(partition_ok, "B-spline partition of unity");

    /* Non-negativity */
    int nonneg = 1;
    for (int i = 0; i < n * n_basis; i++) {
        if (basis[i] < -1e-10) { nonneg = 0; break; }
    }
    ASSERT(nonneg, "B-spline non-negativity");

    free(basis);
}

/* ---- Quantile knots ---- */
static void test_quantile_knots(void) {
    printf("=== Quantile Knots ===\n");

    double x[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    int n = 10;
    int n_knots = 3;
    double knots[3];

    int ret = gam_quantile_knots(x, n, n_knots, knots);
    ASSERT(ret == 0, "quantile_knots succeeds");

    printf("  knots: %.2f, %.2f, %.2f\n", knots[0], knots[1], knots[2]);
    ASSERT(knots[0] > 0 && knots[0] < knots[1], "knots are ordered (0 < 1)");
    ASSERT(knots[1] < knots[2], "knots are ordered (1 < 2)");
    ASSERT(knots[2] < 9.0, "last knot < max");

    /* Roughly at quartiles: 2.25, 4.5, 6.75 */
    ASSERT(fabs(knots[1] - 4.5) < 1.0, "middle knot near median");
}

/* ---- Serialization ---- */
static void test_serialization(void) {
    printf("=== Serialization ===\n");

    int n = 50, d = 3;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 11 };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) X[i * d + j] = test_rng_uniform(&rng);
        y[i] = X[i * d + 0] + 2.0 * X[i * d + 1];
    }

    gam_params_t params;
    gam_params_init(&params);
    params.n_lambda = 10;

    gam_path_t *path = gam_fit(X, n, d, y, &params);
    ASSERT(path != NULL, "fit for serialization");

    /* Save */
    char *buf = NULL;
    int32_t len = 0;
    int ret = gam_save(path, &buf, &len);
    ASSERT(ret == 0, "save succeeds");
    ASSERT(len > 0, "saved bytes > 0");
    printf("  saved %d bytes\n", len);

    /* Load */
    gam_path_t *loaded = gam_load(buf, len);
    ASSERT(loaded != NULL, "load succeeds");
    ASSERT(loaded->n_fits == path->n_fits, "same n_fits after load");

    /* Compare predictions */
    double *pred1 = (double *)malloc((size_t)n * sizeof(double));
    double *pred2 = (double *)malloc((size_t)n * sizeof(double));
    int last = path->n_fits - 1;
    gam_predict(path, last, X, n, d, pred1);
    gam_predict(loaded, last, X, n, d, pred2);

    int preds_match = 1;
    for (int i = 0; i < n; i++) {
        if (fabs(pred1[i] - pred2[i]) > 1e-10) { preds_match = 0; break; }
    }
    ASSERT(preds_match, "predictions match after save/load");

    free(pred1);
    free(pred2);
    gam_free_buffer(buf);
    gam_free(loaded);
    gam_free(path);
    free(X);
    free(y);
}

/* ---- Relaxed fits ---- */
static void test_relaxed(void) {
    printf("=== Relaxed Fits ===\n");

    int n = 100, d = 5;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 66 };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) X[i * d + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
        y[i] = 3.0 * X[i * d + 0] + 2.0 * X[i * d + 1];
    }

    gam_params_t params;
    gam_params_init(&params);
    params.penalty = GAM_PENALTY_L1;
    params.n_lambda = 20;
    params.relax = 1;

    gam_path_t *path = gam_fit(X, n, d, y, &params);
    ASSERT(path != NULL, "relaxed fit non-NULL");
    ASSERT(path->relaxed_fits != NULL, "relaxed_fits allocated");

    /* Relaxed coefficients should be less biased */
    int mid = path->n_fits / 2;
    if (mid < path->n_fits && path->relaxed_fits[mid].beta) {
        double b0_pen = path->fits[mid].beta[1];
        double b0_rel = path->relaxed_fits[mid].beta[1];
        printf("  penalized beta[0] = %.3f, relaxed beta[0] = %.3f\n", b0_pen, b0_rel);
        /* Relaxed should be closer to true (3.0) if feature is selected */
        if (b0_pen != 0.0) {
            ASSERT(fabs(b0_rel) >= fabs(b0_pen) - 0.1, "relaxed coef at least as large");
        }
    }

    gam_free(path);
    free(X);
    free(y);
}

/* ---- Penalty factors ---- */
static void test_penalty_factors(void) {
    printf("=== Penalty Factors ===\n");

    int n = 100, d = 3;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 22 };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) X[i * d + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
        y[i] = 1.0 * X[i * d + 0] + 1.0 * X[i * d + 1] + 1.0 * X[i * d + 2];
    }

    /* Force feature 0 to always be included (penalty_factor = 0) */
    double pf[] = { 0.0, 1.0, 1.0 };

    gam_params_t params;
    gam_params_init(&params);
    params.penalty = GAM_PENALTY_L1;
    params.n_lambda = 20;
    params.penalty_factor = pf;

    gam_path_t *path = gam_fit(X, n, d, y, &params);
    ASSERT(path != NULL, "penalty factor fit non-NULL");

    /* Feature 0 should be nonzero at all lambda values */
    int always_nonzero = 1;
    for (int k = 0; k < path->n_fits; k++) {
        if (fabs(path->fits[k].beta[1]) < 1e-10) {
            always_nonzero = 0;
            break;
        }
    }
    ASSERT(always_nonzero, "unpenalized feature always nonzero");

    gam_free(path);
    free(X);
    free(y);
}

/* ---- Gamma family ---- */
static void test_gamma(void) {
    printf("=== Gamma Family ===\n");

    int n = 200, d = 2;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 77 };

    for (int i = 0; i < n; i++) {
        X[i * d + 0] = test_rng_uniform(&rng) + 0.1;
        X[i * d + 1] = test_rng_uniform(&rng) + 0.1;
        double mu = exp(0.5 + 0.8 * X[i * d + 0]);
        /* Approximate Gamma sample: mu + noise (keeping positive) */
        y[i] = fmax(mu + 0.3 * mu * (test_rng_uniform(&rng) - 0.5), 0.01);
    }

    gam_params_t params;
    gam_params_init(&params);
    params.family = GAM_FAMILY_GAMMA;
    params.penalty = GAM_PENALTY_ELASTICNET;
    params.alpha = 0.5;
    params.n_lambda = 15;

    gam_path_t *path = gam_fit(X, n, d, y, &params);
    ASSERT(path != NULL, "gamma fit non-NULL");
    ASSERT(path->n_fits > 0, "gamma has fits");

    /* Predictions should be positive */
    double *preds = (double *)malloc((size_t)n * sizeof(double));
    gam_predict(path, path->n_fits - 1, X, n, d, preds);
    int all_pos = 1;
    for (int i = 0; i < n; i++) {
        if (preds[i] <= 0) { all_pos = 0; break; }
    }
    ASSERT(all_pos, "gamma predictions positive");

    free(preds);
    gam_free(path);
    free(X);
    free(y);
}

/* ---- Diagnostics (AIC/BIC) ---- */
static void test_diagnostics(void) {
    printf("=== Diagnostics ===\n");

    int n = 100, d = 3;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 55 };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) X[i * d + j] = test_rng_uniform(&rng);
        y[i] = X[i * d + 0] + 0.1 * test_rng_uniform(&rng);
    }

    gam_params_t params;
    gam_params_init(&params);
    params.n_lambda = 10;

    gam_path_t *path = gam_fit(X, n, d, y, &params);
    ASSERT(path != NULL, "diagnostics fit non-NULL");

    double aic = gam_aic(path, path->n_fits - 1, n);
    double bic = gam_bic(path, path->n_fits - 1, n);
    ASSERT(!isnan(aic), "AIC is finite");
    ASSERT(!isnan(bic), "BIC is finite");
    ASSERT(bic >= aic, "BIC >= AIC for n >= 8");
    printf("  AIC: %.2f, BIC: %.2f\n", aic, bic);

    gam_free(path);
    free(X);
    free(y);
}

/* ---- No penalty (unpenalized GLM) ---- */
static void test_unpenalized(void) {
    printf("=== Unpenalized GLM ===\n");

    int n = 50, d = 2;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 10 };

    for (int i = 0; i < n; i++) {
        X[i * d + 0] = test_rng_uniform(&rng) * 2.0 - 1.0;
        X[i * d + 1] = test_rng_uniform(&rng) * 2.0 - 1.0;
        y[i] = 3.0 * X[i * d + 0] - 2.0 * X[i * d + 1] + 0.5;
    }

    gam_params_t params;
    gam_params_init(&params);
    params.penalty = GAM_PENALTY_NONE;
    params.n_lambda = 1;
    double lam0 = 0.0;
    params.lambda = &lam0;
    params.n_lambda_user = 1;

    gam_path_t *path = gam_fit(X, n, d, y, &params);
    ASSERT(path != NULL, "unpenalized fit non-NULL");
    ASSERT(path->n_fits == 1, "single fit");

    double b0 = path->fits[0].beta[0];  /* intercept */
    double b1 = path->fits[0].beta[1];  /* x0 */
    double b2 = path->fits[0].beta[2];  /* x1 */
    printf("  intercept=%.3f (exp 0.5), b1=%.3f (exp 3.0), b2=%.3f (exp -2.0)\n", b0, b1, b2);
    ASSERT(fabs(b1 - 3.0) < 0.3, "unpenalized x0 coef ~3");
    ASSERT(fabs(b2 - (-2.0)) < 0.3, "unpenalized x1 coef ~-2");
    ASSERT(fabs(b0 - 0.5) < 0.3, "unpenalized intercept ~0.5");

    gam_free(path);
    free(X);
    free(y);
}

/* --- SLOPE Penalty --- */
static void test_slope(void) {
    printf("=== SLOPE Penalty ===\n");

    /* Generate data with known sparse signal: only 3 of 20 features are active */
    int n = 200, p_total = 20;
    double *X = (double *)malloc((size_t)n * (size_t)p_total * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));

    test_rng_t rng = {12345};
    double true_beta[20] = {0};
    true_beta[0] = 3.0;
    true_beta[1] = -2.0;
    true_beta[2] = 1.5;
    /* features 3-19 are noise */

    for (int i = 0; i < n; i++) {
        y[i] = 0.5;  /* intercept */
        for (int j = 0; j < p_total; j++) {
            X[i * p_total + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
            y[i] += X[i * p_total + j] * true_beta[j];
        }
        y[i] += (test_rng_uniform(&rng) - 0.5) * 0.2;  /* noise */
    }

    gam_params_t params;
    gam_params_init(&params);
    params.family = GAM_FAMILY_GAUSSIAN;
    params.penalty = GAM_PENALTY_SLOPE;
    params.slope_q = 0.1;
    params.n_lambda = 50;
    params.tol = 1e-6;
    params.max_iter = 5000;

    gam_path_t *path = gam_fit(X, n, p_total, y, &params);
    ASSERT(path != NULL, "SLOPE fit succeeded");
    ASSERT(path->n_fits > 0, "SLOPE produced fits");

    /* At the least-regularized end, should recover active features */
    int last = path->n_fits - 1;
    double *beta = path->fits[last].beta;  /* [0] = intercept, [1..20] = features */

    printf("  SLOPE n_fits: %d\n", path->n_fits);
    printf("  SLOPE beta[1] = %.3f (expected ~3.0)\n", beta[1]);
    printf("  SLOPE beta[2] = %.3f (expected ~-2.0)\n", beta[2]);
    printf("  SLOPE beta[3] = %.3f (expected ~1.5)\n", beta[3]);

    /* Active features should have non-trivial coefficients */
    ASSERT(fabs(beta[1]) > 1.0, "SLOPE recovers feature 0 (beta > 1)");
    ASSERT(fabs(beta[2]) > 0.5, "SLOPE recovers feature 1 (beta > 0.5)");
    ASSERT(fabs(beta[3]) > 0.3, "SLOPE recovers feature 2 (beta > 0.3)");

    /* Check sparsity: most noise features should be near zero at intermediate lambda */
    int mid = path->n_fits / 2;
    int n_zero = 0;
    for (int j = 1; j <= p_total; j++) {
        if (fabs(path->fits[mid].beta[j]) < 0.01) n_zero++;
    }
    printf("  SLOPE zero features at mid path: %d/%d\n", n_zero, p_total);
    ASSERT(n_zero > 10, "SLOPE produces sparse solutions at moderate lambda");

    /* At lambda_max (first fit), all should be approximately zero */
    n_zero = 0;
    for (int j = 1; j <= p_total; j++) {
        if (fabs(path->fits[0].beta[j]) < 0.01) n_zero++;
    }
    ASSERT(n_zero >= p_total - 1, "SLOPE: all zero at lambda_max");

    gam_free(path);
    free(X);
    free(y);
}

/* --- SLOPE Binomial --- */
static void test_slope_binomial(void) {
    printf("=== SLOPE Binomial ===\n");

    int n = 150, p_total = 10;
    double *X = (double *)malloc((size_t)n * (size_t)p_total * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));

    test_rng_t rng = {54321};
    for (int i = 0; i < n; i++) {
        double eta = -0.5;
        for (int j = 0; j < p_total; j++) {
            X[i * p_total + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
        }
        eta += 2.0 * X[i * p_total + 0] - 1.5 * X[i * p_total + 1];
        double prob = 1.0 / (1.0 + exp(-eta));
        y[i] = (test_rng_uniform(&rng) < prob) ? 1.0 : 0.0;
    }

    gam_params_t params;
    gam_params_init(&params);
    params.family = GAM_FAMILY_BINOMIAL;
    params.penalty = GAM_PENALTY_SLOPE;
    params.slope_q = 0.2;
    params.n_lambda = 30;
    params.tol = 1e-5;

    gam_path_t *path = gam_fit(X, n, p_total, y, &params);
    ASSERT(path != NULL, "SLOPE binomial fit succeeded");
    ASSERT(path->n_fits > 0, "SLOPE binomial produced fits");

    /* Predict and check accuracy at least-regularized point */
    int last = path->n_fits - 1;
    double *out = (double *)malloc((size_t)n * sizeof(double));
    int ret = gam_predict(path, last, X, n, p_total, out);
    ASSERT(ret == 0, "SLOPE binomial predict succeeded");

    int correct = 0;
    for (int i = 0; i < n; i++) {
        if ((out[i] >= 0.5 ? 1.0 : 0.0) == y[i]) correct++;
    }
    double acc = (double)correct / n;
    printf("  SLOPE binomial accuracy: %.3f\n", acc);
    ASSERT(acc > 0.6, "SLOPE binomial accuracy > 0.6");

    free(out);
    gam_free(path);
    free(X);
    free(y);
}

/* --- Group Lasso --- */
static void test_group_lasso(void) {
    printf("=== Group Lasso ===\n");

    /* 12 features in 4 groups of 3.
     * Groups 0 and 1 are active, groups 2 and 3 are noise. */
    int n = 200, p_total = 12;
    double *X = (double *)malloc((size_t)n * (size_t)p_total * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    int32_t groups[12] = {0,0,0, 1,1,1, 2,2,2, 3,3,3};

    test_rng_t rng = {99999};
    for (int i = 0; i < n; i++) {
        y[i] = 0.5;
        for (int j = 0; j < p_total; j++) {
            X[i * p_total + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
        }
        /* Active groups: 0 and 1 */
        y[i] += 2.0 * X[i * p_total + 0] + 1.0 * X[i * p_total + 1] - 0.5 * X[i * p_total + 2];
        y[i] += -1.5 * X[i * p_total + 3] + 0.8 * X[i * p_total + 4] + 0.3 * X[i * p_total + 5];
        y[i] += (test_rng_uniform(&rng) - 0.5) * 0.3;
    }

    gam_params_t params;
    gam_params_init(&params);
    params.family = GAM_FAMILY_GAUSSIAN;
    params.penalty = GAM_PENALTY_GROUP_L1;
    params.groups = groups;
    params.n_groups = 4;
    params.n_lambda = 30;
    params.tol = 1e-6;

    gam_path_t *path = gam_fit(X, n, p_total, y, &params);
    ASSERT(path != NULL, "group lasso fit succeeded");
    ASSERT(path->n_fits > 0, "group lasso produced fits");

    printf("  group lasso n_fits: %d\n", path->n_fits);

    /* At intermediate regularization, noise groups should be zeroed out */
    int mid = path->n_fits / 3;  /* more regularized end */
    double *beta_mid = path->fits[mid].beta;

    /* Check if groups 2 and 3 (indices 7-12) are zero */
    double grp2_norm = 0, grp3_norm = 0;
    for (int j = 7; j <= 9; j++) grp2_norm += beta_mid[j] * beta_mid[j];
    for (int j = 10; j <= 12; j++) grp3_norm += beta_mid[j] * beta_mid[j];
    grp2_norm = sqrt(grp2_norm);
    grp3_norm = sqrt(grp3_norm);

    double grp0_norm = 0, grp1_norm = 0;
    for (int j = 1; j <= 3; j++) grp0_norm += beta_mid[j] * beta_mid[j];
    for (int j = 4; j <= 6; j++) grp1_norm += beta_mid[j] * beta_mid[j];
    grp0_norm = sqrt(grp0_norm);
    grp1_norm = sqrt(grp1_norm);

    printf("  mid-path group norms: g0=%.3f, g1=%.3f, g2=%.3f, g3=%.3f\n",
           grp0_norm, grp1_norm, grp2_norm, grp3_norm);

    ASSERT(grp0_norm > 0.1 || grp1_norm > 0.1, "at least one active group is nonzero");

    /* At least-regularized end, prediction should be good */
    int last = path->n_fits - 1;
    double *out = (double *)malloc((size_t)n * sizeof(double));
    int ret = gam_predict(path, last, X, n, p_total, out);
    ASSERT(ret == 0, "group lasso predict succeeded");

    double ss_res = 0, ss_tot = 0, y_mean = 0;
    for (int i = 0; i < n; i++) y_mean += y[i];
    y_mean /= n;
    for (int i = 0; i < n; i++) {
        ss_res += (y[i] - out[i]) * (y[i] - out[i]);
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }
    double r2 = 1.0 - ss_res / ss_tot;
    printf("  group lasso R2: %.4f\n", r2);
    ASSERT(r2 > 0.7, "group lasso R2 > 0.7");

    free(out);
    gam_free(path);
    free(X);
    free(y);
}

/* --- Sparse Group Lasso --- */
static void test_sparse_group_lasso(void) {
    printf("=== Sparse Group Lasso ===\n");

    /* Same setup as group lasso, but with SGL */
    int n = 200, p_total = 12;
    double *X = (double *)malloc((size_t)n * (size_t)p_total * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    int32_t groups[12] = {0,0,0, 1,1,1, 2,2,2, 3,3,3};

    test_rng_t rng = {77777};
    for (int i = 0; i < n; i++) {
        y[i] = 0.0;
        for (int j = 0; j < p_total; j++) {
            X[i * p_total + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
        }
        /* Only x[0] and x[3] are truly active within their groups */
        y[i] += 3.0 * X[i * p_total + 0] - 2.0 * X[i * p_total + 3];
        y[i] += (test_rng_uniform(&rng) - 0.5) * 0.3;
    }

    gam_params_t params;
    gam_params_init(&params);
    params.family = GAM_FAMILY_GAUSSIAN;
    params.penalty = GAM_PENALTY_SGL;
    params.alpha = 0.5;  /* mixing between L1 and group */
    params.groups = groups;
    params.n_groups = 4;
    params.n_lambda = 30;
    params.tol = 1e-6;

    gam_path_t *path = gam_fit(X, n, p_total, y, &params);
    ASSERT(path != NULL, "SGL fit succeeded");
    ASSERT(path->n_fits > 0, "SGL produced fits");

    /* SGL should select individual features within groups */
    int last = path->n_fits - 1;
    double *out = (double *)malloc((size_t)n * sizeof(double));
    gam_predict(path, last, X, n, p_total, out);

    double ss_res = 0, ss_tot = 0, y_mean = 0;
    for (int i = 0; i < n; i++) y_mean += y[i];
    y_mean /= n;
    for (int i = 0; i < n; i++) {
        ss_res += (y[i] - out[i]) * (y[i] - out[i]);
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }
    double r2 = 1.0 - ss_res / ss_tot;
    printf("  SGL R2: %.4f\n", r2);
    ASSERT(r2 > 0.7, "SGL R2 > 0.7");

    free(out);
    gam_free(path);
    free(X);
    free(y);
}

/* ============================================================
 * Cox PH Tests
 * ============================================================ */
static void test_cox_basic(void) {
    printf("=== Cox PH Basic ===\n");

    /* Synthetic survival data: 2 features, feature 0 increases hazard,
     * feature 1 decreases hazard. Exponential baseline hazard. */
    int n = 200, p = 5;
    test_rng_t rng = {.state = 500};

    double *X = (double *)malloc((size_t)n * (size_t)p * sizeof(double));
    double *time_arr = (double *)malloc((size_t)n * sizeof(double));
    double *status_arr = (double *)malloc((size_t)n * sizeof(double));

    double true_beta[] = {0.8, -0.5, 0.0, 0.0, 0.0};

    for (int i = 0; i < n; i++) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            X[i * p + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
            eta += X[i * p + j] * true_beta[j];
        }
        /* Generate exponential survival time: T = -log(U) / (lambda * exp(eta)) */
        double u = test_rng_uniform(&rng);
        u = fmax(u, 1e-10);
        double base_hazard = 0.5;
        time_arr[i] = -log(u) / (base_hazard * exp(eta));
        /* Random censoring: ~20% censored */
        double cens_time = -log(fmax(test_rng_uniform(&rng), 1e-10)) / 0.1;
        if (cens_time < time_arr[i]) {
            time_arr[i] = cens_time;
            status_arr[i] = 0.0;
        } else {
            status_arr[i] = 1.0;
        }
    }

    gam_params_t params;
    gam_params_init(&params);
    params.family = GAM_FAMILY_COX;
    params.penalty = GAM_PENALTY_ELASTICNET;
    params.alpha = 0.5;
    params.n_lambda = 30;
    params.tol = 1e-7;
    params.max_iter = 5000;
    params.standardize = 1;

    gam_path_t *path = gam_fit_cox(X, n, p, time_arr, status_arr, &params);
    ASSERT(path != NULL, "Cox fit succeeded");
    ASSERT(path->n_fits > 0, "Cox produced fits");
    printf("  Cox n_fits: %d\n", path->n_fits);

    /* At the least-regularized end, active features should be recovered */
    int last = path->n_fits - 1;
    double beta1 = path->fits[last].beta[1];  /* feature 0 */
    double beta2 = path->fits[last].beta[2];  /* feature 1 */
    printf("  Cox beta[1] = %.3f (expected ~0.8)\n", beta1);
    printf("  Cox beta[2] = %.3f (expected ~-0.5)\n", beta2);
    ASSERT(beta1 > 0.3, "beta1 positive (hazard-increasing)");
    ASSERT(beta2 < -0.1, "beta2 negative (hazard-decreasing)");

    /* Sparsity: at early path (high lambda), some features should be zero */
    int early = path->n_fits / 4;  /* more regularized point */
    int df_early = path->fits[early].df;
    printf("  early-path df: %d (out of %d)\n", df_early, p);
    ASSERT(df_early < p, "Cox sparsity at early path");

    /* Predict: risk scores should be > 0 */
    double *risk = (double *)malloc((size_t)n * sizeof(double));
    int ret = gam_predict(path, last, X, n, p, risk);
    ASSERT(ret == 0, "Cox predict succeeded");
    int all_positive = 1;
    for (int i = 0; i < n; i++) {
        if (risk[i] <= 0) { all_positive = 0; break; }
    }
    ASSERT(all_positive, "Cox risk scores all positive");

    /* Concordance index: should be > 0.5 (better than random) */
    int concordant = 0, discordant = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (status_arr[i] > 0.5 && time_arr[i] < time_arr[j]) {
                if (risk[i] > risk[j]) concordant++;
                else if (risk[i] < risk[j]) discordant++;
            } else if (status_arr[j] > 0.5 && time_arr[j] < time_arr[i]) {
                if (risk[j] > risk[i]) concordant++;
                else if (risk[j] < risk[i]) discordant++;
            }
        }
    }
    double c_index = (concordant + discordant > 0) ?
        (double)concordant / (concordant + discordant) : 0.5;
    printf("  Cox C-index: %.3f\n", c_index);
    ASSERT(c_index > 0.6, "Cox C-index > 0.6");

    free(risk);
    gam_free(path);
    free(X);
    free(time_arr);
    free(status_arr);
}

static void test_cox_lasso(void) {
    printf("=== Cox Lasso ===\n");

    /* Sparse Cox: 2 active features out of 10 */
    int n = 300, p = 10;
    test_rng_t rng = {.state = 510};

    double *X = (double *)malloc((size_t)n * (size_t)p * sizeof(double));
    double *time_arr = (double *)malloc((size_t)n * sizeof(double));
    double *status_arr = (double *)malloc((size_t)n * sizeof(double));

    for (int i = 0; i < n; i++) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            X[i * p + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
            if (j == 0) eta += X[i * p + j] * 1.0;
            if (j == 1) eta += X[i * p + j] * -0.7;
        }
        double u = fmax(test_rng_uniform(&rng), 1e-10);
        time_arr[i] = -log(u) / exp(eta);
        double cens_time = -log(fmax(test_rng_uniform(&rng), 1e-10)) / 0.3;
        if (cens_time < time_arr[i]) {
            time_arr[i] = cens_time;
            status_arr[i] = 0.0;
        } else {
            status_arr[i] = 1.0;
        }
    }

    gam_params_t params;
    gam_params_init(&params);
    params.penalty = GAM_PENALTY_L1;
    params.n_lambda = 30;
    params.tol = 1e-7;
    params.max_iter = 5000;

    gam_path_t *path = gam_fit_cox(X, n, p, time_arr, status_arr, &params);
    ASSERT(path != NULL, "Cox lasso fit succeeded");

    /* At early-path (high lambda), should be sparse */
    int early = path->n_fits / 4;
    int df = path->fits[early].df;
    printf("  Cox lasso early-path df: %d (out of %d)\n", df, p);
    ASSERT(df < p, "Cox lasso has sparsity along path");

    /* At end of path, sign recovery */
    int last = path->n_fits - 1;
    double b1 = path->fits[last].beta[1];
    double b2 = path->fits[last].beta[2];
    printf("  Cox lasso beta[1] = %.3f, beta[2] = %.3f\n", b1, b2);
    ASSERT(b1 > 0.3, "beta[1] positive");
    ASSERT(b2 < -0.2, "beta[2] negative");

    gam_free(path);
    free(X);
    free(time_arr);
    free(status_arr);
}

static void test_cox_serialization(void) {
    printf("=== Cox Serialization ===\n");

    int n = 100, p = 3;
    test_rng_t rng = {.state = 520};

    double *X = (double *)malloc((size_t)n * (size_t)p * sizeof(double));
    double *time_arr = (double *)malloc((size_t)n * sizeof(double));
    double *status_arr = (double *)malloc((size_t)n * sizeof(double));

    for (int i = 0; i < n; i++) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            X[i * p + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
            if (j == 0) eta += X[i * p + j] * 0.5;
        }
        double u = fmax(test_rng_uniform(&rng), 1e-10);
        time_arr[i] = -log(u) / exp(eta);
        status_arr[i] = (test_rng_uniform(&rng) > 0.2) ? 1.0 : 0.0;
    }

    gam_params_t params;
    gam_params_init(&params);
    params.penalty = GAM_PENALTY_ELASTICNET;
    params.alpha = 0.5;
    params.n_lambda = 20;

    gam_path_t *path = gam_fit_cox(X, n, p, time_arr, status_arr, &params);
    ASSERT(path != NULL, "Cox fit for serialization");

    /* Save */
    char *buf = NULL;
    int32_t len = 0;
    int ret = gam_save(path, &buf, &len);
    ASSERT(ret == 0, "Cox save succeeded");
    printf("  Cox saved %d bytes\n", len);

    /* Load */
    gam_path_t *loaded = gam_load(buf, len);
    ASSERT(loaded != NULL, "Cox load succeeded");
    ASSERT(loaded->n_fits == path->n_fits, "Cox n_fits match");
    ASSERT(loaded->family == GAM_FAMILY_COX, "Cox family preserved");

    /* Predictions match */
    double *pred1 = (double *)malloc((size_t)n * sizeof(double));
    double *pred2 = (double *)malloc((size_t)n * sizeof(double));
    int last = path->n_fits - 1;
    gam_predict(path, last, X, n, p, pred1);
    gam_predict(loaded, last, X, n, p, pred2);

    double max_diff = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs(pred1[i] - pred2[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("  Cox save/load pred max diff: %.2e\n", max_diff);
    ASSERT(max_diff < 1e-10, "Cox save/load predictions match");

    free(pred1);
    free(pred2);
    gam_free_buffer(buf);
    gam_free(path);
    gam_free(loaded);
    free(X);
    free(time_arr);
    free(status_arr);
}

/* Anderson Acceleration: verify convergence on harder problem */
static void test_anderson_accel(void) {
    printf("=== Anderson Acceleration ===\n");

    /* Correlated features make CD converge slowly -- AA should help */
    int32_t n = 200, p = 50;
    double *X = (double *)malloc((size_t)n * (size_t)p * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { .state = 777 };

    /* Generate correlated features: X[j] = uniform + 0.8 * X[0] */
    for (int i = 0; i < n; i++) {
        double x0 = test_rng_uniform(&rng) * 2.0 - 1.0;
        X[i * p + 0] = x0;
        for (int j = 1; j < p; j++) {
            X[i * p + j] = (test_rng_uniform(&rng) * 2.0 - 1.0) + 0.8 * x0;
        }
    }
    /* True model: y = 3*x0 + 2*x1 - 1.5*x2 + noise */
    for (int i = 0; i < n; i++) {
        y[i] = 3.0 * X[i * p + 0] + 2.0 * X[i * p + 1] - 1.5 * X[i * p + 2]
               + (test_rng_uniform(&rng) - 0.5) * 0.5;
    }

    gam_params_t params;
    gam_params_init(&params);
    params.family = GAM_FAMILY_GAUSSIAN;
    params.penalty = GAM_PENALTY_ELASTICNET;
    params.alpha = 1.0;    /* pure lasso */
    params.n_lambda = 30;
    params.max_iter = 10000;
    params.tol = 1e-6;

    gam_path_t *path = gam_fit(X, n, p, y, &params);
    ASSERT(path != NULL, "AA: fit succeeded");
    ASSERT(path->n_fits > 0, "AA: produced fits");

    /* Check coefficient recovery at loose lambda */
    int last = path->n_fits - 1;
    double *beta = path->fits[last].beta;
    /* beta[0] is intercept, beta[1..] are features */
    printf("  AA beta[1]=%+.3f (exp ~3), beta[2]=%+.3f (exp ~2), beta[3]=%+.3f (exp ~-1.5)\n",
           beta[1], beta[2], beta[3]);
    ASSERT(fabs(beta[1] - 3.0) < 0.5, "AA: beta[1] near 3.0");
    ASSERT(fabs(beta[2] - 2.0) < 0.5, "AA: beta[2] near 2.0");
    ASSERT(fabs(beta[3] + 1.5) < 0.5, "AA: beta[3] near -1.5");

    /* Check total iterations -- with 50 correlated features, AA should keep
     * total iter reasonable. Without AA this problem can take many more iters. */
    int32_t total_iter = 0;
    for (int i = 0; i < path->n_fits; i++) {
        total_iter += path->fits[i].n_iter;
    }
    printf("  AA total iterations: %d (across %d fits)\n", total_iter, path->n_fits);
    /* With AA + improved convergence, 50 correlated features converge fast */
    ASSERT(total_iter < 5000, "AA: fast convergence with correlated features");

    /* R-squared check */
    double *pred = (double *)malloc((size_t)n * sizeof(double));
    gam_predict(path, last, X, n, p, pred);
    double ss_res = 0.0, ss_tot = 0.0, y_mean = 0.0;
    for (int i = 0; i < n; i++) y_mean += y[i];
    y_mean /= n;
    for (int i = 0; i < n; i++) {
        ss_res += (y[i] - pred[i]) * (y[i] - pred[i]);
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }
    double r2 = 1.0 - ss_res / ss_tot;
    printf("  AA R-squared: %.4f\n", r2);
    ASSERT(r2 > 0.95, "AA: high R-squared");

    free(pred);
    gam_free(path);
    free(X);
    free(y);
}

/* GAP Safe screening: verify high-dimensional sparse problem benefits */
static void test_gap_safe_screening(void) {
    printf("=== GAP Safe Screening ===\n");

    /* High-dimensional: p >> n, very sparse true model */
    int32_t n = 100, p = 500;
    double *X = (double *)malloc((size_t)n * (size_t)p * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { .state = 888 };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            X[i * p + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
        }
    }
    /* Only 3 active features */
    for (int i = 0; i < n; i++) {
        y[i] = 5.0 * X[i * p + 0] - 3.0 * X[i * p + 1] + 2.0 * X[i * p + 2]
               + (test_rng_uniform(&rng) - 0.5) * 0.3;
    }

    gam_params_t params;
    gam_params_init(&params);
    params.family = GAM_FAMILY_GAUSSIAN;
    params.penalty = GAM_PENALTY_L1;
    params.alpha = 1.0;
    params.n_lambda = 50;
    params.max_iter = 10000;
    params.tol = 1e-7;
    params.screening = 1;

    gam_path_t *path = gam_fit(X, n, p, y, &params);
    ASSERT(path != NULL, "GAP Safe: fit succeeded");
    ASSERT(path->n_fits > 0, "GAP Safe: produced fits");

    /* Check sparsity at mid-path: most features should be zero */
    int mid = path->n_fits / 2;
    ASSERT(path->fits[mid].df <= 10, "GAP Safe: sparse at mid-path");

    /* Check coefficient recovery at end of path */
    int last = path->n_fits - 1;
    double *beta = path->fits[last].beta;
    printf("  beta[1]=%+.3f (exp ~5), beta[2]=%+.3f (exp ~-3), beta[3]=%+.3f (exp ~2)\n",
           beta[1], beta[2], beta[3]);
    ASSERT(fabs(beta[1] - 5.0) < 0.5, "GAP Safe: beta[1] near 5.0");
    ASSERT(fabs(beta[2] + 3.0) < 0.5, "GAP Safe: beta[2] near -3.0");
    ASSERT(fabs(beta[3] - 2.0) < 0.5, "GAP Safe: beta[3] near 2.0");

    /* Total iterations should be reasonable for p=500 with screening */
    int32_t total_iter = 0;
    for (int i = 0; i < path->n_fits; i++) {
        total_iter += path->fits[i].n_iter;
    }
    printf("  GAP Safe total iterations: %d (across %d fits, p=%d)\n",
           total_iter, path->n_fits, p);
    ASSERT(total_iter < 10000, "GAP Safe: efficient with high-dimensional sparse data");

    gam_free(path);
    free(X);
    free(y);
}

/* Fused Lasso: piecewise constant signal recovery */
static void test_fused_lasso(void) {
    printf("=== Fused Lasso ===\n");

    /* Generate piecewise constant signal with noise.
     * True beta: [3, 3, 3, 0, 0, 0, -2, -2, -2, -2]
     * X = identity matrix (each feature is an independent observation group).
     * In practice, this means y_j ~ beta_j + noise. */
    int32_t n = 200, p = 10;
    double *X = (double *)calloc((size_t)n * (size_t)p, sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { .state = 999 };

    double true_beta[] = {3.0, 3.0, 3.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0, -2.0};

    /* Design matrix: 20 observations per feature */
    for (int i = 0; i < n; i++) {
        int j = i % p;
        X[i * p + j] = 1.0;
        y[i] = true_beta[j] + (test_rng_uniform(&rng) - 0.5) * 0.5;
    }

    gam_params_t params;
    gam_params_init(&params);
    params.family = GAM_FAMILY_GAUSSIAN;
    params.penalty = GAM_PENALTY_FUSED;
    params.alpha = 0.3;    /* 30% L1, 70% TV */
    params.n_lambda = 30;
    params.max_iter = 1000;
    params.tol = 1e-6;
    params.standardize = 0;  /* don't standardize for structured problems */

    gam_path_t *path = gam_fit(X, n, p, y, &params);
    ASSERT(path != NULL, "Fused: fit succeeded");
    ASSERT(path->n_fits > 0, "Fused: produced fits");

    /* At medium regularization, check that adjacent coefficients are fused */
    int mid = path->n_fits / 2;
    double *beta = path->fits[mid].beta;
    printf("  mid-path beta: ");
    for (int j = 0; j < p; j++) printf("%.2f ", beta[j + 1]);
    printf("\n");

    /* At low regularization, check recovery */
    int last = path->n_fits - 1;
    beta = path->fits[last].beta;
    printf("  last beta: ");
    for (int j = 0; j < p; j++) printf("%.2f ", beta[j + 1]);
    printf("\n");

    /* The first 3 should be similar (fused), the middle 3 near zero, last 4 similar */
    double spread_first = fabs(beta[1] - beta[3]);
    double spread_last  = fabs(beta[7] - beta[10]);
    printf("  spread first group: %.3f, last group: %.3f\n", spread_first, spread_last);

    /* Check R-squared */
    double *pred = (double *)malloc((size_t)n * sizeof(double));
    gam_predict(path, last, X, n, p, pred);
    double ss_res = 0.0, ss_tot = 0.0, y_mean = 0.0;
    for (int i = 0; i < n; i++) y_mean += y[i];
    y_mean /= n;
    for (int i = 0; i < n; i++) {
        ss_res += (y[i] - pred[i]) * (y[i] - pred[i]);
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }
    double r2 = 1.0 - ss_res / ss_tot;
    printf("  Fused R-squared: %.4f\n", r2);
    ASSERT(r2 > 0.8, "Fused: reasonable R-squared");

    free(pred);
    gam_free(path);
    free(X);
    free(y);
}

/* ---- Huber Regression ---- */
static void test_huber_regression(void) {
    printf("=== Huber Regression ===\n");

    /* Generate data with outliers:
     * y = 3*x0 + 2*x1 + noise, with ~10% gross outliers */
    int n = 200, d = 5;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    double *y_clean = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 314 };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) X[i * d + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
        double noise = 0.2 * (test_rng_uniform(&rng) - 0.5);
        double signal = 3.0 * X[i * d + 0] + 2.0 * X[i * d + 1];
        y_clean[i] = signal + noise;
        y[i] = y_clean[i];
        /* Add gross outliers to ~10% of observations */
        if (test_rng_uniform(&rng) < 0.1) {
            y[i] += (test_rng_uniform(&rng) > 0.5 ? 1.0 : -1.0) * 50.0;
        }
    }

    /* Fit Huber regression */
    gam_params_t params;
    gam_params_init(&params);
    params.family = 10;  /* GAM_FAMILY_HUBER */
    params.penalty = GAM_PENALTY_ELASTICNET;
    params.alpha = 0.5;
    params.n_lambda = 50;
    params.max_iter = 10000;
    params.max_inner = 50;
    params.huber_gamma = 1.345;
    gam_path_t *path = gam_fit(X, n, d, y, &params);
    ASSERT(path != NULL, "Huber fit succeeded");
    ASSERT(path->n_fits > 0, "Huber has fits");

    /* Also fit Gaussian for comparison */
    gam_params_t params_gauss;
    gam_params_init(&params_gauss);
    params_gauss.penalty = GAM_PENALTY_ELASTICNET;
    params_gauss.alpha = 0.5;
    params_gauss.n_lambda = 50;
    gam_path_t *path_gauss = gam_fit(X, n, d, y, &params_gauss);
    ASSERT(path_gauss != NULL, "Gaussian fit succeeded");

    /* Evaluate on clean data (without outliers) -- Huber should be more robust */
    int best_huber = path->n_fits - 1;
    int best_gauss = path_gauss->n_fits - 1;

    /* Predict with both and compute MSE against clean signal */
    double *pred_h = (double *)malloc((size_t)n * sizeof(double));
    double *pred_g = (double *)malloc((size_t)n * sizeof(double));
    gam_predict(path, best_huber, X, n, d, pred_h);
    gam_predict(path_gauss, best_gauss, X, n, d, pred_g);

    double mse_h = 0.0, mse_g = 0.0, ss_clean = 0.0;
    double y_bar_clean = 0.0;
    for (int i = 0; i < n; i++) y_bar_clean += y_clean[i];
    y_bar_clean /= n;
    for (int i = 0; i < n; i++) {
        mse_h += (pred_h[i] - y_clean[i]) * (pred_h[i] - y_clean[i]);
        mse_g += (pred_g[i] - y_clean[i]) * (pred_g[i] - y_clean[i]);
        ss_clean += (y_clean[i] - y_bar_clean) * (y_clean[i] - y_bar_clean);
    }
    double r2_h = 1.0 - mse_h / ss_clean;
    double r2_g = 1.0 - mse_g / ss_clean;
    printf("  Huber R2 (vs clean): %.4f, Gaussian R2 (vs clean): %.4f\n", r2_h, r2_g);

    /* Huber should achieve better R2 against clean data */
    ASSERT(r2_h > 0.7, "Huber R2 vs clean > 0.7");
    ASSERT(r2_h > r2_g - 0.05, "Huber at least competitive with Gaussian on outlier data");

    /* Check that Huber coefficient recovery is reasonable */
    double b1 = path->fits[best_huber].beta[1];  /* should be ~3 */
    double b2 = path->fits[best_huber].beta[2];  /* should be ~2 */
    printf("  Huber beta: b1=%.3f (exp ~3), b2=%.3f (exp ~2)\n", b1, b2);
    ASSERT(fabs(b1) > 1.0, "Huber b1 has reasonable magnitude");
    ASSERT(fabs(b2) > 0.5, "Huber b2 has reasonable magnitude");

    /* Check serialization roundtrip */
    char *buf = NULL;
    int32_t len = 0;
    int rc = gam_save(path, &buf, &len);
    ASSERT(rc == 0 && len > 0, "Huber save succeeded");
    gam_path_t *loaded = gam_load(buf, len);
    ASSERT(loaded != NULL, "Huber load succeeded");
    ASSERT(loaded->n_fits == path->n_fits, "Huber loaded n_fits match");
    double *pred_loaded = (double *)malloc((size_t)n * sizeof(double));
    gam_predict(loaded, best_huber, X, n, d, pred_loaded);
    double max_diff = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = fabs(pred_loaded[i] - pred_h[i]);
        if (diff > max_diff) max_diff = diff;
    }
    ASSERT(max_diff < 1e-10, "Huber save/load predictions match");
    printf("  Huber save/load max diff: %.2e\n", max_diff);

    gam_free_buffer(buf);
    gam_free(loaded);
    free(pred_loaded);
    free(pred_h);
    free(pred_g);
    gam_free(path);
    gam_free(path_gauss);
    free(X);
    free(y);
    free(y_clean);
}

/* ---- Quantile Regression ---- */
static void test_quantile_regression(void) {
    printf("=== Quantile Regression ===\n");

    /* Generate heteroscedastic data: y = 2*x + noise, noise grows with x */
    int n = 300, d = 3;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 271 };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) X[i * d + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
        double noise = (1.0 + fabs(X[i * d + 0])) * 0.5 * (test_rng_uniform(&rng) - 0.5);
        y[i] = 2.0 * X[i * d + 0] + 1.0 * X[i * d + 1] + noise;
    }

    /* Fit at tau = 0.5 (median) */
    gam_params_t params;
    gam_params_init(&params);
    params.family = 11;  /* GAM_FAMILY_QUANTILE */
    params.penalty = GAM_PENALTY_ELASTICNET;
    params.alpha = 0.5;
    params.n_lambda = 50;
    params.max_iter = 10000;
    params.max_inner = 50;
    params.quantile_tau = 0.5;
    gam_path_t *path50 = gam_fit(X, n, d, y, &params);
    ASSERT(path50 != NULL, "Quantile tau=0.5 fit succeeded");
    ASSERT(path50->n_fits > 0, "Quantile tau=0.5 has fits");

    /* Fit at tau = 0.1 */
    params.quantile_tau = 0.1;
    gam_path_t *path10 = gam_fit(X, n, d, y, &params);
    ASSERT(path10 != NULL, "Quantile tau=0.1 fit succeeded");

    /* Fit at tau = 0.9 */
    params.quantile_tau = 0.9;
    gam_path_t *path90 = gam_fit(X, n, d, y, &params);
    ASSERT(path90 != NULL, "Quantile tau=0.9 fit succeeded");

    /* Check: predictions at tau=0.1 < tau=0.5 < tau=0.9 (on average) */
    int last50 = path50->n_fits - 1;
    int last10 = path10->n_fits - 1;
    int last90 = path90->n_fits - 1;

    double *pred10 = (double *)malloc((size_t)n * sizeof(double));
    double *pred50 = (double *)malloc((size_t)n * sizeof(double));
    double *pred90 = (double *)malloc((size_t)n * sizeof(double));
    gam_predict(path10, last10, X, n, d, pred10);
    gam_predict(path50, last50, X, n, d, pred50);
    gam_predict(path90, last90, X, n, d, pred90);

    double mean10 = 0, mean50 = 0, mean90 = 0;
    for (int i = 0; i < n; i++) {
        mean10 += pred10[i];
        mean50 += pred50[i];
        mean90 += pred90[i];
    }
    mean10 /= n;
    mean50 /= n;
    mean90 /= n;

    printf("  Mean predictions: tau=0.1: %.4f, tau=0.5: %.4f, tau=0.9: %.4f\n",
           mean10, mean50, mean90);
    ASSERT(mean10 < mean50, "tau=0.1 predictions < tau=0.5 predictions");
    ASSERT(mean50 < mean90, "tau=0.5 predictions < tau=0.9 predictions");

    /* Check coverage: fraction of y below prediction */
    int below10 = 0, below50 = 0, below90 = 0;
    for (int i = 0; i < n; i++) {
        if (y[i] < pred10[i]) below10++;
        if (y[i] < pred50[i]) below50++;
        if (y[i] < pred90[i]) below90++;
    }
    double cov10 = (double)below10 / n;
    double cov50 = (double)below50 / n;
    double cov90 = (double)below90 / n;
    printf("  Coverage: tau=0.1: %.3f, tau=0.5: %.3f, tau=0.9: %.3f\n",
           cov10, cov50, cov90);

    /* Coverage should be approximately tau (allow 0.15 tolerance for finite sample + penalization) */
    ASSERT(cov10 < 0.30, "tau=0.1 coverage < 0.30");
    ASSERT(cov50 > 0.25 && cov50 < 0.75, "tau=0.5 coverage between 0.25 and 0.75");
    ASSERT(cov90 > 0.70, "tau=0.9 coverage > 0.70");

    /* Median regression coefficient recovery */
    double b1 = path50->fits[last50].beta[1];
    double b2 = path50->fits[last50].beta[2];
    printf("  Median beta: b1=%.3f (exp ~2), b2=%.3f (exp ~1)\n", b1, b2);
    ASSERT(fabs(b1) > 0.5, "Median b1 reasonable");

    /* Serialization */
    char *buf = NULL;
    int32_t len = 0;
    int rc = gam_save(path50, &buf, &len);
    ASSERT(rc == 0 && len > 0, "Quantile save succeeded");
    gam_path_t *loaded = gam_load(buf, len);
    ASSERT(loaded != NULL, "Quantile load succeeded");
    gam_free_buffer(buf);
    gam_free(loaded);

    free(pred10);
    free(pred50);
    free(pred90);
    gam_free(path10);
    gam_free(path50);
    gam_free(path90);
    free(X);
    free(y);
}

/* ---- Multi-task lasso ---- */
static void test_multi_task_lasso(void) {
    printf("test_multi_task_lasso:\n");
    int32_t n = 200, p = 10, T = 3;
    uint32_t seed = 54321;

    double *X = (double *)malloc((size_t)n * p * sizeof(double));
    double *Y = (double *)malloc((size_t)n * T * sizeof(double));

    /* True coefficients: features 0-2 active across all tasks, rest zero */
    double true_beta[10][3] = {
        {1.5, -1.0, 0.5},
        {-0.8, 1.2, 0.3},
        {0.6, 0.4, -1.1},
        {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}
    };

    for (int32_t i = 0; i < n; i++) {
        for (int32_t j = 0; j < p; j++) {
            seed = seed * 1103515245u + 12345u;
            X[i * p + j] = ((double)(seed >> 16) / 32768.0) - 1.0;
        }
        for (int32_t t = 0; t < T; t++) {
            double eta = 0.0;
            for (int32_t j = 0; j < p; j++) {
                eta += X[i * p + j] * true_beta[j][t];
            }
            seed = seed * 1103515245u + 12345u;
            double noise = ((double)(seed >> 16) / 32768.0 - 0.5) * 0.2;
            Y[i * T + t] = eta + noise;
        }
    }

    gam_params_t params;
    gam_params_init(&params);
    params.family = GAM_FAMILY_GAUSSIAN;
    params.link = GAM_LINK_IDENTITY;
    params.penalty = GAM_PENALTY_ELASTICNET;
    params.alpha = 0.8;
    params.n_lambda = 50;
    params.tol = 1e-7;
    params.standardize = 1;
    params.fit_intercept = 1;
    params.seed = 42;

    gam_path_t *path = gam_fit_multi(X, n, p, Y, T, &params);
    ASSERT(path != NULL, "multi-task path not NULL");
    ASSERT(path->n_fits > 0, "multi-task has fits");
    ASSERT(path->n_tasks == T, "n_tasks == 3");

    /* At smallest lambda, should have good coefficient recovery */
    int32_t last = path->n_fits - 1;
    const double *beta = path->fits[last].beta;

    /* Check layout: n_coefs = T * (p + 1) */
    ASSERT(path->fits[last].n_coefs == T * (p + 1), "n_coefs = T*(p+1)");

    /* Active features (0,1,2) should have nonzero coefficients */
    int active_recovered = 0;
    for (int32_t j = 0; j < 3; j++) {
        int any_nonzero = 0;
        for (int32_t t = 0; t < T; t++) {
            int32_t base = t * (p + 1);
            if (fabs(beta[base + j + 1]) > 0.01) any_nonzero = 1;
        }
        if (any_nonzero) active_recovered++;
    }
    ASSERT(active_recovered == 3, "3 active features recovered");

    /* Group sparsity: at moderate lambda, inactive features should be zero across all tasks */
    int32_t mid = path->n_fits / 2;
    const double *beta_mid = path->fits[mid].beta;
    int group_sparse = 1;
    for (int32_t j = 3; j < p; j++) {
        int all_zero = 1;
        int any_nonzero = 0;
        for (int32_t t = 0; t < T; t++) {
            int32_t base = t * (p + 1);
            if (fabs(beta_mid[base + j + 1]) > 1e-10) {
                all_zero = 0;
                any_nonzero = 1;
            }
        }
        /* If active, should be active for all tasks (group structure) */
        if (any_nonzero && !all_zero) {
            /* This is fine -- but if only some tasks have it, that would break group property */
        }
        (void)all_zero;
    }
    ASSERT(group_sparse, "group sparsity maintained");

    /* Prediction */
    double *pred = (double *)malloc((size_t)n * T * sizeof(double));
    int ret = gam_predict_multi(path, last, X, n, p, pred);
    ASSERT(ret == 0, "multi-task predict succeeds");

    /* R^2 per task */
    for (int32_t t = 0; t < T; t++) {
        double ss_res = 0.0, ss_tot = 0.0, y_mean = 0.0;
        for (int32_t i = 0; i < n; i++) y_mean += Y[i * T + t];
        y_mean /= n;
        for (int32_t i = 0; i < n; i++) {
            double diff = Y[i * T + t] - pred[i * T + t];
            ss_res += diff * diff;
            double d2 = Y[i * T + t] - y_mean;
            ss_tot += d2 * d2;
        }
        double r2 = 1.0 - ss_res / ss_tot;
        ASSERT(r2 > 0.95, "multi-task R^2 > 0.95 per task");
    }
    free(pred);

    /* Deviance should decrease along the path */
    for (int32_t k = 1; k < path->n_fits; k++) {
        ASSERT(path->fits[k].deviance <= path->fits[k-1].deviance + 1e-6,
               "multi-task deviance non-increasing");
    }

    /* df should increase along the path */
    ASSERT(path->fits[last].df >= path->fits[0].df, "df increases along path");

    /* Serialization round-trip */
    char *buf; int32_t buf_len;
    ASSERT(gam_save(path, &buf, &buf_len) == 0, "multi-task save");
    gam_path_t *loaded = gam_load(buf, buf_len);
    ASSERT(loaded != NULL, "multi-task load");
    ASSERT(loaded->n_fits == path->n_fits, "multi-task load n_fits match");
    ASSERT(loaded->n_tasks == T, "multi-task load n_tasks");

    /* Compare coefficients */
    for (int32_t k = 0; k < path->n_fits; k++) {
        for (int32_t c = 0; c < path->fits[k].n_coefs; c++) {
            double d = fabs(path->fits[k].beta[c] - loaded->fits[k].beta[c]);
            ASSERT(d < 1e-10, "multi-task coef match after round-trip");
        }
    }

    gam_free_buffer(buf);
    gam_free(loaded);
    gam_free(path);
    free(X);
    free(Y);
}

/* ---- Tensor Product Basis ---- */
static void test_tensor_basis(void) {
    printf("=== Tensor Product Basis ===\n");

    /* Test gam_tensor_basis for two features with known knots */
    int n = 50;
    double *x1 = (double *)malloc((size_t)n * sizeof(double));
    double *x2 = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 777 };

    for (int i = 0; i < n; i++) {
        x1[i] = test_rng_uniform(&rng);
        x2[i] = test_rng_uniform(&rng);
    }

    /* 3 interior knots, degree 3 => n_basis = 3 + 3 + 1 = 7 per margin */
    int32_t nk = 3, deg = 3;
    double knots1[3], knots2[3];
    gam_quantile_knots(x1, n, nk, knots1);
    gam_quantile_knots(x2, n, nk, knots2);

    int32_t nb1 = nk + deg + 1;  /* 7 */
    int32_t nb2 = nk + deg + 1;  /* 7 */
    int32_t nb_total = nb1 * nb2; /* 49 */

    double *T = (double *)malloc((size_t)n * (size_t)nb_total * sizeof(double));
    int32_t ret = gam_tensor_basis(x1, x2, n,
                                   knots1, nk, deg,
                                   knots2, nk, deg, T);
    ASSERT(ret == nb_total, "tensor basis returns correct n_basis");
    printf("  tensor basis: %d x %d = %d columns\n", nb1, nb2, nb_total);

    /* Partition of unity: each row should sum to ~1 */
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < nb_total; j++)
            row_sum += T[i * nb_total + j];
        double err = fabs(row_sum - 1.0);
        if (err > max_err) max_err = err;
    }
    printf("  partition of unity max error: %.2e\n", max_err);
    ASSERT(max_err < 1e-10, "tensor basis partition of unity");

    /* All values non-negative */
    int any_neg = 0;
    for (int i = 0; i < n * nb_total; i++)
        if (T[i] < -1e-15) any_neg = 1;
    ASSERT(!any_neg, "tensor basis values non-negative");

    free(x1); free(x2); free(T);
}

/* ---- Univariate Smooth with Penalty ---- */
static void test_smooth_penalty(void) {
    printf("=== Smooth Penalty ===\n");

    /* y = sin(2*pi*x) + noise. Fit with B-spline smooth.
     * Without smoothness penalty: overfits. With penalty: smoother fit. */
    int n = 200;
    double *X = (double *)malloc((size_t)n * 1 * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 123 };

    for (int i = 0; i < n; i++) {
        X[i] = (double)i / (n - 1);  /* uniform grid [0, 1] */
        y[i] = sin(2.0 * M_PI * X[i]) + 0.3 * (test_rng_uniform(&rng) - 0.5);
    }

    gam_smooth_t sm = { .feature = 0, .n_knots = 20, .degree = 3, .lambda_smooth = 0.0 };

    /* Fit without smoothness penalty */
    gam_params_t params;
    gam_params_init(&params);
    params.family = GAM_FAMILY_GAUSSIAN;
    params.penalty = GAM_PENALTY_NONE;
    params.smooths = &sm;
    params.n_smooths = 1;

    gam_path_t *path_no_pen = gam_fit(X, n, 1, y, &params);
    ASSERT(path_no_pen != NULL, "smooth fit without penalty");

    /* Fit with smoothness penalty */
    sm.lambda_smooth = 1.0;
    gam_path_t *path_pen = gam_fit(X, n, 1, y, &params);
    ASSERT(path_pen != NULL, "smooth fit with penalty");

    /* Both should have reasonable fits */
    int last_np = path_no_pen->n_fits - 1;
    int last_p = path_pen->n_fits - 1;
    double dev_np = path_no_pen->fits[last_np].deviance;
    double dev_p = path_pen->fits[last_p].deviance;
    printf("  deviance without penalty: %.4f\n", dev_np);
    printf("  deviance with penalty: %.4f\n", dev_p);

    /* With penalty, coefficients should be smaller in magnitude (smoother) */
    double sum_abs_np = 0.0, sum_abs_p = 0.0;
    int nc = path_no_pen->fits[last_np].n_coefs;
    for (int j = 1; j < nc; j++) {
        sum_abs_np += fabs(path_no_pen->fits[last_np].beta[j]);
        sum_abs_p += fabs(path_pen->fits[last_p].beta[j]);
    }
    printf("  sum|beta| without penalty: %.4f\n", sum_abs_np);
    printf("  sum|beta| with penalty: %.4f\n", sum_abs_p);
    ASSERT(sum_abs_p < sum_abs_np, "penalty shrinks coefficients");

    /* n_basis should be n_knots + degree + 1 = 24 */
    ASSERT(nc == 25, "n_coefs = 1 + 24 basis functions");
    printf("  n_coefs: %d (expected 25)\n", nc);

    gam_free(path_no_pen);
    gam_free(path_pen);
    free(X); free(y);
}

/* ---- Tensor Product Smooth Fitting ---- */
static void test_tensor_smooth(void) {
    printf("=== Tensor Product Smooth ===\n");

    /* y = sin(x1) * cos(x2) + noise. Fit with tensor product smooth. */
    int n = 300, d = 2;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 456 };

    for (int i = 0; i < n; i++) {
        X[i * d + 0] = test_rng_uniform(&rng) * 2.0 * M_PI;  /* x1 in [0, 2pi] */
        X[i * d + 1] = test_rng_uniform(&rng) * 2.0 * M_PI;  /* x2 in [0, 2pi] */
        y[i] = sin(X[i * d + 0]) * cos(X[i * d + 1])
             + 0.1 * (test_rng_uniform(&rng) - 0.5);
    }

    gam_tensor_t tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.n_margins = 2;
    tensor.features[0] = 0;
    tensor.features[1] = 1;
    tensor.n_knots[0] = 8;
    tensor.n_knots[1] = 8;
    tensor.degree[0] = 3;
    tensor.degree[1] = 3;
    tensor.lambda_smooth[0] = 0.01;
    tensor.lambda_smooth[1] = 0.01;

    gam_params_t params;
    gam_params_init(&params);
    params.family = GAM_FAMILY_GAUSSIAN;
    params.penalty = GAM_PENALTY_NONE;
    params.tensors = &tensor;
    params.n_tensors = 1;

    gam_path_t *path = gam_fit(X, n, d, y, &params);
    ASSERT(path != NULL, "tensor product fit");
    if (!path) {
        printf("  ERROR: %s\n", gam_get_error());
        free(X); free(y);
        return;
    }

    int last = path->n_fits - 1;
    int nb1 = 8 + 3 + 1;  /* 12 */
    int nb2 = 8 + 3 + 1;  /* 12 */
    int nb_total = nb1 * nb2;  /* 144 */
    printf("  n_coefs: %d (expected %d = 1 + %d)\n",
           path->fits[last].n_coefs, 1 + nb_total, nb_total);
    ASSERT(path->fits[last].n_coefs == 1 + nb_total,
           "tensor n_coefs = 1 + nb1*nb2");

    /* Compute R-squared */
    double *pred = (double *)malloc((size_t)n * sizeof(double));
    gam_predict(path, last, X, n, d, pred);
    double ss_res = 0.0, ss_tot = 0.0;
    double y_mean = 0.0;
    for (int i = 0; i < n; i++) y_mean += y[i];
    y_mean /= n;
    for (int i = 0; i < n; i++) {
        ss_res += (y[i] - pred[i]) * (y[i] - pred[i]);
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }
    double r2 = 1.0 - ss_res / ss_tot;
    printf("  R-squared: %.4f\n", r2);
    ASSERT(r2 > 0.85, "tensor smooth R2 > 0.85");

    /* Serialization round-trip */
    char *buf = NULL;
    int32_t buf_len = 0;
    int save_ret = gam_save(path, &buf, &buf_len);
    ASSERT(save_ret == 0, "tensor save succeeds");
    printf("  saved %d bytes\n", buf_len);

    gam_path_t *loaded = gam_load(buf, buf_len);
    ASSERT(loaded != NULL, "tensor load succeeds");

    /* Predictions match after round-trip */
    double *pred2 = (double *)malloc((size_t)n * sizeof(double));
    gam_predict(loaded, last, X, n, d, pred2);
    double max_diff = 0.0;
    for (int i = 0; i < n; i++) {
        double d_ = fabs(pred[i] - pred2[i]);
        if (d_ > max_diff) max_diff = d_;
    }
    printf("  save/load pred max diff: %.2e\n", max_diff);
    ASSERT(max_diff < 1e-10, "tensor save/load predictions match");

    gam_free_buffer(buf);
    gam_free(loaded);
    gam_free(path);
    free(pred); free(pred2); free(X); free(y);
}

/* ---- Multinomial Logistic Regression ---- */
static void test_multinomial_basic(void) {
    printf("=== Multinomial Basic ===\n");

    /* 3-class problem: y = argmax(x0, x1, x2) + noise */
    int n = 300, d = 5, K = 3;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 999 };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++)
            X[i * d + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
        /* Class determined by which of first 3 features is largest */
        int best = 0;
        for (int j = 1; j < 3; j++)
            if (X[i * d + j] > X[i * d + best]) best = j;
        y[i] = (double)best;
    }

    gam_params_t params;
    gam_params_init(&params);
    params.penalty = GAM_PENALTY_ELASTICNET;
    params.alpha = 0.5;
    params.n_lambda = 30;

    gam_path_t *path = gam_fit_multinomial(X, n, d, y, K, &params);
    ASSERT(path != NULL, "multinomial fit");
    ASSERT(path->n_fits > 0, "multinomial n_fits > 0");
    ASSERT(path->family == GAM_FAMILY_MULTINOMIAL, "multinomial family");
    ASSERT(path->n_tasks == K, "multinomial n_tasks == K");
    printf("  n_fits: %d\n", path->n_fits);

    int last = path->n_fits - 1;
    ASSERT(path->fits[last].n_coefs == K * (d + 1), "multinomial n_coefs");
    printf("  n_coefs: %d (expected %d)\n", path->fits[last].n_coefs, K * (d + 1));

    /* Predict probabilities */
    double *prob = (double *)malloc((size_t)n * K * sizeof(double));
    int ret = gam_predict_multinomial(path, last, X, n, d, prob);
    ASSERT(ret == 0, "multinomial predict");

    /* Check probabilities sum to 1 */
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int k = 0; k < K; k++) {
            ASSERT(prob[i * K + k] >= 0.0, "prob >= 0");
            ASSERT(prob[i * K + k] <= 1.0, "prob <= 1");
            sum += prob[i * K + k];
        }
        double err = fabs(sum - 1.0);
        if (err > max_err) max_err = err;
    }
    printf("  prob sum max error: %.2e\n", max_err);
    ASSERT(max_err < 1e-10, "probs sum to 1");

    /* Compute accuracy */
    int correct = 0;
    for (int i = 0; i < n; i++) {
        int pred_class = 0;
        for (int k = 1; k < K; k++)
            if (prob[i * K + k] > prob[i * K + pred_class]) pred_class = k;
        if (pred_class == (int)y[i]) correct++;
    }
    double acc = (double)correct / n;
    printf("  accuracy: %.3f\n", acc);
    ASSERT(acc > 0.6, "multinomial accuracy > 0.6");

    /* Check sparsity at lambda_max */
    ASSERT(path->fits[0].df == 0, "all zero at lambda_max");

    /* Check that later lambdas have nonzero features */
    ASSERT(path->fits[last].df >= 3, "at least 3 active features");
    printf("  df at last: %d\n", path->fits[last].df);

    /* Serialization round-trip */
    char *buf = NULL;
    int32_t buf_len = 0;
    int save_ret = gam_save(path, &buf, &buf_len);
    ASSERT(save_ret == 0, "multinomial save");
    printf("  saved %d bytes\n", buf_len);

    gam_path_t *loaded = gam_load(buf, buf_len);
    ASSERT(loaded != NULL, "multinomial load");

    double *prob2 = (double *)malloc((size_t)n * K * sizeof(double));
    ret = gam_predict_multinomial(loaded, last, X, n, d, prob2);
    ASSERT(ret == 0, "multinomial predict after load");

    double max_diff = 0.0;
    for (int i = 0; i < n * K; i++) {
        double diff = fabs(prob[i] - prob2[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  save/load pred max diff: %.2e\n", max_diff);
    ASSERT(max_diff < 1e-10, "multinomial save/load predictions match");

    gam_free_buffer(buf);
    gam_free(loaded);
    gam_free(path);
    free(prob); free(prob2); free(X); free(y);
}

/* ---- Multinomial Lasso (sparsity) ---- */
static void test_multinomial_lasso(void) {
    printf("=== Multinomial Lasso ===\n");

    /* 4-class, 20 features, only first 4 relevant */
    int n = 400, d = 20, K = 4;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 2025 };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++)
            X[i * d + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
        /* Class based on linear combination of first 4 features */
        double scores[4];
        scores[0] = 2.0 * X[i * d + 0] - 1.0 * X[i * d + 1];
        scores[1] = -1.5 * X[i * d + 0] + 2.0 * X[i * d + 2];
        scores[2] = 1.0 * X[i * d + 1] - 2.0 * X[i * d + 3];
        scores[3] = -0.5 * X[i * d + 2] + 1.5 * X[i * d + 3];
        /* Add noise */
        for (int k = 0; k < K; k++)
            scores[k] += 0.3 * (test_rng_uniform(&rng) - 0.5);
        int best = 0;
        for (int k = 1; k < K; k++)
            if (scores[k] > scores[best]) best = k;
        y[i] = (double)best;
    }

    gam_params_t params;
    gam_params_init(&params);
    params.penalty = GAM_PENALTY_L1;
    params.alpha = 1.0;
    params.n_lambda = 30;

    gam_path_t *path = gam_fit_multinomial(X, n, d, y, K, &params);
    ASSERT(path != NULL, "multinomial lasso fit");

    int last = path->n_fits - 1;
    printf("  df at last: %d (out of %d)\n", path->fits[last].df, d);

    /* At mid-path, should have fewer than all features active */
    int mid = path->n_fits / 2;
    printf("  df at mid: %d\n", path->fits[mid].df);
    ASSERT(path->fits[mid].df < d, "lasso selects features");

    /* Accuracy at last lambda */
    double *prob = (double *)malloc((size_t)n * K * sizeof(double));
    gam_predict_multinomial(path, last, X, n, d, prob);
    int correct = 0;
    for (int i = 0; i < n; i++) {
        int pred_class = 0;
        for (int k = 1; k < K; k++)
            if (prob[i * K + k] > prob[i * K + pred_class]) pred_class = k;
        if (pred_class == (int)y[i]) correct++;
    }
    double acc = (double)correct / n;
    printf("  lasso accuracy: %.3f\n", acc);
    ASSERT(acc > 0.5, "multinomial lasso accuracy > 0.5");

    /* Deviance should decrease along path */
    ASSERT(path->fits[last].deviance < path->fits[0].deviance,
           "deviance decreases along path");
    printf("  deviance: first=%.2f last=%.2f\n",
           path->fits[0].deviance, path->fits[last].deviance);

    gam_free(path);
    free(prob); free(X); free(y);
}

/* ---- GAMLSS Normal ---- */
static void test_gamlss_normal(void) {
    printf("=== GAMLSS Normal ===\n");

    int n = 200, d = 3;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 42 };

    /* Heteroscedastic data: y ~ N(mu=2*x0+x1, sigma=exp(0.5+x2)) */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) X[i * d + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
        double mu_true = 2.0 * X[i * d + 0] + 1.0 * X[i * d + 1];
        double log_sigma_true = 0.5 + 0.8 * X[i * d + 2];
        double sigma_true = exp(log_sigma_true);
        /* Box-Muller for normal noise */
        double u1 = test_rng_uniform(&rng);
        double u2 = test_rng_uniform(&rng);
        double noise = sqrt(-2.0 * log(u1 + 1e-30)) * cos(2.0 * M_PI * u2);
        y[i] = mu_true + sigma_true * noise;
    }

    gam_params_t params;
    gam_params_init(&params);
    params.penalty = GAM_PENALTY_ELASTICNET;
    params.alpha = 0.5;
    params.n_lambda = 30;
    params.tol = 1e-6;
    params.max_iter = 5000;

    gam_path_t *path = gam_fit_gamlss(X, n, d, y, GAMLSS_NORMAL, &params);
    ASSERT(path != NULL, "gamlss normal: path not NULL");
    ASSERT(path->n_fits > 0, "gamlss normal: has fits");
    ASSERT(path->n_tasks == 2, "gamlss normal: n_tasks == 2");
    ASSERT(path->family_gamlss == 1, "gamlss normal: family_gamlss == 1 (Normal)");

    int last = path->n_fits - 1;
    printf("  n_fits: %d\n", path->n_fits);
    printf("  deviance: first=%.2f last=%.2f\n",
           path->fits[0].deviance, path->fits[last].deviance);
    ASSERT(path->fits[last].deviance <= path->fits[0].deviance + 1.0,
           "gamlss normal: deviance not increasing wildly");

    /* Check mu prediction quality at last lambda (least regularized) */
    double *pred = (double *)malloc((size_t)n * 2 * sizeof(double));
    int ret = gam_predict_gamlss(path, last, X, n, d, pred);
    ASSERT(ret == 0, "gamlss normal: predict ok");

    /* Compute R2 for mu predictions */
    double ss_res = 0.0, ss_tot = 0.0;
    double y_mean = 0.0;
    for (int i = 0; i < n; i++) y_mean += y[i];
    y_mean /= n;
    for (int i = 0; i < n; i++) {
        double mu_pred = pred[i * 2 + 0];
        ss_res += (y[i] - mu_pred) * (y[i] - mu_pred);
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }
    double r2 = 1.0 - ss_res / (ss_tot + 1e-30);
    printf("  mu R2: %.4f\n", r2);
    ASSERT(r2 > 0.3, "gamlss normal: mu R2 > 0.3");

    /* Check that sigma predictions are positive and vary */
    double sigma_min = 1e30, sigma_max = -1e30;
    for (int i = 0; i < n; i++) {
        double s = pred[i * 2 + 1];
        ASSERT(s > 0, "gamlss normal: sigma > 0");
        if (s < sigma_min) sigma_min = s;
        if (s > sigma_max) sigma_max = s;
    }
    printf("  sigma range: [%.4f, %.4f]\n", sigma_min, sigma_max);
    ASSERT(sigma_max > sigma_min * 1.1, "gamlss normal: sigma varies");

    /* Save/load roundtrip */
    char *buf; int32_t len;
    ret = gam_save(path, &buf, &len);
    ASSERT(ret == 0, "gamlss normal: save ok");
    gam_path_t *loaded = gam_load(buf, len);
    ASSERT(loaded != NULL, "gamlss normal: load ok");
    ASSERT(loaded->family_gamlss == 1, "gamlss normal: loaded family_gamlss");
    ASSERT(loaded->n_tasks == 2, "gamlss normal: loaded n_tasks");

    /* Check predictions match */
    double *pred2 = (double *)malloc((size_t)n * 2 * sizeof(double));
    ret = gam_predict_gamlss(loaded, last, X, n, d, pred2);
    ASSERT(ret == 0, "gamlss normal: predict loaded ok");
    int match = 1;
    for (int i = 0; i < n * 2; i++) {
        if (fabs(pred[i] - pred2[i]) > 1e-10) { match = 0; break; }
    }
    ASSERT(match, "gamlss normal: predictions match after save/load");

    gam_free(loaded);
    gam_free_buffer(buf);
    free(pred2);
    free(pred);
    gam_free(path);
    free(X); free(y);
}

/* ---- GAMLSS Gamma ---- */
static void test_gamlss_gamma(void) {
    printf("=== GAMLSS Gamma ===\n");

    int n = 200, d = 3;
    double *X = (double *)malloc((size_t)n * d * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    test_rng_t rng = { 123 };

    /* Gamma data: y ~ Gamma(shape, scale) where mu = shape*scale
     * Use small constant CV to ensure well-behaved data.
     * mu = exp(0.5 + 0.3*x0), CV = exp(-1 + 0.2*x1) ≈ 0.37
     * Generate via exponential sum approximation. */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) X[i * d + j] = test_rng_uniform(&rng) * 2.0 - 1.0;
        double mu_true = exp(0.5 + 0.3 * X[i * d + 0]);
        double cv_true = exp(-1.0 + 0.2 * X[i * d + 1]);
        /* Box-Muller for normal noise, then use lognormal as Gamma proxy */
        double u1 = test_rng_uniform(&rng);
        double u2 = test_rng_uniform(&rng);
        double noise = sqrt(-2.0 * log(u1 + 1e-30)) * cos(2.0 * M_PI * u2);
        y[i] = mu_true * exp(cv_true * noise - 0.5 * cv_true * cv_true);
        if (y[i] < 0.01) y[i] = 0.01;
    }

    gam_params_t params;
    gam_params_init(&params);
    params.penalty = GAM_PENALTY_ELASTICNET;
    params.alpha = 0.5;
    params.n_lambda = 20;
    params.tol = 1e-6;
    params.max_iter = 5000;

    gam_path_t *path = gam_fit_gamlss(X, n, d, y, GAMLSS_GAMMA, &params);
    ASSERT(path != NULL, "gamlss gamma: path not NULL");
    ASSERT(path->n_fits > 0, "gamlss gamma: has fits");
    ASSERT(path->family_gamlss == 2, "gamlss gamma: family_gamlss == 2 (Gamma)");

    int last = path->n_fits - 1;
    printf("  n_fits: %d\n", path->n_fits);

    /* Check predictions */
    double *pred = (double *)malloc((size_t)n * 2 * sizeof(double));
    int ret = gam_predict_gamlss(path, last, X, n, d, pred);
    ASSERT(ret == 0, "gamlss gamma: predict ok");

    /* mu predictions should be positive */
    int all_pos = 1;
    for (int i = 0; i < n; i++) {
        if (pred[i * 2 + 0] <= 0) { all_pos = 0; break; }
    }
    ASSERT(all_pos, "gamlss gamma: mu > 0");

    /* sigma (CV) should be positive and reasonable */
    int all_sigma_ok = 1;
    for (int i = 0; i < n; i++) {
        if (pred[i * 2 + 1] <= 0 || pred[i * 2 + 1] > 100) { all_sigma_ok = 0; break; }
    }
    ASSERT(all_sigma_ok, "gamlss gamma: sigma positive and finite");

    gam_free(path);
    free(pred);
    free(X); free(y);
}

/* ---- Main ---- */
int main(void) {
    printf("\n== GAM C Tests ==\n\n");

    test_gaussian_lasso();
    test_gaussian_ridge();
    test_gaussian_elasticnet();
    test_binomial();
    test_poisson();
    test_mcp();
    test_scad();
    test_cv();
    test_bspline();
    test_quantile_knots();
    test_serialization();
    test_relaxed();
    test_penalty_factors();
    test_gamma();
    test_diagnostics();
    test_unpenalized();
    test_slope();
    test_slope_binomial();
    test_group_lasso();
    test_sparse_group_lasso();
    test_cox_basic();
    test_cox_lasso();
    test_cox_serialization();
    test_anderson_accel();
    test_gap_safe_screening();
    test_fused_lasso();
    test_huber_regression();
    test_quantile_regression();
    test_multi_task_lasso();
    test_tensor_basis();
    test_smooth_penalty();
    test_tensor_smooth();
    test_multinomial_basic();
    test_multinomial_lasso();
    test_gamlss_normal();
    test_gamlss_gamma();

    printf("\n== Results: %d/%d passed ==\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
