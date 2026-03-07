/*
 * wl_api.c -- C ABI wrapper for GAM (WASM/FFI boundary)
 *
 * All params as primitives (no struct passing across ABI).
 */

#include "gam.h"
#include <stdlib.h>

const char *wl_gam_get_last_error(void) {
    return gam_get_error();
}

/* Fit GLM (no smooth terms) */
gam_path_t *wl_gam_fit(
    const double *X, int nrow, int ncol,
    const double *y,
    int family, int link, int penalty,
    double alpha, int n_lambda, double lambda_min_ratio,
    double gamma_mcp, double gamma_scad,
    double tol, int max_iter, int max_inner, int screening,
    int n_folds, int standardize, int fit_intercept, int relax,
    double tweedie_power, double neg_binom_theta,
    double slope_q,
    int seed,
    double huber_gamma, double quantile_tau
) {
    gam_params_t params;
    gam_params_init(&params);
    params.family = family;
    params.link = link;
    params.penalty = penalty;
    params.alpha = alpha;
    params.n_lambda = n_lambda;
    params.lambda_min_ratio = lambda_min_ratio;
    params.gamma_mcp = gamma_mcp;
    params.gamma_scad = gamma_scad;
    params.tol = tol;
    params.max_iter = max_iter;
    params.max_inner = max_inner;
    params.screening = screening;
    params.n_folds = n_folds;
    params.standardize = standardize;
    params.fit_intercept = fit_intercept;
    params.relax = relax;
    params.tweedie_power = tweedie_power;
    params.neg_binom_theta = neg_binom_theta;
    params.slope_q = slope_q > 0.0 ? slope_q : 0.1;
    params.huber_gamma = huber_gamma > 0.0 ? huber_gamma : 1.345;
    params.quantile_tau = (quantile_tau > 0.0 && quantile_tau < 1.0) ? quantile_tau : 0.5;
    params.seed = (uint32_t)seed;
    return gam_fit(X, (int32_t)nrow, (int32_t)ncol, y, &params);
}

/* Fit with user-supplied lambda sequence */
gam_path_t *wl_gam_fit_lambda(
    const double *X, int nrow, int ncol,
    const double *y,
    int family, int link, int penalty,
    double alpha,
    const double *lambda_seq, int n_lambda,
    double tol, int max_iter, int max_inner,
    int standardize, int fit_intercept,
    int seed
) {
    gam_params_t params;
    gam_params_init(&params);
    params.family = family;
    params.link = link;
    params.penalty = penalty;
    params.alpha = alpha;
    params.lambda = (double *)lambda_seq;
    params.n_lambda_user = n_lambda;
    params.n_lambda = n_lambda;
    params.tol = tol;
    params.max_iter = max_iter;
    params.max_inner = max_inner;
    params.standardize = standardize;
    params.fit_intercept = fit_intercept;
    params.seed = (uint32_t)seed;
    return gam_fit(X, (int32_t)nrow, (int32_t)ncol, y, &params);
}

/* Fit with penalty factors */
gam_path_t *wl_gam_fit_pf(
    const double *X, int nrow, int ncol,
    const double *y,
    int family, int link, int penalty,
    double alpha, int n_lambda, double lambda_min_ratio,
    const double *penalty_factor,
    double tol, int max_iter,
    int standardize, int fit_intercept,
    int seed
) {
    gam_params_t params;
    gam_params_init(&params);
    params.family = family;
    params.link = link;
    params.penalty = penalty;
    params.alpha = alpha;
    params.n_lambda = n_lambda;
    params.lambda_min_ratio = lambda_min_ratio;
    params.penalty_factor = (double *)penalty_factor;
    params.tol = tol;
    params.max_iter = max_iter;
    params.standardize = standardize;
    params.fit_intercept = fit_intercept;
    params.seed = (uint32_t)seed;
    return gam_fit(X, (int32_t)nrow, (int32_t)ncol, y, &params);
}

/* Fit with group structure (group lasso / sparse group lasso) */
gam_path_t *wl_gam_fit_groups(
    const double *X, int nrow, int ncol,
    const double *y,
    int family, int link, int penalty,
    double alpha, int n_lambda, double lambda_min_ratio,
    const int *groups, int n_groups,
    double tol, int max_iter,
    int standardize, int fit_intercept,
    int seed
) {
    gam_params_t params;
    gam_params_init(&params);
    params.family = family;
    params.link = link;
    params.penalty = penalty;
    params.alpha = alpha;
    params.n_lambda = n_lambda;
    params.lambda_min_ratio = lambda_min_ratio;
    params.groups = (int32_t *)groups;
    params.n_groups = (int32_t)n_groups;
    params.tol = tol;
    params.max_iter = max_iter;
    params.standardize = standardize;
    params.fit_intercept = fit_intercept;
    params.seed = (uint32_t)seed;
    return gam_fit(X, (int32_t)nrow, (int32_t)ncol, y, &params);
}

/* Fit Cox PH model */
gam_path_t *wl_gam_fit_cox(
    const double *X, int nrow, int ncol,
    const double *time, const double *status,
    int penalty, double alpha, int n_lambda, double lambda_min_ratio,
    double gamma_mcp, double gamma_scad,
    double tol, int max_iter,
    int standardize,
    int seed
) {
    gam_params_t params;
    gam_params_init(&params);
    params.family = GAM_FAMILY_COX;
    params.link = GAM_LINK_LOG;
    params.penalty = penalty;
    params.alpha = alpha;
    params.n_lambda = n_lambda;
    params.lambda_min_ratio = lambda_min_ratio;
    params.gamma_mcp = gamma_mcp;
    params.gamma_scad = gamma_scad;
    params.tol = tol;
    params.max_iter = max_iter;
    params.standardize = standardize;
    params.fit_intercept = 0;  /* Cox has no intercept */
    params.seed = (uint32_t)seed;
    return gam_fit_cox(X, (int32_t)nrow, (int32_t)ncol, time, status, &params);
}

/* Fit multi-task (multi-response) elastic net */
gam_path_t *wl_gam_fit_multi(
    const double *X, int nrow, int ncol,
    const double *Y, int n_tasks,
    int penalty, double alpha, int n_lambda, double lambda_min_ratio,
    double tol, int max_iter,
    int standardize, int fit_intercept,
    int seed
) {
    gam_params_t params;
    gam_params_init(&params);
    params.family = GAM_FAMILY_GAUSSIAN;
    params.link = GAM_LINK_IDENTITY;
    params.penalty = penalty;
    params.alpha = alpha;
    params.n_lambda = n_lambda;
    params.lambda_min_ratio = lambda_min_ratio;
    params.tol = tol;
    params.max_iter = max_iter;
    params.standardize = standardize;
    params.fit_intercept = fit_intercept;
    params.seed = (uint32_t)seed;
    return gam_fit_multi(X, (int32_t)nrow, (int32_t)ncol, Y, (int32_t)n_tasks, &params);
}

int wl_gam_predict_multi(const gam_path_t *path, int fit_idx,
                          const double *X, int nrow, int ncol, double *out) {
    return gam_predict_multi(path, (int32_t)fit_idx, X, (int32_t)nrow, (int32_t)ncol, out);
}

int wl_gam_get_n_tasks(const gam_path_t *path) {
    return path ? path->n_tasks : 0;
}

/* Multinomial logistic regression */
gam_path_t *wl_gam_fit_multinomial(
    const double *X, int nrow, int ncol,
    const double *y, int n_classes,
    int penalty, double alpha, int n_lambda, double lambda_min_ratio,
    double tol, int max_iter, int max_inner,
    int standardize, int fit_intercept,
    int seed
) {
    gam_params_t params;
    gam_params_init(&params);
    params.penalty = penalty;
    params.alpha = alpha;
    params.n_lambda = n_lambda;
    params.lambda_min_ratio = lambda_min_ratio;
    params.tol = tol;
    params.max_iter = max_iter;
    params.max_inner = max_inner;
    params.standardize = standardize;
    params.fit_intercept = fit_intercept;
    params.seed = (uint32_t)seed;
    return gam_fit_multinomial(X, (int32_t)nrow, (int32_t)ncol, y,
                               (int32_t)n_classes, &params);
}

int wl_gam_predict_multinomial(const gam_path_t *path, int fit_idx,
                                const double *X, int nrow, int ncol, double *out) {
    return gam_predict_multinomial(path, (int32_t)fit_idx, X,
                                   (int32_t)nrow, (int32_t)ncol, out);
}

/* GAMLSS (distributional regression) */
gam_path_t *wl_gam_fit_gamlss(
    const double *X, int nrow, int ncol,
    const double *y, int distribution,
    int penalty, double alpha, int n_lambda, double lambda_min_ratio,
    double tol, int max_iter,
    int standardize, int fit_intercept,
    int seed
) {
    gam_params_t params;
    gam_params_init(&params);
    params.penalty = penalty;
    params.alpha = alpha;
    params.n_lambda = n_lambda;
    params.lambda_min_ratio = lambda_min_ratio;
    params.tol = tol;
    params.max_iter = max_iter;
    params.standardize = standardize;
    params.fit_intercept = fit_intercept;
    params.seed = (uint32_t)seed;
    return gam_fit_gamlss(X, (int32_t)nrow, (int32_t)ncol, y,
                           (int32_t)distribution, &params);
}

int wl_gam_predict_gamlss(const gam_path_t *path, int fit_idx,
                            const double *X, int nrow, int ncol, double *out) {
    return gam_predict_gamlss(path, (int32_t)fit_idx, X,
                               (int32_t)nrow, (int32_t)ncol, out);
}

int wl_gam_get_family_gamlss(const gam_path_t *path) {
    return path ? path->family_gamlss : 0;
}

int wl_gam_predict(const gam_path_t *path, int fit_idx,
                    const double *X, int nrow, int ncol, double *out) {
    return gam_predict(path, (int32_t)fit_idx, X, (int32_t)nrow, (int32_t)ncol, out);
}

int wl_gam_predict_eta(const gam_path_t *path, int fit_idx,
                        const double *X, int nrow, int ncol, double *out) {
    return gam_predict_eta(path, (int32_t)fit_idx, X, (int32_t)nrow, (int32_t)ncol, out);
}

int wl_gam_predict_proba(const gam_path_t *path, int fit_idx,
                          const double *X, int nrow, int ncol, double *out) {
    return gam_predict_proba(path, (int32_t)fit_idx, X, (int32_t)nrow, (int32_t)ncol, out);
}

int wl_gam_get_n_fits(const gam_path_t *path) {
    return path ? path->n_fits : 0;
}

int wl_gam_get_n_features(const gam_path_t *path) {
    return path ? path->n_features : 0;
}

int wl_gam_get_n_coefs(const gam_path_t *path) {
    return path ? path->n_coefs : 0;
}

int wl_gam_get_family(const gam_path_t *path) {
    return path ? path->family : -1;
}

int wl_gam_get_idx_min(const gam_path_t *path) {
    return path ? path->idx_min : -1;
}

int wl_gam_get_idx_1se(const gam_path_t *path) {
    return path ? path->idx_1se : -1;
}

double wl_gam_get_lambda(const gam_path_t *path, int fit_idx) {
    if (!path || fit_idx < 0 || fit_idx >= path->n_fits) return 0.0;
    return path->fits[fit_idx].lambda;
}

double wl_gam_get_deviance(const gam_path_t *path, int fit_idx) {
    if (!path || fit_idx < 0 || fit_idx >= path->n_fits) return 0.0;
    return path->fits[fit_idx].deviance;
}

int wl_gam_get_df(const gam_path_t *path, int fit_idx) {
    if (!path || fit_idx < 0 || fit_idx >= path->n_fits) return 0;
    return path->fits[fit_idx].df;
}

double wl_gam_get_cv_mean(const gam_path_t *path, int fit_idx) {
    if (!path || fit_idx < 0 || fit_idx >= path->n_fits) return 0.0;
    return path->fits[fit_idx].cv_mean;
}

double wl_gam_get_cv_se(const gam_path_t *path, int fit_idx) {
    if (!path || fit_idx < 0 || fit_idx >= path->n_fits) return 0.0;
    return path->fits[fit_idx].cv_se;
}

/* Get coefficient for a specific fit and feature index (0 = intercept) */
double wl_gam_get_coef(const gam_path_t *path, int fit_idx, int coef_idx) {
    if (!path || fit_idx < 0 || fit_idx >= path->n_fits) return 0.0;
    if (coef_idx < 0 || coef_idx >= path->fits[fit_idx].n_coefs) return 0.0;
    return path->fits[fit_idx].beta[coef_idx];
}

int wl_gam_save(const gam_path_t *path, char **out_buf, int *out_len) {
    int32_t len32;
    int ret = gam_save(path, out_buf, &len32);
    if (ret == 0 && out_len) *out_len = (int)len32;
    return ret;
}

gam_path_t *wl_gam_load(const char *buf, int len) {
    return gam_load(buf, (int32_t)len);
}

void wl_gam_free(gam_path_t *path) {
    gam_free(path);
}

void wl_gam_free_buffer(void *ptr) {
    gam_free_buffer(ptr);
}

/* B-spline utility for JS */
int wl_gam_bspline_basis(
    const double *x, int n,
    const double *knots, int n_knots, int degree,
    double *out
) {
    return gam_bspline_basis(x, (int32_t)n, knots, (int32_t)n_knots, (int32_t)degree, out);
}
