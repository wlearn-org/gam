/*
 * gam.h -- Generalized Additive Models (C11, from scratch)
 *
 * Covers the full GLM/GAM/GLMM hierarchy:
 *   - GLM: penalized regression (elastic net, lasso, ridge, MCP, SCAD)
 *   - GAM: basis-expanded smooth terms (B-splines)
 *   - GLMM: random effects as penalized smooth terms (planned)
 *
 * Model families: Gaussian, Binomial, Poisson, Gamma, Inverse Gaussian,
 *                 Negative Binomial, Tweedie, Multinomial, Cox PH
 *
 * Solvers: pathwise coordinate descent (warm-started lambda sequence),
 *          proximal Newton for non-Gaussian, IRLS for unpenalized,
 *          block coordinate descent for group penalties
 *
 * Features: regularization path, strong screening rules, active set,
 *           K-fold CV, relaxed fits, per-feature penalty factors,
 *           box constraints, observation weights, offsets,
 *           B-spline basis expansion for smooth terms
 */

#ifndef GAM_H
#define GAM_H

#include <stdint.h>
#include <stddef.h>

/* ========== Constants ========== */

/* Model families */
enum {
    GAM_FAMILY_GAUSSIAN    = 0,
    GAM_FAMILY_BINOMIAL    = 1,
    GAM_FAMILY_POISSON     = 2,
    GAM_FAMILY_GAMMA       = 3,
    GAM_FAMILY_INV_GAUSS   = 4,
    GAM_FAMILY_NEG_BINOM   = 5,
    GAM_FAMILY_TWEEDIE     = 6,
    GAM_FAMILY_MULTINOMIAL = 7,
    GAM_FAMILY_COX         = 8,
    GAM_FAMILY_HUBER       = 10,  /* Huber robust regression */
    GAM_FAMILY_QUANTILE    = 11   /* quantile regression (check loss) */
};

/* Link functions */
enum {
    GAM_LINK_IDENTITY  = 0,
    GAM_LINK_LOG       = 1,
    GAM_LINK_LOGIT     = 2,
    GAM_LINK_PROBIT    = 3,
    GAM_LINK_CLOGLOG   = 4,
    GAM_LINK_INVERSE   = 5,
    GAM_LINK_INV_SQ    = 6,
    GAM_LINK_SQRT      = 7
};

/* Penalty types */
enum {
    GAM_PENALTY_NONE       = 0,
    GAM_PENALTY_L1         = 1,   /* lasso */
    GAM_PENALTY_L2         = 2,   /* ridge */
    GAM_PENALTY_ELASTICNET = 3,   /* alpha*L1 + (1-alpha)*L2 */
    GAM_PENALTY_MCP        = 4,
    GAM_PENALTY_SCAD       = 5,
    GAM_PENALTY_GROUP_L1   = 6,   /* group lasso */
    GAM_PENALTY_SGL        = 7,   /* sparse group lasso */
    GAM_PENALTY_SLOPE      = 8,   /* SLOPE: sorted L1, FDR control */
    GAM_PENALTY_FUSED      = 9    /* fused lasso: L1 + TV(beta) */
};

/* Term types (for GAM) */
enum {
    GAM_TERM_LINEAR   = 0,   /* plain linear predictor (GLM) */
    GAM_TERM_SMOOTH   = 1,   /* B-spline smooth term */
    GAM_TERM_TENSOR   = 2    /* tensor product smooth (interaction) */
};

/* GAMLSS distribution types (distributional regression) */
enum {
    GAMLSS_NORMAL  = 0,    /* Normal(mu, sigma):  mu=identity, sigma=log */
    GAMLSS_GAMMA   = 1,    /* Gamma(mu, sigma):   mu=log, sigma=log (CV param) */
    GAMLSS_BETA    = 2     /* Beta(mu, phi):      mu=logit, phi=log (precision) */
};

/* ========== Structures ========== */

/* Smooth term specification (for GAM) */
typedef struct {
    int32_t feature;        /* feature index */
    int32_t n_knots;        /* number of interior knots (default 20) */
    int32_t degree;         /* B-spline degree (default 3 = cubic) */
    double  lambda_smooth;  /* smoothness penalty (0 = no penalty) */
} gam_smooth_t;

/* Tensor product smooth term (2D+ interaction) */
typedef struct {
    int32_t n_margins;         /* number of marginal dimensions (2-4) */
    int32_t features[4];       /* feature indices per margin */
    int32_t n_knots[4];        /* interior knots per margin (default 5) */
    int32_t degree[4];         /* spline degree per margin (default 3) */
    double  lambda_smooth[4];  /* smoothness penalty per marginal direction */
} gam_tensor_t;

/* Hyperparameters */
typedef struct {
    int32_t  family;            /* GAM_FAMILY_* */
    int32_t  link;              /* GAM_LINK_* (-1 = canonical) */
    int32_t  penalty;           /* GAM_PENALTY_* */
    double   alpha;             /* elastic net mixing: 1.0 = lasso, 0.0 = ridge */
    double   lambda_min_ratio;  /* ratio of lambda_min to lambda_max (default 1e-4 if n>=p, 1e-2 if n<p) */
    int32_t  n_lambda;          /* number of lambda values (default 100) */
    double  *lambda;            /* user-supplied lambda sequence (NULL = auto) */
    int32_t  n_lambda_user;     /* length of user-supplied lambda */

    /* Penalty modifiers */
    double  *penalty_factor;    /* per-feature penalty weight (NULL = all 1.0), length n_features */
    double  *lower_bounds;      /* per-feature lower bound (NULL = -inf), length n_features */
    double  *upper_bounds;      /* per-feature upper bound (NULL = +inf), length n_features */

    /* Nonconvex penalty params */
    double   gamma_mcp;         /* MCP gamma, default 3.0 */
    double   gamma_scad;        /* SCAD gamma, default 3.7 */

    /* Group penalty */
    int32_t *groups;            /* group assignment per feature (NULL = no groups) */
    int32_t  n_groups;          /* number of groups */

    /* SLOPE params */
    double   slope_q;           /* target FDR level for SLOPE (default 0.1) */
    double  *slope_lambda;      /* user-supplied SLOPE lambda sequence (NULL = BH) */
    int32_t  slope_n_lambda;    /* length of user-supplied SLOPE lambda sequence */

    /* Fused lasso params */
    int32_t  fused_order;       /* difference order: 1 = fused, 2 = linear trend filter (default 1) */

    /* Robust regression params */
    double   huber_gamma;       /* Huber threshold: quadratic for |r| <= gamma, linear otherwise (default 1.345) */
    double   quantile_tau;      /* quantile level for quantile regression, 0 < tau < 1 (default 0.5) */

    /* Solver control */
    double   tol;               /* convergence tolerance (default 1e-7) */
    int32_t  max_iter;          /* max CD iterations per lambda (default 10000) */
    int32_t  max_inner;         /* max inner IRLS iterations for non-Gaussian (default 25) */
    int32_t  screening;         /* 0 = none, 1 = strong rules (default 1) */

    /* Cross-validation */
    int32_t  n_folds;           /* 0 = no CV, >0 = K-fold CV */
    int32_t  stratified;        /* stratify folds by class (binomial/multinomial) */

    /* Relaxed fit */
    int32_t  relax;             /* 0 = no, 1 = relaxed fit (refit active set without penalty) */

    /* Data handling */
    int32_t  standardize;       /* standardize features before fitting (default 1) */
    int32_t  fit_intercept;     /* fit intercept term (default 1) */
    double  *sample_weight;     /* observation weights (NULL = uniform) */
    double  *offset;            /* fixed offset in linear predictor (NULL = none) */

    /* Family-specific */
    double   tweedie_power;     /* Tweedie variance power (1 < p < 2, default 1.5) */
    double   neg_binom_theta;   /* NB dispersion (0 = estimate, >0 = fixed) */

    /* GAM smooth terms */
    gam_smooth_t *smooths;      /* smooth term specifications (NULL = GLM only) */
    int32_t  n_smooths;         /* number of smooth terms */

    /* Tensor product smooth terms */
    gam_tensor_t *tensors;      /* tensor product specifications (NULL = no interactions) */
    int32_t  n_tensors;         /* number of tensor product terms */

    /* Misc */
    uint32_t seed;              /* for CV fold assignment */
} gam_params_t;

/* Single model at one lambda value */
typedef struct {
    double  *beta;              /* coefficients (intercept at [0], features at [1..]) */
    int32_t  n_coefs;           /* total coefficients including intercept */
    double   lambda;            /* regularization parameter */
    double   deviance;          /* model deviance */
    double   null_deviance;     /* null model deviance */
    int32_t  df;                /* number of nonzero coefficients (excl. intercept) */
    int32_t  n_iter;            /* iterations to converge */
    double   cv_mean;           /* mean CV error (if CV was run, else NAN) */
    double   cv_se;             /* SE of CV error (if CV was run, else NAN) */
} gam_fit_t;

/* Regularization path result */
typedef struct {
    gam_fit_t *fits;            /* array of fits along the path */
    int32_t    n_fits;          /* number of lambda values actually computed */
    int32_t    idx_min;         /* index of lambda with minimum CV error (-1 if no CV) */
    int32_t    idx_1se;         /* index of most-regularized lambda within 1 SE (-1 if no CV) */
    int32_t    n_features;      /* number of original features */
    int32_t    n_coefs;         /* number of coefficients per model (incl. intercept) */
    int32_t    family;          /* family used */
    int32_t    link;            /* link function used */
    int32_t    penalty;         /* penalty used */
    double     alpha;           /* elastic net alpha */
    double    *x_mean;          /* feature means (if standardized), length n_features */
    double    *x_sd;            /* feature SDs (if standardized), length n_features */
    double     y_mean;          /* response mean (Gaussian only) */
    double     y_sd;            /* response SD (Gaussian only) */
    /* GAM-specific */
    int32_t    n_basis_total;   /* total basis columns (smooth terms expanded) */
    int32_t   *basis_map;       /* maps basis column -> original feature index */
    gam_smooth_t *smooths;      /* smooth term specs (owned copy, for prediction) */
    int32_t    n_smooths;       /* number of smooth terms */
    gam_tensor_t *tensors;      /* tensor product specs (owned copy, for prediction) */
    int32_t    n_tensors;       /* number of tensor product terms */

    /* Relaxed fits (if relax=1) */
    gam_fit_t *relaxed_fits;    /* same length as fits, NULL if relax=0 */

    /* Multi-task (n_tasks <= 1 means single-response) */
    int32_t    n_tasks;         /* number of response columns */

    /* GAMLSS (0 = not GAMLSS, >0 = GAMLSS distribution type + 1) */
    int32_t    family_gamlss;   /* 0=none, 1=GAMLSS_NORMAL+1, 2=GAMLSS_GAMMA+1, 3=GAMLSS_BETA+1 */
} gam_path_t;

/* ========== Public API ========== */

/* Initialize params to defaults */
void gam_params_init(gam_params_t *params);

/* Get canonical link for a family */
int32_t gam_canonical_link(int32_t family);

/* Fit regularization path.
 * X: row-major float64, nrow * ncol
 * y: float64 (binary 0/1 for binomial, counts for Poisson, continuous for Gaussian, etc.)
 *    For Cox: y = time, and status must be passed via gam_fit_cox() instead.
 * Returns NULL on error (check gam_get_error()) */
gam_path_t *gam_fit(
    const double *X, int32_t nrow, int32_t ncol,
    const double *y,
    const gam_params_t *params
);

/* Fit Cox PH model (separate because it needs time + status).
 * time: survival/follow-up times
 * status: event indicator (1 = event, 0 = censored) */
gam_path_t *gam_fit_cox(
    const double *X, int32_t nrow, int32_t ncol,
    const double *time, const double *status,
    const gam_params_t *params
);

/* Fit multi-task regularization path (Gaussian, identity link).
 * X: row-major float64, nrow * ncol
 * Y: row-major float64, nrow * n_tasks (multiple response columns)
 * Uses L1/L2 mixed norm penalty across tasks (shared feature selection).
 * Returns NULL on error (check gam_get_error()) */
gam_path_t *gam_fit_multi(
    const double *X, int32_t nrow, int32_t ncol,
    const double *Y, int32_t n_tasks,
    const gam_params_t *params
);

/* Fit multinomial logistic regression with regularization path.
 * X: row-major float64, nrow * ncol
 * y: float64 class labels (0, 1, ..., n_classes-1)
 * n_classes: number of classes (must be >= 3; use binomial for 2-class)
 * Penalty applied per-feature across all classes (grouped).
 * Coefficients: n_classes * (ncol + 1) per fit.
 * Returns NULL on error (check gam_get_error()) */
gam_path_t *gam_fit_multinomial(
    const double *X, int32_t nrow, int32_t ncol,
    const double *y, int32_t n_classes,
    const gam_params_t *params
);

/* Predict class probabilities for multinomial model.
 * out: float64 array of length nrow * n_classes, row-major, caller-allocated.
 * Returns 0 on success, -1 on error. */
int gam_predict_multinomial(
    const gam_path_t *path, int32_t fit_idx,
    const double *X, int32_t nrow, int32_t ncol,
    double *out
);

/* Predict for multi-task model.
 * out: float64 array of length nrow * n_tasks, row-major, caller-allocated.
 * Returns 0 on success, -1 on error. */
int gam_predict_multi(
    const gam_path_t *path, int32_t fit_idx,
    const double *X, int32_t nrow, int32_t ncol,
    double *out
);

/* Predict linear predictor (eta = X*beta + offset) for a single fit.
 * out: float64 array of length nrow, caller-allocated.
 * Returns 0 on success, -1 on error. */
int gam_predict_eta(
    const gam_path_t *path, int32_t fit_idx,
    const double *X, int32_t nrow, int32_t ncol,
    double *out
);

/* Predict response (mu = g^{-1}(eta)) for a single fit.
 * out: float64 array of length nrow, caller-allocated.
 * Returns 0 on success, -1 on error. */
int gam_predict(
    const gam_path_t *path, int32_t fit_idx,
    const double *X, int32_t nrow, int32_t ncol,
    double *out
);

/* Predict class probabilities (binomial/multinomial).
 * For binomial: out has length nrow (P(y=1)).
 * For multinomial: out has length nrow * n_classes.
 * Returns 0 on success, -1 on error. */
int gam_predict_proba(
    const gam_path_t *path, int32_t fit_idx,
    const double *X, int32_t nrow, int32_t ncol,
    double *out
);

/* Get deviance, AIC, BIC for a single fit */
double gam_deviance(const gam_path_t *path, int32_t fit_idx);
double gam_aic(const gam_path_t *path, int32_t fit_idx, int32_t nrow);
double gam_bic(const gam_path_t *path, int32_t fit_idx, int32_t nrow);

/* Serialize path to binary blob.
 * Returns 0 on success, -1 on error. */
int gam_save(const gam_path_t *path, char **out_buf, int32_t *out_len);

/* Deserialize path from binary blob.
 * Returns NULL on error. */
gam_path_t *gam_load(const char *buf, int32_t len);

/* Free path and all owned memory. */
void gam_free(gam_path_t *path);

/* Free a buffer returned by gam_save. */
void gam_free_buffer(void *ptr);

/* Get last error message. */
const char *gam_get_error(void);

/* ========== GAMLSS (distributional regression) ========== */

/* Fit GAMLSS distributional regression.
 * Models both location (mu) and scale (sigma/phi) as functions of X.
 * Coefficients: 2*(ncol+1) per fit -- first (ncol+1) for mu, next for sigma/phi.
 * Returns path with n_tasks=2, family_gamlss set to distribution type.
 * Returns NULL on error (check gam_get_error()). */
gam_path_t *gam_fit_gamlss(
    const double *X, int32_t nrow, int32_t ncol,
    const double *y,
    int32_t distribution,    /* GAMLSS_NORMAL, GAMLSS_GAMMA, GAMLSS_BETA */
    const gam_params_t *params
);

/* Predict mu and sigma/phi for GAMLSS model.
 * out: nrow * 2 doubles, row-major [mu_0, sigma_0, mu_1, sigma_1, ...]
 * Returns 0 on success, -1 on error. */
int gam_predict_gamlss(
    const gam_path_t *path, int32_t fit_idx,
    const double *X, int32_t nrow, int32_t ncol,
    double *out
);

/* ========== B-spline utilities (for GAM) ========== */

/* Compute B-spline basis matrix for a single feature.
 * x: input values, length n
 * knots: interior knots (sorted), length n_knots
 * degree: spline degree (3 = cubic)
 * out: output basis matrix, length n * n_basis, row-major
 *      n_basis = n_knots + degree + 1
 * Returns n_basis on success, -1 on error. */
int32_t gam_bspline_basis(
    const double *x, int32_t n,
    const double *knots, int32_t n_knots, int32_t degree,
    double *out
);

/* Compute quantile-spaced knots for a feature.
 * x: input values, length n
 * n_knots: desired number of interior knots
 * knots_out: output array, length n_knots, caller-allocated
 * Returns 0 on success, -1 on error. */
int gam_quantile_knots(
    const double *x, int32_t n,
    int32_t n_knots,
    double *knots_out
);

/* Compute tensor product basis for two features (row-wise Kronecker product).
 * x1, x2: input values, length n
 * Returns n_basis = n_basis1 * n_basis2 on success, -1 on error.
 * out: output basis matrix, length n * n_basis, row-major, caller-allocated. */
int32_t gam_tensor_basis(
    const double *x1, const double *x2, int32_t n,
    const double *knots1, int32_t n_knots1, int32_t degree1,
    const double *knots2, int32_t n_knots2, int32_t degree2,
    double *out
);

/* Compute 2nd-derivative smoothness penalty matrix for B-spline basis.
 * n_basis: number of basis functions
 * degree: spline degree
 * knots_full: full knot vector (including boundary repeats), length n_basis + degree + 1
 * out: output penalty matrix, length n_basis * n_basis, row-major (symmetric)
 * Returns 0 on success, -1 on error. */
int gam_smoothness_penalty(
    int32_t n_basis, int32_t degree,
    const double *knots_full,
    double *out
);

#endif /* GAM_H */
