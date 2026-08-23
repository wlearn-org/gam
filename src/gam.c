/*
 * gam.c -- Generalized Additive Models (C11, from scratch)
 *
 * Core algorithm: pathwise coordinate descent with warm starts.
 * For non-Gaussian families: proximal Newton (IRLS outer, CD inner).
 *
 * References:
 *   Friedman, Hastie, Tibshirani (2010). "Regularization Paths for
 *     Generalized Linear Models via Coordinate Descent." JSS 33(1).
 *   Zou & Hastie (2005). "Regularization and Variable Selection via
 *     the Elastic Net." JRSS-B 67(2):301-320.
 *   Fan & Li (2001). "Variable Selection via Nonconcave Penalized
 *     Likelihood." JASA 96(456):1348-1360. (SCAD)
 *   Zhang (2010). "Nearly Unbiased Variable Selection Under Minimax
 *     Concave Penalty." Annals of Statistics 38(2):894-942. (MCP)
 */

#define _POSIX_C_SOURCE 200809L

#include "gam.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========== Thread-local error message ========== */

static _Thread_local char gam_error_buf[256] = {0};

static void gam_set_error(const char *msg) {
    strncpy(gam_error_buf, msg, sizeof(gam_error_buf) - 1);
    gam_error_buf[sizeof(gam_error_buf) - 1] = '\0';
}

const char *gam_get_error(void) {
    return gam_error_buf;
}

/* ========== Math helpers ========== */

static inline double soft_threshold(double z, double lambda) {
    if (z > lambda) return z - lambda;
    if (z < -lambda) return z + lambda;
    return 0.0;
}

static inline double clamp(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/* Inverse normal CDF (quantile function) -- Beasley-Springer-Moro approximation */
static double norm_quantile(double p) {
    if (p <= 0.0) return -1e30;
    if (p >= 1.0) return 1e30;
    if (fabs(p - 0.5) < 1e-15) return 0.0;

    /* Rational approximation for central region */
    double r, x;
    if (p > 0.5) {
        r = 1.0 - p;
    } else {
        r = p;
    }
    double t = sqrt(-2.0 * log(r));
    /* Abramowitz & Stegun 26.2.23 */
    double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
    double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
    x = t - (c0 + t * (c1 + t * c2)) / (1.0 + t * (d1 + t * (d2 + t * d3)));
    if (p < 0.5) x = -x;
    return x;
}

/* ========== SLOPE helpers ========== */

/* Generate BH (Benjamini-Hochberg) lambda sequence for SLOPE.
 * lambda_j = Phi^{-1}(1 - j*q/(2*p)) for j = 1..p
 * Ensures non-increasing and positive. */
static void slope_bh_sequence(double *seq, int32_t p, double q) {
    for (int32_t j = 0; j < p; j++) {
        double prob = 1.0 - (j + 1) * q / (2.0 * p);
        if (prob <= 0.5) prob = 0.5 + 1e-10;  /* ensure positive quantile */
        seq[j] = norm_quantile(prob);
    }
    /* Ensure non-increasing */
    for (int32_t j = 1; j < p; j++) {
        if (seq[j] > seq[j - 1]) seq[j] = seq[j - 1];
    }
}

/* Comparison function for sorting indices by descending absolute value */
typedef struct { int32_t idx; double abs_val; } slope_pair_t;

static int slope_cmp_desc(const void *a, const void *b) {
    double va = ((const slope_pair_t *)a)->abs_val;
    double vb = ((const slope_pair_t *)b)->abs_val;
    if (va > vb) return -1;
    if (va < vb) return 1;
    return 0;
}

/* SLOPE proximal operator: prox_{t*J}(z) where J(beta) = sum_j slope_lambda_j * |beta|_(j)
 * |beta|_(j) is the j-th largest |beta_j|.
 *
 * Algorithm (Bogdan et al. 2015, ADMM formulation):
 * 1. Sort |z| in decreasing order -> u
 * 2. Apply isotonic regression of (|z| - t*slope_lambda) with weights to get v
 * 3. Threshold: result_j = sign(z_j) * max(v_{rank(j)}, 0)
 *
 * We use the fast prox from Larsson et al. (2025): "SLOPE via ADMM"
 * which reduces to pool-adjacent-violators on the sorted sequence.
 */
static void slope_prox(
    double *beta, int32_t p,
    const double *slope_lambda, double scale,
    double *work_abs __attribute__((unused)), slope_pair_t *work_pairs, double *work_v
) {
    /* Build sorted pairs */
    for (int32_t j = 0; j < p; j++) {
        work_pairs[j].idx = j;
        work_pairs[j].abs_val = fabs(beta[j]);
    }
    qsort(work_pairs, (size_t)p, sizeof(slope_pair_t), slope_cmp_desc);

    /* Compute s_j = |z|_(j) - scale * slope_lambda_j */
    for (int32_t j = 0; j < p; j++) {
        work_v[j] = work_pairs[j].abs_val - scale * slope_lambda[j];
    }

    /* Isotonic regression (PAVA) on work_v with unit weights, non-increasing constraint.
     * This gives the proximal operator of the sorted L1 norm. */
    double *pava_out = (double *)malloc((size_t)p * sizeof(double));
    if (!pava_out) return;  /* allocation failure -- leave beta unchanged */

    /* Copy to pava_out */
    for (int32_t j = 0; j < p; j++) pava_out[j] = work_v[j];

    /* PAVA for non-increasing sequence */
    for (int32_t j = 1; j < p; j++) {
        if (pava_out[j] > pava_out[j - 1]) {
            /* Merge with previous block */
            int32_t start = j - 1;
            double sum = pava_out[j - 1] + pava_out[j];
            int32_t count = 2;

            /* Keep merging backwards while violated */
            while (start > 0 && sum / count > pava_out[start - 1]) {
                start--;
                sum += pava_out[start];
                count++;
            }
            double avg = sum / count;
            for (int32_t k = start; k <= j; k++) {
                pava_out[k] = avg;
            }
        }
    }

    /* Apply: beta_j = sign(z_j) * max(pava_out[rank_j], 0) */
    for (int32_t j = 0; j < p; j++) {
        double v = fmax(pava_out[j], 0.0);
        int32_t orig_idx = work_pairs[j].idx;
        beta[orig_idx] = (beta[orig_idx] >= 0 ? 1.0 : -1.0) * v;
    }

    free(pava_out);
}

/* Standard normal CDF (for probit link) */
static double norm_cdf(double x) {
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

/* Standard normal PDF (for probit link derivative) */
static double norm_pdf(double x) {
    return exp(-0.5 * x * x) / sqrt(2.0 * M_PI);
}

/* ========== Link functions ========== */

/* g(mu) -> eta (currently unused but part of the API for future use) */
static double link_fn(int32_t link, double mu) __attribute__((unused));
static double link_fn(int32_t link, double mu) {
    switch (link) {
        case GAM_LINK_IDENTITY: return mu;
        case GAM_LINK_LOG:      return log(fmax(mu, 1e-10));
        case GAM_LINK_LOGIT:    return log(fmax(mu, 1e-10) / fmax(1.0 - mu, 1e-10));
        case GAM_LINK_PROBIT:   {
            double cmu = clamp(mu, 1e-10, 1.0 - 1e-10);
            /* inverse of norm_cdf -- use approximation via erfinv */
            /* For simplicity, use Newton iteration on norm_cdf */
            double eta = 0.0;
            for (int i = 0; i < 20; i++) {
                double f = norm_cdf(eta) - cmu;
                double df = norm_pdf(eta);
                if (fabs(df) < 1e-15) break;
                eta -= f / df;
            }
            return eta;
        }
        case GAM_LINK_CLOGLOG:  return log(-log(fmax(1.0 - mu, 1e-10)));
        case GAM_LINK_INVERSE:  return 1.0 / fmax(fabs(mu), 1e-10) * (mu >= 0 ? 1 : -1);
        case GAM_LINK_INV_SQ:   return 1.0 / fmax(mu * mu, 1e-20);
        case GAM_LINK_SQRT:     return sqrt(fmax(mu, 0.0));
        default:                return mu;
    }
}

/* g^{-1}(eta) -> mu */
static double linkinv_fn(int32_t link, double eta) {
    switch (link) {
        case GAM_LINK_IDENTITY: return eta;
        case GAM_LINK_LOG:      return exp(clamp(eta, -30.0, 30.0));
        case GAM_LINK_LOGIT:    {
            double e = clamp(eta, -30.0, 30.0);
            return 1.0 / (1.0 + exp(-e));
        }
        case GAM_LINK_PROBIT:   return norm_cdf(eta);
        case GAM_LINK_CLOGLOG:  return 1.0 - exp(-exp(clamp(eta, -30.0, 30.0)));
        case GAM_LINK_INVERSE:  return 1.0 / fmax(fabs(eta), 1e-10) * (eta >= 0 ? 1 : -1);
        case GAM_LINK_INV_SQ:   return 1.0 / sqrt(fmax(fabs(eta), 1e-20));
        case GAM_LINK_SQRT:     return eta * fabs(eta);  /* sign-preserving square */
        default:                return eta;
    }
}

/* d(mu)/d(eta) = d(linkinv)/d(eta) */
static double dmu_deta(int32_t link, double eta) {
    switch (link) {
        case GAM_LINK_IDENTITY: return 1.0;
        case GAM_LINK_LOG:      return exp(clamp(eta, -30.0, 30.0));
        case GAM_LINK_LOGIT:    {
            double mu = linkinv_fn(GAM_LINK_LOGIT, eta);
            return fmax(mu * (1.0 - mu), 1e-10);
        }
        case GAM_LINK_PROBIT:   return fmax(norm_pdf(eta), 1e-10);
        case GAM_LINK_CLOGLOG:  {
            double e = clamp(eta, -30.0, 30.0);
            return fmax(exp(e - exp(e)), 1e-10);
        }
        case GAM_LINK_INVERSE:  return -1.0 / fmax(eta * eta, 1e-20);
        case GAM_LINK_INV_SQ:   return -0.5 / fmax(fabs(eta) * sqrt(fabs(eta)), 1e-20);
        case GAM_LINK_SQRT:     return 2.0 * fabs(eta);
        default:                return 1.0;
    }
}

/* ========== Variance functions V(mu) ========== */

static double variance_fn(int32_t family, double mu, double tweedie_p, double nb_theta) {
    switch (family) {
        case GAM_FAMILY_GAUSSIAN:  return 1.0;
        case GAM_FAMILY_BINOMIAL:  return fmax(mu * (1.0 - mu), 1e-10);
        case GAM_FAMILY_POISSON:   return fmax(mu, 1e-10);
        case GAM_FAMILY_GAMMA:     return fmax(mu * mu, 1e-10);
        case GAM_FAMILY_INV_GAUSS: return fmax(mu * mu * mu, 1e-10);
        case GAM_FAMILY_NEG_BINOM: return fmax(mu + mu * mu / fmax(nb_theta, 1e-10), 1e-10);
        case GAM_FAMILY_TWEEDIE:   return fmax(pow(fmax(mu, 1e-10), tweedie_p), 1e-10);
        default:                   return 1.0;
    }
}

/* ========== Deviance contribution per observation ========== */

static double deviance_unit(int32_t family, double y, double mu,
                            double tweedie_p, double nb_theta) {
    double safe_mu = fmax(mu, 1e-10);
    switch (family) {
        case GAM_FAMILY_GAUSSIAN:
            return (y - mu) * (y - mu);
        case GAM_FAMILY_BINOMIAL: {
            double p = clamp(mu, 1e-10, 1.0 - 1e-10);
            double d = 0.0;
            if (y > 0) d += y * log(y / p);
            if (y < 1) d += (1.0 - y) * log((1.0 - y) / (1.0 - p));
            return 2.0 * d;
        }
        case GAM_FAMILY_POISSON: {
            double d = 0.0;
            if (y > 0) d = y * log(y / safe_mu);
            d -= (y - safe_mu);
            return 2.0 * d;
        }
        case GAM_FAMILY_GAMMA:
            return 2.0 * (-log(y / safe_mu) + (y - safe_mu) / safe_mu);
        case GAM_FAMILY_INV_GAUSS:
            return (y - safe_mu) * (y - safe_mu) / (y * safe_mu * safe_mu);
        case GAM_FAMILY_NEG_BINOM: {
            double theta = fmax(nb_theta, 1e-10);
            double d = 0.0;
            if (y > 0) d += y * log(y / safe_mu);
            d += (y + theta) * log((safe_mu + theta) / (y + theta));
            return 2.0 * d;
        }
        case GAM_FAMILY_TWEEDIE: {
            /* Tweedie deviance: 2 * [y^(2-p)/((1-p)(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p)] */
            double p = tweedie_p;
            if (fabs(p - 1.0) < 1e-6) {
                /* Poisson-like */
                double d = (y > 0) ? y * log(y / safe_mu) : 0.0;
                return 2.0 * (d - (y - safe_mu));
            }
            if (fabs(p - 2.0) < 1e-6) {
                /* Gamma-like */
                return 2.0 * (-log(y / safe_mu) + (y - safe_mu) / safe_mu);
            }
            double t1 = pow(fmax(y, 1e-30), 2.0 - p) / ((1.0 - p) * (2.0 - p));
            double t2 = y * pow(safe_mu, 1.0 - p) / (1.0 - p);
            double t3 = pow(safe_mu, 2.0 - p) / (2.0 - p);
            return 2.0 * (t1 - t2 + t3);
        }
        case GAM_FAMILY_HUBER:
            /* Not called directly; placeholder returning squared error */
            return (y - mu) * (y - mu);
        case GAM_FAMILY_QUANTILE:
            /* Not called directly; placeholder */
            return (y - mu) * (y - mu);
        default:
            return (y - mu) * (y - mu);
    }
}

/* ========== Canonical link ========== */

int32_t gam_canonical_link(int32_t family) {
    switch (family) {
        case GAM_FAMILY_GAUSSIAN:    return GAM_LINK_IDENTITY;
        case GAM_FAMILY_BINOMIAL:    return GAM_LINK_LOGIT;
        case GAM_FAMILY_POISSON:     return GAM_LINK_LOG;
        case GAM_FAMILY_GAMMA:       return GAM_LINK_INVERSE;
        case GAM_FAMILY_INV_GAUSS:   return GAM_LINK_INV_SQ;
        case GAM_FAMILY_NEG_BINOM:   return GAM_LINK_LOG;
        case GAM_FAMILY_TWEEDIE:     return GAM_LINK_LOG;
        case GAM_FAMILY_MULTINOMIAL: return GAM_LINK_LOGIT;
        case GAM_FAMILY_COX:         return GAM_LINK_LOG;
        case GAM_FAMILY_HUBER:       return GAM_LINK_IDENTITY;
        case GAM_FAMILY_QUANTILE:    return GAM_LINK_IDENTITY;
        default:                     return GAM_LINK_IDENTITY;
    }
}

/* ========== Params defaults ========== */

void gam_params_init(gam_params_t *params) {
    memset(params, 0, sizeof(*params));
    params->family = GAM_FAMILY_GAUSSIAN;
    params->link = -1;  /* canonical */
    params->penalty = GAM_PENALTY_ELASTICNET;
    params->alpha = 1.0;  /* lasso by default */
    params->lambda_min_ratio = 0.0;  /* auto */
    params->n_lambda = 100;
    params->gamma_mcp = 3.0;
    params->gamma_scad = 3.7;
    params->tol = 1e-7;
    params->max_iter = 10000;
    params->max_inner = 25;
    params->screening = 1;
    params->standardize = 1;
    params->fit_intercept = 1;
    params->tweedie_power = 1.5;
    params->slope_q = 0.1;
    params->fused_order = 1;
    params->huber_gamma = 1.345;   /* 95% efficiency at normal */
    params->quantile_tau = 0.5;    /* median regression */
    params->seed = 42;
}

/* ========== B-spline basis ========== */

/* De Boor recursion for B-spline basis evaluation */
int32_t gam_bspline_basis(
    const double *x, int32_t n,
    const double *knots, int32_t n_knots, int32_t degree,
    double *out
) {
    /* Full knot vector: degree+1 repeats of min at start, interior knots, degree+1 repeats of max at end */
    int32_t n_basis = n_knots + degree + 1;
    int32_t n_full = n_basis + degree + 1;

    double *t = (double *)calloc((size_t)n_full, sizeof(double));
    if (!t) { gam_set_error("bspline: alloc failed"); return -1; }

    /* Find data range */
    double xmin = x[0], xmax = x[0];
    for (int32_t i = 1; i < n; i++) {
        if (x[i] < xmin) xmin = x[i];
        if (x[i] > xmax) xmax = x[i];
    }

    /* Build full knot vector */
    for (int32_t i = 0; i <= degree; i++) t[i] = xmin;
    for (int32_t i = 0; i < n_knots; i++) t[degree + 1 + i] = knots[i];
    for (int32_t i = 0; i <= degree; i++) t[n_knots + degree + 1 + i] = xmax;

    /* Initialize output to zero */
    memset(out, 0, (size_t)n * (size_t)n_basis * sizeof(double));

    /* For each data point, evaluate all basis functions via de Boor */
    /* Use a temporary buffer for the recursion */
    double *B = (double *)calloc((size_t)(n_basis + degree), sizeof(double));
    if (!B) { free(t); gam_set_error("bspline: alloc failed"); return -1; }

    for (int32_t i = 0; i < n; i++) {
        double xi = x[i];
        /* Clamp to knot range */
        if (xi < xmin) xi = xmin;
        if (xi > xmax) xi = xmax;
        /* Handle right boundary: shift slightly left */
        if (xi >= xmax) xi = xmax - 1e-10 * (xmax - xmin + 1.0);

        /* Degree 0: indicator functions */
        int32_t nb0 = n_full - 1;
        for (int32_t j = 0; j < nb0; j++) {
            B[j] = (xi >= t[j] && xi < t[j + 1]) ? 1.0 : 0.0;
        }

        /* Recursive build for degrees 1..degree */
        for (int32_t d = 1; d <= degree; d++) {
            int32_t nb_d = nb0 - d;
            for (int32_t j = 0; j < nb_d; j++) {
                double left = 0.0, right = 0.0;
                double denom1 = t[j + d] - t[j];
                if (denom1 > 0) left = (xi - t[j]) / denom1 * B[j];
                double denom2 = t[j + d + 1] - t[j + 1];
                if (denom2 > 0) right = (t[j + d + 1] - xi) / denom2 * B[j + 1];
                B[j] = left + right;
            }
        }

        /* Copy to output row */
        for (int32_t j = 0; j < n_basis; j++) {
            out[i * n_basis + j] = B[j];
        }
    }

    free(B);
    free(t);
    return n_basis;
}

/* Compute quantile-spaced knots */
int gam_quantile_knots(
    const double *x, int32_t n,
    int32_t n_knots,
    double *knots_out
) {
    if (n < 2 || n_knots < 1) {
        gam_set_error("quantile_knots: invalid input");
        return -1;
    }

    /* Sort a copy */
    double *sorted = (double *)malloc((size_t)n * sizeof(double));
    if (!sorted) { gam_set_error("quantile_knots: alloc failed"); return -1; }
    memcpy(sorted, x, (size_t)n * sizeof(double));

    /* Simple insertion sort for moderate n (will be called on feature columns) */
    for (int32_t i = 1; i < n; i++) {
        double key = sorted[i];
        int32_t j = i - 1;
        while (j >= 0 && sorted[j] > key) {
            sorted[j + 1] = sorted[j];
            j--;
        }
        sorted[j + 1] = key;
    }

    /* Place knots at quantiles 1/(n_knots+1), 2/(n_knots+1), ... */
    for (int32_t k = 0; k < n_knots; k++) {
        double q = (double)(k + 1) / (double)(n_knots + 1);
        double idx = q * (n - 1);
        int32_t lo = (int32_t)idx;
        int32_t hi = lo + 1;
        if (hi >= n) hi = n - 1;
        double frac = idx - lo;
        knots_out[k] = sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
    }

    free(sorted);
    return 0;
}

/* Smoothness penalty matrix (2nd derivative) */
int gam_smoothness_penalty(
    int32_t n_basis, int32_t degree,
    const double *knots_full,
    double *out
) {
    if (degree < 2) {
        /* No 2nd derivative penalty for degree < 2 */
        memset(out, 0, (size_t)n_basis * (size_t)n_basis * sizeof(double));
        return 0;
    }

    /* Numerical integration of B''_i * B''_j over the knot span */
    /* Use Gauss-Legendre quadrature with enough points */
    int32_t n_intervals = n_basis + degree + 1 - 1;
    int32_t n_quad = 5;  /* 5-point GL per interval */

    /* GL nodes and weights on [-1, 1] (5-point) */
    static const double gl_x[5] = {
        -0.9061798459, -0.5384693101, 0.0,
         0.5384693101,  0.9061798459
    };
    static const double gl_w[5] = {
        0.2369268851, 0.4786286705, 0.5688888889,
        0.4786286705, 0.2369268851
    };

    memset(out, 0, (size_t)n_basis * (size_t)n_basis * sizeof(double));

    /* For each knot interval, evaluate 2nd derivative of each basis function
     * via finite differences on the basis evaluation */
    double h = 1e-5;  /* finite difference step */

    /* Allocate temp buffers for basis evaluation at shifted points */
    double *b_minus = (double *)calloc((size_t)n_basis, sizeof(double));
    double *b_center = (double *)calloc((size_t)n_basis, sizeof(double));
    double *b_plus = (double *)calloc((size_t)n_basis, sizeof(double));
    if (!b_minus || !b_center || !b_plus) {
        free(b_minus); free(b_center); free(b_plus);
        gam_set_error("smoothness_penalty: alloc failed");
        return -1;
    }

    /* Get knot span from the full knot vector */
    double xmin = knots_full[degree];
    double xmax = knots_full[n_basis];

    for (int32_t interval = 0; interval < n_intervals; interval++) {
        double a = knots_full[interval];
        double b = knots_full[interval + 1];
        if (b <= a) continue;
        if (b < xmin || a > xmax) continue;
        a = fmax(a, xmin);
        b = fmin(b, xmax);
        double half_len = 0.5 * (b - a);
        double mid = 0.5 * (a + b);

        for (int32_t q = 0; q < n_quad; q++) {
            double x_q = mid + half_len * gl_x[q];
            double w_q = gl_w[q] * half_len;

            /* Evaluate basis and 2nd derivatives via central finite diff */
            double pts[3] = { x_q - h, x_q, x_q + h };
            for (int32_t p = 0; p < 3; p++) {
                double *buf = (p == 0) ? b_minus : (p == 1) ? b_center : b_plus;
                /* Evaluate single point via gam_bspline_basis */
                /* Use the interior knots (strip boundary repeats) */
                int32_t n_interior = n_basis - degree - 1;
                const double *interior_knots = knots_full + degree + 1;
                gam_bspline_basis(&pts[p], 1, interior_knots, n_interior, degree, buf);
            }

            /* 2nd derivative: (f(x+h) - 2f(x) + f(x-h)) / h^2 */
            for (int32_t i = 0; i < n_basis; i++) {
                double d2i = (b_plus[i] - 2.0 * b_center[i] + b_minus[i]) / (h * h);
                for (int32_t j = i; j < n_basis; j++) {
                    double d2j = (b_plus[j] - 2.0 * b_center[j] + b_minus[j]) / (h * h);
                    double val = w_q * d2i * d2j;
                    out[i * n_basis + j] += val;
                    if (j != i) out[j * n_basis + i] += val;
                }
            }
        }
    }

    free(b_minus);
    free(b_center);
    free(b_plus);
    return 0;
}

/* ========== Tensor product basis ========== */

int32_t gam_tensor_basis(
    const double *x1, const double *x2, int32_t n,
    const double *knots1, int32_t n_knots1, int32_t degree1,
    const double *knots2, int32_t n_knots2, int32_t degree2,
    double *out
) {
    int32_t nb1 = n_knots1 + degree1 + 1;
    int32_t nb2 = n_knots2 + degree2 + 1;
    int32_t nb_total = nb1 * nb2;

    /* Compute marginal bases */
    double *B1 = (double *)calloc((size_t)n * (size_t)nb1, sizeof(double));
    double *B2 = (double *)calloc((size_t)n * (size_t)nb2, sizeof(double));
    if (!B1 || !B2) {
        free(B1); free(B2);
        gam_set_error("tensor_basis: alloc failed");
        return -1;
    }

    if (gam_bspline_basis(x1, n, knots1, n_knots1, degree1, B1) < 0 ||
        gam_bspline_basis(x2, n, knots2, n_knots2, degree2, B2) < 0) {
        free(B1); free(B2);
        return -1;
    }

    /* Row-wise Kronecker product */
    for (int32_t i = 0; i < n; i++) {
        for (int32_t j1 = 0; j1 < nb1; j1++) {
            for (int32_t j2 = 0; j2 < nb2; j2++) {
                out[i * nb_total + j1 * nb2 + j2] =
                    B1[i * nb1 + j1] * B2[i * nb2 + j2];
            }
        }
    }

    free(B1);
    free(B2);
    return nb_total;
}

/* ========== Internal: coordinate descent solver ========== */

/* Penalty block info for smooth/tensor terms (forward declaration for cd_work_t) */
typedef struct {
    int32_t  col_start;
    int32_t  block_size;
    double  *S;       /* block_size x block_size penalty matrix (owned) */
    double   lambda;
} penalty_block_t;

/* Working data for the CD solver */
typedef struct {
    int32_t  n;           /* rows */
    int32_t  p;           /* columns (features or basis-expanded) */
    double  *Xs;          /* standardized X, n*p row-major */
    double  *y;           /* response (or working response for IRLS) */
    double  *w;           /* working weights, length n */
    double  *r;           /* residuals, length n */
    double  *eta;         /* linear predictor, length n */
    double  *mu;          /* fitted values, length n */
    double  *beta;        /* coefficients, length p */
    double   intercept;   /* intercept */
    double  *x_mean;      /* column means, length p */
    double  *x_sd;        /* column SDs, length p */
    double  *xw_sq;       /* weighted sum of x^2 per feature, length p */
    double  *pf;          /* penalty factors, length p */
    double  *lb;          /* lower bounds, length p */
    double  *ub;          /* upper bounds, length p */
    int32_t *active;      /* active set flags, length p */
    int32_t *ever_active; /* ever been nonzero, length p */
    int32_t  family;
    int32_t  link;
    int32_t  penalty;
    double   alpha;
    double   gamma_mcp;
    double   gamma_scad;
    double   tweedie_p;
    double   nb_theta;
    int32_t  fit_intercept;
    double   tol;
    double  *sample_weight; /* user weights, length n (NULL = uniform) */
    double  *offset;        /* offset, length n (NULL = none) */

    /* SLOPE-specific */
    double  *slope_seq;     /* SLOPE lambda sequence, length p (NULL if not SLOPE) */
    slope_pair_t *slope_pairs; /* work buffer for SLOPE prox, length p */
    double  *slope_work;    /* work buffer for SLOPE prox, length p */
    double  *slope_abs;     /* work buffer for SLOPE prox, length p */

    /* Group penalty */
    int32_t *groups;        /* group assignment per feature, length p (NULL = no groups) */
    int32_t  n_groups;      /* number of groups */
    int32_t *group_start;   /* start index per group, length n_groups */
    int32_t *group_size;    /* group size, length n_groups */
    double   sgl_alpha;     /* SGL mixing: alpha * L1 + (1-alpha) * group L1 (default 0.5) */

    /* Fused lasso */
    int32_t  fused_order;   /* difference order (1 = fused, 2 = trend filter, default 1) */

    /* Robust regression */
    double   huber_gamma;   /* Huber threshold */
    double   quantile_tau;  /* quantile level */

    /* Smoothness penalty blocks (for GAM smooth/tensor terms) */
    int32_t  n_penalty_blocks;
    penalty_block_t *penalty_blocks;
} cd_work_t;

/* Compute lambda_max: smallest lambda for which all beta are zero */
static double compute_lambda_max(const cd_work_t *w) {
    double max_grad = 0.0;
    for (int32_t j = 0; j < w->p; j++) {
        if (w->pf[j] == 0.0) continue;  /* unpenalized */
        double grad = 0.0;
        for (int32_t i = 0; i < w->n; i++) {
            grad += w->Xs[i * w->p + j] * w->r[i] * w->w[i];
        }
        grad = fabs(grad) / w->n;
        double scaled = grad / fmax(w->pf[j], 1e-15);
        if (scaled > max_grad) max_grad = scaled;
    }
    if (w->alpha > 0) max_grad /= fmax(w->alpha, 1e-10);
    return max_grad;
}

/* Apply penalty proximal operator */
static double penalized_update(
    double z, double v, double lambda, double alpha,
    int32_t penalty, double gamma_mcp, double gamma_scad,
    double pf, double lb, double ub
) {
    double l1 = lambda * alpha * pf;
    double l2 = lambda * (1.0 - alpha) * pf;

    double result;
    switch (penalty) {
        case GAM_PENALTY_L1:
            result = soft_threshold(z, l1) / (v + l2);
            break;
        case GAM_PENALTY_L2:
            result = z / (v + lambda * pf);
            break;
        case GAM_PENALTY_ELASTICNET:
            result = soft_threshold(z, l1) / (v + l2);
            break;
        case GAM_PENALTY_MCP: {
            double absz = fabs(z);
            if (absz <= gamma_mcp * l1 * (v + l2)) {
                result = soft_threshold(z, l1) / (v + l2 - 1.0 / gamma_mcp);
                if (v + l2 - 1.0 / gamma_mcp <= 0)
                    result = soft_threshold(z, l1) / (v + l2);
            } else {
                result = z / v;
            }
            break;
        }
        case GAM_PENALTY_SCAD: {
            double absz = fabs(z);
            if (absz <= (v + l2 + l1)) {
                result = soft_threshold(z, l1) / (v + l2);
            } else if (absz <= gamma_scad * l1 * (v + l2)) {
                result = soft_threshold(z, gamma_scad * l1 / (gamma_scad - 1.0))
                         / (v + l2 - 1.0 / (gamma_scad - 1.0));
                if (v + l2 - 1.0 / (gamma_scad - 1.0) <= 0)
                    result = z / v;
            } else {
                result = z / v;
            }
            break;
        }
        default:  /* no penalty */
            result = z / v;
            break;
    }

    return clamp(result, lb, ub);
}

/* Single coordinate descent pass (Gaussian or weighted LS).
 * Returns the maximum weighted coordinate change: max_j |delta_j| * sqrt(xw_sq_j).
 * Returns 0.0 if nothing changed. */
static double cd_pass(cd_work_t *w, double lambda, int active_only) {
    double max_change = 0.0;

    for (int32_t j = 0; j < w->p; j++) {
        if (active_only && !w->active[j]) continue;

        double old_beta = w->beta[j];

        /* Compute partial residual gradient: z = (1/n) * sum(x_j * r * w) + xw_sq_j * beta_j */
        double z = 0.0;
        for (int32_t i = 0; i < w->n; i++) {
            z += w->Xs[i * w->p + j] * w->r[i] * w->w[i];
        }
        z = z / w->n + w->xw_sq[j] * old_beta;

        double v = w->xw_sq[j];

        /* Add smoothness penalty contribution */
        if (w->n_penalty_blocks > 0) {
            for (int32_t b = 0; b < w->n_penalty_blocks; b++) {
                int32_t cs = w->penalty_blocks[b].col_start;
                int32_t bs = w->penalty_blocks[b].block_size;
                if (j >= cs && j < cs + bs) {
                    int32_t jj = j - cs;
                    double lam_s = w->penalty_blocks[b].lambda;
                    const double *S = w->penalty_blocks[b].S;
                    /* Add diagonal to denominator */
                    v += lam_s * S[jj * bs + jj];
                    /* Subtract off-diagonal penalty gradient from z */
                    for (int32_t k = 0; k < bs; k++) {
                        if (k != jj)
                            z -= lam_s * S[jj * bs + k] * w->beta[cs + k];
                    }
                    break;
                }
            }
        }

        if (v < 1e-15) continue;

        double new_beta;
        if (w->pf[j] == 0.0) {
            /* Unpenalized feature */
            new_beta = z / v;
        } else {
            new_beta = penalized_update(
                z, v, lambda, w->alpha,
                w->penalty, w->gamma_mcp, w->gamma_scad,
                w->pf[j], w->lb[j], w->ub[j]
            );
        }

        if (new_beta != old_beta) {
            double delta = new_beta - old_beta;
            w->beta[j] = new_beta;
            /* Update residuals */
            for (int32_t i = 0; i < w->n; i++) {
                w->r[i] -= delta * w->Xs[i * w->p + j];
            }
            /* Track max weighted change (glmnet convergence criterion) */
            double wchange = fabs(delta) * sqrt(v);
            if (wchange > max_change) max_change = wchange;
            if (new_beta != 0.0) {
                w->active[j] = 1;
                w->ever_active[j] = 1;
            }
        }
    }

    /* Update intercept */
    if (w->fit_intercept) {
        double sum_wr = 0.0, sum_w = 0.0;
        for (int32_t i = 0; i < w->n; i++) {
            sum_wr += w->r[i] * w->w[i];
            sum_w += w->w[i];
        }
        double delta = sum_wr / fmax(sum_w, 1e-15);
        if (fabs(delta) > 1e-15) {
            w->intercept += delta;
            for (int32_t i = 0; i < w->n; i++) {
                w->r[i] -= delta;
            }
            if (fabs(delta) > max_change) max_change = fabs(delta);
        }
    }

    return max_change;
}

/* Block coordinate descent pass for Group Lasso / Sparse Group Lasso.
 * Updates entire groups at once using group soft-threshold. */
static int group_cd_pass(cd_work_t *w, double lambda, int active_only) {
    int any_changed = 0;

    for (int32_t g = 0; g < w->n_groups; g++) {
        int32_t gs = w->group_start[g];
        int32_t gsize = w->group_size[g];
        double sqrt_gsize = sqrt((double)gsize);

        /* Check if any member of the group is active */
        if (active_only) {
            int any_active = 0;
            for (int32_t k = 0; k < gsize; k++) {
                if (w->active[gs + k]) { any_active = 1; break; }
            }
            if (!any_active) continue;
        }

        /* Compute gradient vector z_g and weighted norm v_g for the group */
        double z_norm = 0.0;
        double v_avg = 0.0;
        double *z_g = (double *)malloc((size_t)gsize * sizeof(double));
        if (!z_g) continue;

        for (int32_t k = 0; k < gsize; k++) {
            int32_t j = gs + k;
            double z = 0.0;
            for (int32_t i = 0; i < w->n; i++) {
                z += w->Xs[i * w->p + j] * w->r[i] * w->w[i];
            }
            z = z / w->n + w->xw_sq[j] * w->beta[j];
            z_g[k] = z;
            z_norm += z * z;
            v_avg += w->xw_sq[j];
        }
        z_norm = sqrt(z_norm);
        v_avg /= gsize;
        if (v_avg < 1e-15) { free(z_g); continue; }

        /* Penalty factor (average for group) */
        double pf_g = 0.0;
        for (int32_t k = 0; k < gsize; k++) pf_g += w->pf[gs + k];
        pf_g /= gsize;
        if (pf_g == 0.0) {
            /* Unpenalized group: standard update per coordinate */
            for (int32_t k = 0; k < gsize; k++) {
                int32_t j = gs + k;
                double new_beta = z_g[k] / w->xw_sq[j];
                double delta = new_beta - w->beta[j];
                if (fabs(delta) > 1e-15) {
                    w->beta[j] = new_beta;
                    for (int32_t i = 0; i < w->n; i++) {
                        w->r[i] -= delta * w->Xs[i * w->p + j];
                    }
                    any_changed = 1;
                }
            }
            free(z_g);
            continue;
        }

        double l_group = lambda * pf_g * sqrt_gsize;

        if (w->penalty == GAM_PENALTY_SGL) {
            /* Sparse Group Lasso: L1 within group + group L2 penalty
             * First apply element-wise soft-threshold, then group threshold */
            double l1 = lambda * w->sgl_alpha * pf_g;
            double l_grp = lambda * (1.0 - w->sgl_alpha) * pf_g * sqrt_gsize;

            /* Element-wise soft-threshold */
            for (int32_t k = 0; k < gsize; k++) {
                z_g[k] = soft_threshold(z_g[k], l1);
            }
            /* Recompute norm after soft-threshold */
            z_norm = 0.0;
            for (int32_t k = 0; k < gsize; k++) z_norm += z_g[k] * z_g[k];
            z_norm = sqrt(z_norm);
            l_group = l_grp;
        }

        /* Group soft-threshold: beta_g = (1 - l_group / (v * ||z_g||))_+ * z_g / v */
        if (z_norm <= l_group / v_avg) {
            /* Entire group set to zero */
            for (int32_t k = 0; k < gsize; k++) {
                int32_t j = gs + k;
                double delta = -w->beta[j];
                if (fabs(delta) > 1e-15) {
                    w->beta[j] = 0.0;
                    for (int32_t i = 0; i < w->n; i++) {
                        w->r[i] -= delta * w->Xs[i * w->p + j];
                    }
                    any_changed = 1;
                }
            }
        } else {
            /* Shrink group */
            double scale = (1.0 - l_group / (v_avg * z_norm));
            for (int32_t k = 0; k < gsize; k++) {
                int32_t j = gs + k;
                double new_beta = scale * z_g[k] / v_avg;
                new_beta = clamp(new_beta, w->lb[j], w->ub[j]);
                double delta = new_beta - w->beta[j];
                if (fabs(delta) > 1e-15) {
                    w->beta[j] = new_beta;
                    for (int32_t i = 0; i < w->n; i++) {
                        w->r[i] -= delta * w->Xs[i * w->p + j];
                    }
                    any_changed = 1;
                    w->active[j] = 1;
                    w->ever_active[j] = 1;
                }
            }
        }

        free(z_g);
    }

    /* Update intercept */
    if (w->fit_intercept) {
        double sum_wr = 0.0, sum_w = 0.0;
        for (int32_t i = 0; i < w->n; i++) {
            sum_wr += w->r[i] * w->w[i];
            sum_w += w->w[i];
        }
        double delta = sum_wr / fmax(sum_w, 1e-15);
        if (fabs(delta) > 1e-15) {
            w->intercept += delta;
            for (int32_t i = 0; i < w->n; i++) {
                w->r[i] -= delta;
            }
            any_changed = 1;
        }
    }

    return any_changed;
}

/* Proximal gradient descent pass for SLOPE penalty.
 * Unlike CD, SLOPE updates all coordinates simultaneously because the
 * sorted L1 penalty couples all coordinates.
 * Uses backtracking line search for step size. */
static int slope_pgd_pass(cd_work_t *w, double lambda) {
    int any_changed = 0;

    /* Compute full gradient: g_j = -(1/n) sum_i x_ij * r_i * w_i + xw_sq_j * beta_j
     *                              = xw_sq_j * beta_j - (1/n) sum_i x_ij * r_i * w_i
     * Actually, the gradient of the smooth part (1/2n) sum w_i (y_i - X beta)^2 w.r.t. beta_j:
     *   g_j = -(1/n) sum x_ij r_i w_i  (where r = y - X beta - intercept)
     * We want z_j = beta_j - (1/L) * g_j  where g_j = -grad_j
     */

    /* Step size = 1/L where L = max eigenvalue of X^T W X / n
     * Approximate by L = max_j xw_sq_j (sum of weighted squared columns) */
    double L = 0.0;
    for (int32_t j = 0; j < w->p; j++) {
        if (w->xw_sq[j] > L) L = w->xw_sq[j];
    }
    if (L < 1e-10) L = 1.0;

    /* Compute z = beta + (1/L) * X^T W r / n  (gradient step) */
    double *z = w->slope_abs;  /* reuse as temp buffer */
    for (int32_t j = 0; j < w->p; j++) {
        double grad = 0.0;
        for (int32_t i = 0; i < w->n; i++) {
            grad += w->Xs[i * w->p + j] * w->r[i] * w->w[i];
        }
        grad /= w->n;
        z[j] = w->beta[j] + grad / L;
    }

    /* Apply SLOPE prox: prox_{(lambda/L)*J}(z) */
    double *prox_in = (double *)malloc((size_t)w->p * sizeof(double));
    if (!prox_in) return 0;
    for (int32_t j = 0; j < w->p; j++) prox_in[j] = z[j];

    slope_prox(prox_in, w->p, w->slope_seq, lambda / L,
               w->slope_abs, w->slope_pairs, w->slope_work);

    /* Update beta and residuals */
    for (int32_t j = 0; j < w->p; j++) {
        /* Apply bounds */
        double new_val = clamp(prox_in[j], w->lb[j], w->ub[j]);
        /* Apply penalty factor: unpenalized features get plain gradient step */
        if (w->pf[j] == 0.0) {
            new_val = z[j];  /* unpenalized: just gradient step */
        }

        double delta = new_val - w->beta[j];
        if (fabs(delta) > 1e-15) {
            w->beta[j] = new_val;
            for (int32_t i = 0; i < w->n; i++) {
                w->r[i] -= delta * w->Xs[i * w->p + j];
            }
            any_changed = 1;
            if (new_val != 0.0) {
                w->active[j] = 1;
                w->ever_active[j] = 1;
            }
        }
    }

    free(prox_in);

    /* Update intercept */
    if (w->fit_intercept) {
        double sum_wr = 0.0, sum_w = 0.0;
        for (int32_t i = 0; i < w->n; i++) {
            sum_wr += w->r[i] * w->w[i];
            sum_w += w->w[i];
        }
        double delta = sum_wr / fmax(sum_w, 1e-15);
        if (fabs(delta) > 1e-15) {
            w->intercept += delta;
            for (int32_t i = 0; i < w->n; i++) {
                w->r[i] -= delta;
            }
            any_changed = 1;
        }
    }

    return any_changed;
}

/* GAP Safe screening rule (Fercoq et al. 2015, Ndiaye et al. 2017).
 * Uses duality gap to construct a safe ball, then tests whether each
 * feature's correlation with the dual point can exceed the threshold.
 * Returns number of features screened out. Only valid for L1/EN/Lasso. */
static int gap_safe_screen(cd_work_t *w, double lambda) {
    /* Only for L1-type penalties */
    if (w->penalty != GAM_PENALTY_L1 && w->penalty != GAM_PENALTY_ELASTICNET &&
        w->penalty != GAM_PENALTY_MCP && w->penalty != GAM_PENALTY_SCAD) {
        return 0;
    }
    if (w->alpha <= 0.0) return 0;  /* pure ridge has no screening */

    int32_t n = w->n, p = w->p;
    double alpha = w->alpha;

    /* Compute primal objective */
    double rss = 0.0;
    for (int32_t i = 0; i < n; i++) {
        rss += w->r[i] * w->r[i] * w->w[i];
    }
    double primal = rss / (2.0 * n);
    for (int32_t j = 0; j < p; j++) {
        double b = fabs(w->beta[j]);
        if (b == 0.0 || w->pf[j] == 0.0) continue;
        double l1 = lambda * alpha * w->pf[j];
        double l2 = lambda * (1.0 - alpha) * w->pf[j];
        primal += l1 * b + 0.5 * l2 * b * b;
    }

    /* Construct dual feasible point: theta = r / (n * scaling)
     * scaling ensures max_j |<x_j, theta>| / pf_j <= alpha * lambda */
    double scaling = alpha * lambda;
    for (int32_t j = 0; j < p; j++) {
        if (w->pf[j] == 0.0) continue;
        double corr = 0.0;
        for (int32_t i = 0; i < n; i++) {
            corr += w->Xs[i * p + j] * w->r[i] * w->w[i];
        }
        corr = fabs(corr) / n;
        double max_allowed = alpha * lambda * w->pf[j];
        if (corr > max_allowed) {
            double s = corr / (max_allowed);
            if (s > scaling / (alpha * lambda)) {
                scaling = corr / w->pf[j];
            }
        }
    }

    /* theta = r * w / (n * scaling), ||theta||^2 */
    double theta_norm2 = 0.0;
    double ry = 0.0;
    for (int32_t i = 0; i < n; i++) {
        double ti = w->r[i] * w->w[i] / (n * scaling);
        theta_norm2 += ti * ti;
        ry += w->y[i] * ti;
    }

    /* Dual objective: D(theta) = -0.5 * n * ||theta||^2 + <y, theta>
     * (for standardized data with Gaussian) */
    double dual = -0.5 * n * theta_norm2;
    /* Add <y_centered, theta> accounting for intercept */
    for (int32_t i = 0; i < n; i++) {
        double yi_fitted = w->intercept;
        for (int32_t j = 0; j < p; j++) {
            yi_fitted += w->Xs[i * p + j] * w->beta[j];
        }
        dual += w->y[i] * w->r[i] * w->w[i] / (n * scaling);
    }

    /* Duality gap */
    double gap = primal - dual;
    if (gap < 0.0) gap = 0.0;  /* numerical issues */

    /* Safe radius */
    double radius = sqrt(2.0 * gap / n);

    /* Screen features */
    int screened = 0;
    for (int32_t j = 0; j < p; j++) {
        if (w->pf[j] == 0.0) { w->active[j] = 1; continue; }
        if (w->ever_active[j] && w->beta[j] != 0.0) { w->active[j] = 1; continue; }

        /* Correlation of x_j with dual point */
        double corr = 0.0;
        double xj_norm = 0.0;
        for (int32_t i = 0; i < n; i++) {
            double xij = w->Xs[i * p + j];
            double ti = w->r[i] * w->w[i] / (n * scaling);
            corr += xij * ti;
            xj_norm += xij * xij * w->w[i];
        }
        xj_norm = sqrt(xj_norm / n);

        /* Safe test: |corr| + ||x_j|| * radius < alpha * lambda * pf_j */
        double threshold = alpha * lambda * w->pf[j];
        if (fabs(corr) + xj_norm * radius < threshold) {
            w->active[j] = 0;
            screened++;
        } else {
            w->active[j] = 1;
        }
    }

    return screened;
}

/* Strong screening rule: discard features that are unlikely to be active */
static void strong_screen(cd_work_t *w, double lambda, double lambda_prev) {
    for (int32_t j = 0; j < w->p; j++) {
        if (w->pf[j] == 0.0) { w->active[j] = 1; continue; }
        if (w->ever_active[j]) { w->active[j] = 1; continue; }

        double grad = 0.0;
        for (int32_t i = 0; i < w->n; i++) {
            grad += w->Xs[i * w->p + j] * w->r[i] * w->w[i];
        }
        grad = fabs(grad) / w->n;

        double threshold = w->alpha * (2.0 * lambda - lambda_prev) * w->pf[j];
        w->active[j] = (grad >= threshold) ? 1 : 0;
    }
}

/* KKT check: verify all screened-out features are still zero */
static int kkt_check(cd_work_t *w, double lambda) {
    int all_ok = 1;
    for (int32_t j = 0; j < w->p; j++) {
        if (w->active[j]) continue;
        if (w->pf[j] == 0.0) continue;

        double grad = 0.0;
        for (int32_t i = 0; i < w->n; i++) {
            grad += w->Xs[i * w->p + j] * w->r[i] * w->w[i];
        }
        grad = fabs(grad) / w->n;

        if (grad > w->alpha * lambda * w->pf[j]) {
            w->active[j] = 1;
            w->ever_active[j] = 1;
            all_ok = 0;
        }
    }
    return all_ok;
}

/* ========== Anderson Acceleration for CD ========== */
/* Bertrand & Massias (2021), AISTATS. Extrapolation of CD iterates. */

#define AA_M 5   /* history depth */
#define AA_K 5   /* apply AA every K CD passes */

typedef struct {
    int32_t dim;          /* p + 1 (beta + intercept) */
    int32_t count;        /* number of stored iterates */
    double *X_hist;       /* (AA_M+1) x dim: past iterates (before CD pass) */
    double *G_hist;       /* (AA_M+1) x dim: past G(x) results (after CD pass) */
    double *dF;           /* AA_M x dim: differences of residuals */
    double *gram;         /* AA_M x AA_M: Gram matrix */
    double *rhs;          /* AA_M: right-hand side */
    double *alpha;        /* AA_M: mixing weights */
} aa_state_t;

static aa_state_t *aa_create(int32_t p) {
    int32_t dim = p + 1;
    aa_state_t *aa = (aa_state_t *)calloc(1, sizeof(aa_state_t));
    if (!aa) return NULL;
    aa->dim = dim;
    aa->count = 0;
    aa->X_hist = (double *)calloc((size_t)(AA_M + 1) * (size_t)dim, sizeof(double));
    aa->G_hist = (double *)calloc((size_t)(AA_M + 1) * (size_t)dim, sizeof(double));
    aa->dF     = (double *)calloc((size_t)AA_M * (size_t)dim, sizeof(double));
    aa->gram   = (double *)calloc((size_t)AA_M * (size_t)AA_M, sizeof(double));
    aa->rhs    = (double *)calloc((size_t)AA_M, sizeof(double));
    aa->alpha  = (double *)calloc((size_t)AA_M, sizeof(double));
    if (!aa->X_hist || !aa->G_hist || !aa->dF || !aa->gram || !aa->rhs || !aa->alpha) {
        free(aa->X_hist); free(aa->G_hist); free(aa->dF);
        free(aa->gram); free(aa->rhs); free(aa->alpha);
        free(aa); return NULL;
    }
    return aa;
}

static void aa_free(aa_state_t *aa) {
    if (!aa) return;
    free(aa->X_hist); free(aa->G_hist); free(aa->dF);
    free(aa->gram); free(aa->rhs); free(aa->alpha);
    free(aa);
}

/* (aa_reset reserved for future IRLS integration) */

/* Store iterate x (before CD) and G(x) (after CD) */
static void aa_store(aa_state_t *aa, const double *beta, double intercept,
                     const double *beta_after, double intercept_after, int32_t p) {
    int32_t slot = aa->count % (AA_M + 1);
    double *xh = aa->X_hist + (size_t)slot * (size_t)aa->dim;
    double *gh = aa->G_hist + (size_t)slot * (size_t)aa->dim;
    memcpy(xh, beta, (size_t)p * sizeof(double));
    xh[p] = intercept;
    memcpy(gh, beta_after, (size_t)p * sizeof(double));
    gh[p] = intercept_after;
    aa->count++;
}

/* Attempt Anderson extrapolation. Returns 1 if successful (result in beta_out, intercept_out).
 * Returns 0 if not enough history or solve fails. */
static int aa_extrapolate(aa_state_t *aa, double *beta_out, double *intercept_out, int32_t p) {
    int32_t m = aa->count - 1;  /* number of differences available */
    if (m < 2) return 0;
    if (m > AA_M) m = AA_M;

    int32_t dim = aa->dim;
    int32_t total = aa->count;

    /* Build residual differences: dF[k] = F[k+1] - F[k] where F[k] = G(x_k) - x_k */
    for (int32_t k = 0; k < m; k++) {
        int32_t idx1 = (total - m + k) % (AA_M + 1);
        int32_t idx0 = (total - m + k - 1 + (AA_M + 1)) % (AA_M + 1);
        double *f1_g = aa->G_hist + (size_t)idx1 * (size_t)dim;
        double *f1_x = aa->X_hist + (size_t)idx1 * (size_t)dim;
        double *f0_g = aa->G_hist + (size_t)idx0 * (size_t)dim;
        double *f0_x = aa->X_hist + (size_t)idx0 * (size_t)dim;
        double *df = aa->dF + (size_t)k * (size_t)dim;
        for (int32_t d = 0; d < dim; d++) {
            df[d] = (f1_g[d] - f1_x[d]) - (f0_g[d] - f0_x[d]);
        }
    }

    /* Build Gram matrix: gram[i][j] = dF[i] . dF[j] */
    for (int32_t i = 0; i < m; i++) {
        for (int32_t j = i; j < m; j++) {
            double dot = 0.0;
            double *di = aa->dF + (size_t)i * (size_t)dim;
            double *dj = aa->dF + (size_t)j * (size_t)dim;
            for (int32_t d = 0; d < dim; d++) dot += di[d] * dj[d];
            aa->gram[i * m + j] = dot;
            aa->gram[j * m + i] = dot;
        }
        /* Tikhonov regularization for numerical stability */
        aa->gram[i * m + i] += 1e-10;
    }

    /* RHS: rhs[i] = dF[i] . F[last] where F[last] = G(x_last) - x_last */
    int32_t last_idx = (total - 1) % (AA_M + 1);
    double *last_g = aa->G_hist + (size_t)last_idx * (size_t)dim;
    double *last_x = aa->X_hist + (size_t)last_idx * (size_t)dim;
    for (int32_t i = 0; i < m; i++) {
        double dot = 0.0;
        double *di = aa->dF + (size_t)i * (size_t)dim;
        for (int32_t d = 0; d < dim; d++) {
            dot += di[d] * (last_g[d] - last_x[d]);
        }
        aa->rhs[i] = dot;
    }

    /* Solve gram * c = rhs via Cholesky (m is tiny, <= AA_M=5) */
    /* In-place Cholesky: L stored in lower triangle of gram */
    double L[AA_M * AA_M];
    memcpy(L, aa->gram, (size_t)(m * m) * sizeof(double));
    for (int32_t i = 0; i < m; i++) {
        for (int32_t j = 0; j <= i; j++) {
            double s = L[i * m + j];
            for (int32_t k = 0; k < j; k++) s -= L[i * m + k] * L[j * m + k];
            if (i == j) {
                if (s <= 0.0) return 0;  /* not positive definite */
                L[i * m + j] = sqrt(s);
            } else {
                L[i * m + j] = s / L[j * m + j];
            }
        }
    }
    /* Forward solve: L * z = rhs */
    double z[AA_M];
    for (int32_t i = 0; i < m; i++) {
        double s = aa->rhs[i];
        for (int32_t j = 0; j < i; j++) s -= L[i * m + j] * z[j];
        z[i] = s / L[i * m + i];
    }
    /* Backward solve: L^T * c = z */
    double c[AA_M];
    for (int32_t i = m - 1; i >= 0; i--) {
        double s = z[i];
        for (int32_t j = i + 1; j < m; j++) s -= L[j * m + i] * c[j];
        c[i] = s / L[i * m + i];
    }

    /* Extrapolate: x_new = G(x_last) - sum(c[k] * dF[k])
     * This is equivalent to the AA-I update from Bertrand & Massias */
    for (int32_t d = 0; d < dim; d++) {
        double val = last_g[d];
        for (int32_t k = 0; k < m; k++) {
            val -= c[k] * aa->dF[k * dim + d];
        }
        if (d < p) beta_out[d] = val;
        else *intercept_out = val;
    }

    return 1;
}

/* Compute penalized objective for Gaussian + identity:
 * 0.5 * sum(r^2 * w) / n + penalty(beta) */
static double gaussian_objective(const cd_work_t *w, double lambda) {
    double obj = 0.0;
    for (int32_t i = 0; i < w->n; i++) {
        obj += w->r[i] * w->r[i] * w->w[i];
    }
    obj /= (2.0 * w->n);

    /* Add penalty */
    for (int32_t j = 0; j < w->p; j++) {
        double b = fabs(w->beta[j]);
        if (b == 0.0) continue;
        double pf = w->pf[j];
        if (pf == 0.0) continue;
        double l1 = lambda * w->alpha * pf;
        double l2 = lambda * (1.0 - w->alpha) * pf;
        switch (w->penalty) {
            case GAM_PENALTY_L1:
                obj += l1 * b;
                break;
            case GAM_PENALTY_L2:
                obj += 0.5 * lambda * pf * b * b;
                break;
            case GAM_PENALTY_ELASTICNET:
                obj += l1 * b + 0.5 * l2 * b * b;
                break;
            case GAM_PENALTY_MCP:
                if (b <= w->gamma_mcp * l1)
                    obj += l1 * b - b * b / (2.0 * w->gamma_mcp);
                else
                    obj += 0.5 * w->gamma_mcp * l1 * l1;
                break;
            case GAM_PENALTY_SCAD:
                if (b <= l1)
                    obj += l1 * b;
                else if (b <= w->gamma_scad * l1)
                    obj += (2.0 * w->gamma_scad * l1 * b - b * b - l1 * l1) /
                           (2.0 * (w->gamma_scad - 1.0));
                else
                    obj += l1 * l1 * (w->gamma_scad + 1.0) / 2.0;
                break;
            default:
                break;
        }
    }
    return obj;
}

/* Compute Huber loss: (1/n) * sum(rho_gamma(r_i)) + penalty */
static double huber_objective(const cd_work_t *w, double lambda) {
    double gamma = w->huber_gamma;
    double obj = 0.0;
    for (int32_t i = 0; i < w->n; i++) {
        double ri = w->r[i];
        double ari = fabs(ri);
        if (ari <= gamma)
            obj += 0.5 * ri * ri;
        else
            obj += gamma * ari - 0.5 * gamma * gamma;
    }
    obj /= w->n;
    /* Add penalty (same as gaussian_objective) */
    for (int32_t j = 0; j < w->p; j++) {
        double b = fabs(w->beta[j]);
        if (b == 0.0) continue;
        double pf = w->pf[j];
        if (pf == 0.0) continue;
        double l1 = lambda * w->alpha * pf;
        double l2 = lambda * (1.0 - w->alpha) * pf;
        switch (w->penalty) {
            case GAM_PENALTY_L1:
            case GAM_PENALTY_ELASTICNET:
                obj += l1 * b + 0.5 * l2 * b * b;
                break;
            case GAM_PENALTY_L2:
                obj += 0.5 * lambda * pf * b * b;
                break;
            default:
                obj += l1 * b + 0.5 * l2 * b * b;
                break;
        }
    }
    return obj;
}

/* Compute quantile check loss: (1/n) * sum(rho_tau(r_i)) + penalty */
static double quantile_objective(const cd_work_t *w, double lambda) {
    double tau = w->quantile_tau;
    double obj = 0.0;
    for (int32_t i = 0; i < w->n; i++) {
        double ri = w->r[i];
        obj += ri * (tau - (ri < 0 ? 1.0 : 0.0));
    }
    obj /= w->n;
    /* Add penalty */
    for (int32_t j = 0; j < w->p; j++) {
        double b = fabs(w->beta[j]);
        if (b == 0.0) continue;
        double pf = w->pf[j];
        if (pf == 0.0) continue;
        double l1 = lambda * w->alpha * pf;
        double l2 = lambda * (1.0 - w->alpha) * pf;
        switch (w->penalty) {
            case GAM_PENALTY_L1:
            case GAM_PENALTY_ELASTICNET:
                obj += l1 * b + 0.5 * l2 * b * b;
                break;
            case GAM_PENALTY_L2:
                obj += 0.5 * lambda * pf * b * b;
                break;
            default:
                obj += l1 * b + 0.5 * l2 * b * b;
                break;
        }
    }
    return obj;
}

/* Update IRLS weights for Huber/quantile from current residuals */
static void update_robust_weights(cd_work_t *w) {
    if (w->family == GAM_FAMILY_HUBER) {
        double gamma = w->huber_gamma;
        for (int32_t i = 0; i < w->n; i++) {
            double ari = fabs(w->r[i]);
            double wi = (ari <= gamma) ? 1.0 : gamma / fmax(ari, 1e-15);
            if (w->sample_weight) wi *= w->sample_weight[i];
            w->w[i] = fmax(wi, 1e-10);
        }
    } else {
        /* Quantile */
        double tau = w->quantile_tau;
        for (int32_t i = 0; i < w->n; i++) {
            double ri = w->r[i];
            double ari = fmax(fabs(ri), 1e-6);
            double wi = (ri >= 0 ? tau : (1.0 - tau)) / ari;
            if (w->sample_weight) wi *= w->sample_weight[i];
            w->w[i] = fmax(wi, 1e-10);
        }
    }
    /* Update xw_sq with new weights */
    for (int32_t j = 0; j < w->p; j++) {
        double s = 0.0;
        for (int32_t i = 0; i < w->n; i++) {
            double xij = w->Xs[i * w->p + j];
            s += xij * xij * w->w[i];
        }
        w->xw_sq[j] = s / w->n;
    }
}

/* Recompute residuals from scratch: r = y - X*beta - intercept */
static void recompute_residuals(cd_work_t *w) {
    for (int32_t i = 0; i < w->n; i++) {
        double fitted = w->intercept;
        for (int32_t j = 0; j < w->p; j++) {
            fitted += w->Xs[i * w->p + j] * w->beta[j];
        }
        w->r[i] = w->y[i] - fitted;
    }
}

/* ========== Fused Lasso (ADMM) ========== */
/* Solves: min 0.5/n ||y - X*beta||^2 + alpha*lambda*||beta||_1
 *              + (1-alpha)*lambda*||D^k * beta||_1
 * where D^k is the k-th order difference operator.
 * Uses ADMM: D*beta = z, with augmented Lagrangian parameter rho. */

/* Apply k-th order difference operator: out = D^k * beta.
 * For order 1: out[j] = beta[j+1] - beta[j], length p-1.
 * For order 2: out[j] = beta[j+2] - 2*beta[j+1] + beta[j], length p-2. */
static void diff_operator(const double *beta, int32_t p, int32_t order, double *out) {
    if (order == 1) {
        for (int32_t j = 0; j < p - 1; j++) {
            out[j] = beta[j + 1] - beta[j];
        }
    } else if (order == 2) {
        for (int32_t j = 0; j < p - 2; j++) {
            out[j] = beta[j + 2] - 2.0 * beta[j + 1] + beta[j];
        }
    }
}

/* Apply transpose of difference operator: out = D^T * v.
 * For order 1: D^T is (p) x (p-1), out[j] sums contributions from adjacent diffs. */
static void diff_operator_T(const double *v, int32_t p, int32_t order, double *out) {
    memset(out, 0, (size_t)p * sizeof(double));
    if (order == 1) {
        for (int32_t j = 0; j < p - 1; j++) {
            out[j]     -= v[j];
            out[j + 1] += v[j];
        }
    } else if (order == 2) {
        for (int32_t j = 0; j < p - 2; j++) {
            out[j]     += v[j];
            out[j + 1] -= 2.0 * v[j];
            out[j + 2] += v[j];
        }
    }
}

/* Fused lasso / trend filtering via ADMM for Gaussian + identity.
 * w->alpha controls L1 vs TV mix: alpha*lambda for L1, (1-alpha)*lambda for TV. */
static int fused_lasso_admm(
    cd_work_t *w, double lambda, int32_t order,
    int32_t max_iter, int32_t *out_iter
) {
    int32_t n = w->n, p = w->p;
    int32_t m = (order == 1) ? p - 1 : (order == 2) ? p - 2 : p - 1;
    if (m <= 0) { *out_iter = 0; return 0; }

    double l1_pen = w->alpha * lambda;
    double tv_pen = (1.0 - w->alpha) * lambda;
    double rho = tv_pen + 1.0;  /* ADMM augmented Lagrangian parameter */

    /* Allocate ADMM variables */
    double *z = (double *)calloc((size_t)m, sizeof(double));    /* consensus: D*beta = z */
    double *u = (double *)calloc((size_t)m, sizeof(double));    /* scaled dual */
    double *Dbeta = (double *)calloc((size_t)m, sizeof(double));
    double *Dtu = (double *)calloc((size_t)p, sizeof(double));
    if (!z || !u || !Dbeta || !Dtu) {
        free(z); free(u); free(Dbeta); free(Dtu);
        return -1;
    }

    int32_t iter;
    for (iter = 0; iter < max_iter; iter++) {
        /* Beta update: solve (X^T*W*X/n + rho*D^T*D) * beta = X^T*W*y/n + rho*D^T*(z - u)
         * We use coordinate descent on the augmented objective:
         * 0.5/n * ||y - X*beta||^2_W + alpha*lambda*||beta||_1
         *   + rho/2 * ||D*beta - z + u||^2 */

        /* Compute D^T*(z - u) for the augmented term */
        double *zmu = Dbeta;  /* reuse buffer temporarily */
        for (int32_t j = 0; j < m; j++) zmu[j] = z[j] - u[j];
        diff_operator_T(zmu, p, order, Dtu);

        /* CD pass for beta update */
        for (int32_t j = 0; j < p; j++) {
            double old_beta = w->beta[j];

            /* Gradient from data: (1/n) * sum(x_j * r * w) + xw_sq * beta_j */
            double grad_data = 0.0;
            for (int32_t i = 0; i < n; i++) {
                grad_data += w->Xs[i * p + j] * w->r[i] * w->w[i];
            }
            grad_data = grad_data / n + w->xw_sq[j] * old_beta;

            /* Gradient from ADMM augmented term: rho * (D^T*D*beta - D^T*(z-u))
             * We need (D^T*D*beta)_j. For order=1: D^T*D is tridiagonal.
             * Instead of forming D^T*D explicitly, use the fact that the CD update
             * for the full objective is:
             * z_j = grad_data + rho * Dtu_j  (augmented gradient)
             * v_j = xw_sq_j + rho * (D^T*D)_{jj}  (augmented Hessian diagonal) */
            double dtd_jj;
            if (order == 1) {
                dtd_jj = (j == 0 || j == p - 1) ? 1.0 : 2.0;
            } else {
                /* order 2: D^T*D pentadiagonal */
                if (j == 0 || j == p - 1)      dtd_jj = 1.0;
                else if (j == 1 || j == p - 2) dtd_jj = 5.0;
                else                            dtd_jj = 6.0;
            }

            double z_j = grad_data + rho * Dtu[j];
            double v_j = w->xw_sq[j] + rho * dtd_jj;

            /* But we need the off-diagonal augmented contributions too.
             * Actually, for a single CD pass, the residuals already track X*beta.
             * The tricky part is the D^T*D*beta term. Let me reconsider.
             *
             * Full gradient for feature j:
             * = data_grad + rho * ((D^T*D*beta)_j - (D^T*(z-u))_j)
             *
             * For order 1, (D^T*D*beta)_j depends on beta[j-1], beta[j], beta[j+1].
             * We can compute it from current beta values. */

            double dtdbeta_j = 0.0;
            if (order == 1) {
                if (j > 0)     dtdbeta_j += w->beta[j] - w->beta[j - 1];
                if (j < p - 1) dtdbeta_j += w->beta[j] - w->beta[j + 1];
            } else {
                /* order 2: more complex, skip for now */
                if (j >= 2)     dtdbeta_j += w->beta[j] - 2.0 * w->beta[j - 1] + w->beta[j - 2];
                if (j >= 1 && j < p - 1)
                    dtdbeta_j += -2.0 * (w->beta[j + 1] - 2.0 * w->beta[j] + w->beta[j - 1]);
                if (j < p - 2) dtdbeta_j += w->beta[j + 2] - 2.0 * w->beta[j + 1] + w->beta[j];
            }

            /* Actually, let me simplify. Use the standard ADMM approach:
             * For the beta update, we minimize:
             *   0.5/n * ||y - X*beta||^2_W + alpha*lambda*||beta||_1
             *     + rho/2 * ||D*beta - (z - u)||^2
             *
             * The coordinate descent update for beta_j is:
             *   z_j = (1/n)*sum(x_j * r * w) + xw_sq_j * beta_j   [data part]
             *          + rho * (Dtu_j - dtdbeta_j + dtd_jj * beta_j) [ADMM part]
             *   v_j = xw_sq_j + rho * dtd_jj
             *   beta_j = soft_threshold(z_j, l1_pen) / v_j  [if penalized]
             */
            z_j = grad_data + rho * (Dtu[j] - dtdbeta_j + dtd_jj * old_beta);
            v_j = w->xw_sq[j] + rho * dtd_jj;

            double new_beta;
            if (w->pf[j] == 0.0) {
                new_beta = z_j / v_j;
            } else {
                new_beta = soft_threshold(z_j, l1_pen * w->pf[j]) / v_j;
            }
            new_beta = clamp(new_beta, w->lb[j], w->ub[j]);

            if (new_beta != old_beta) {
                double delta = new_beta - old_beta;
                w->beta[j] = new_beta;
                for (int32_t i = 0; i < n; i++) {
                    w->r[i] -= delta * w->Xs[i * p + j];
                }
            }
        }

        /* Intercept update */
        if (w->fit_intercept) {
            double sum_wr = 0.0, sum_w = 0.0;
            for (int32_t i = 0; i < n; i++) {
                sum_wr += w->r[i] * w->w[i];
                sum_w += w->w[i];
            }
            double delta = sum_wr / fmax(sum_w, 1e-15);
            if (fabs(delta) > 1e-15) {
                w->intercept += delta;
                for (int32_t i = 0; i < n; i++) w->r[i] -= delta;
            }
        }

        /* Z update: z = soft_threshold(D*beta + u, tv_pen/rho) */
        diff_operator(w->beta, p, order, Dbeta);
        for (int32_t j = 0; j < m; j++) {
            z[j] = soft_threshold(Dbeta[j] + u[j], tv_pen / rho);
        }

        /* Dual update: u = u + D*beta - z */
        double primal_res = 0.0;
        for (int32_t j = 0; j < m; j++) {
            double r_j = Dbeta[j] - z[j];
            primal_res += r_j * r_j;
            u[j] += r_j;
        }
        primal_res = sqrt(primal_res);

        /* Check ADMM convergence */
        if (primal_res < w->tol * sqrt((double)m) && iter > 2) break;
    }

    *out_iter = iter;
    free(z); free(u); free(Dbeta); free(Dtu);
    return 0;
}

/* Fit at a single lambda (Gaussian: pure CD, non-Gaussian: IRLS + CD) */
static int fit_single_lambda(
    cd_work_t *w, double lambda, double lambda_prev,
    int32_t max_iter, int32_t max_inner, int32_t screening,
    int32_t *out_iter
) {
    int32_t iter = 0;

    if ((w->penalty == GAM_PENALTY_GROUP_L1 || w->penalty == GAM_PENALTY_SGL) && w->groups) {
        /* Group Lasso / Sparse Group Lasso: block coordinate descent */
        /* Works for both Gaussian and non-Gaussian via IRLS outer loop */
        int32_t n_outer = (w->family == GAM_FAMILY_GAUSSIAN && w->link == GAM_LINK_IDENTITY) ? 1 : max_inner;

        for (int32_t outer = 0; outer < n_outer; outer++) {
            if (w->family != GAM_FAMILY_GAUSSIAN || w->link != GAM_LINK_IDENTITY) {
                /* IRLS: update working response and weights */
                for (int32_t i = 0; i < w->n; i++) {
                    double eta_i = w->intercept;
                    if (w->offset) eta_i += w->offset[i];
                    for (int32_t j = 0; j < w->p; j++) {
                        eta_i += w->Xs[i * w->p + j] * w->beta[j];
                    }
                    w->eta[i] = eta_i;
                    w->mu[i] = linkinv_fn(w->link, eta_i);
                    double dmu = dmu_deta(w->link, eta_i);
                    double var = variance_fn(w->family, w->mu[i], w->tweedie_p, w->nb_theta);
                    double wi = dmu * dmu / var;
                    if (w->sample_weight) wi *= w->sample_weight[i];
                    w->w[i] = fmax(wi, 1e-10);
                    double z_i = eta_i + (w->y[i] - w->mu[i]) / fmax(dmu, 1e-10);
                    if (w->offset) z_i -= w->offset[i];
                    double fitted = w->intercept;
                    for (int32_t j = 0; j < w->p; j++) {
                        fitted += w->Xs[i * w->p + j] * w->beta[j];
                    }
                    w->r[i] = z_i - fitted;
                }
                for (int32_t j = 0; j < w->p; j++) {
                    double s = 0.0;
                    for (int32_t i = 0; i < w->n; i++) {
                        double xij = w->Xs[i * w->p + j];
                        s += xij * xij * w->w[i];
                    }
                    w->xw_sq[j] = s / w->n;
                }
            }

            for (int32_t inner = 0; inner < max_iter; inner++) {
                int changed = group_cd_pass(w, lambda, 0);
                iter++;
                if (!changed) break;
            }

            if (n_outer == 1) break;

            /* IRLS convergence */
            for (int32_t i = 0; i < w->n; i++) {
                double eta_i = w->intercept;
                if (w->offset) eta_i += w->offset[i];
                for (int32_t j = 0; j < w->p; j++) {
                    eta_i += w->Xs[i * w->p + j] * w->beta[j];
                }
                w->eta[i] = eta_i;
                w->mu[i] = linkinv_fn(w->link, eta_i);
            }
            double cur_dev = 0.0;
            for (int32_t i = 0; i < w->n; i++) {
                cur_dev += deviance_unit(w->family, w->y[i], w->mu[i],
                                         w->tweedie_p, w->nb_theta);
            }
            if (outer > 0 && cur_dev < w->tol) break;
        }
    } else if (w->penalty == GAM_PENALTY_FUSED) {
        /* Fused lasso / trend filtering via ADMM */
        /* Currently Gaussian + identity only */
        if (w->family != GAM_FAMILY_GAUSSIAN || w->link != GAM_LINK_IDENTITY) {
            gam_set_error("Fused lasso currently supports Gaussian family only");
            return -1;
        }
        int rc = fused_lasso_admm(w, lambda, w->fused_order, max_iter, &iter);
        if (rc < 0) return rc;
    } else if (w->penalty == GAM_PENALTY_SLOPE && w->slope_seq) {
        /* SLOPE: proximal gradient descent (penalty couples all coordinates) */
        /* Works for both Gaussian and non-Gaussian via IRLS outer loop */
        int32_t n_outer = (w->family == GAM_FAMILY_GAUSSIAN && w->link == GAM_LINK_IDENTITY) ? 1 : max_inner;

        for (int32_t outer = 0; outer < n_outer; outer++) {
            if (w->family != GAM_FAMILY_GAUSSIAN || w->link != GAM_LINK_IDENTITY) {
                /* IRLS: update working response and weights */
                for (int32_t i = 0; i < w->n; i++) {
                    double eta_i = w->intercept;
                    if (w->offset) eta_i += w->offset[i];
                    for (int32_t j = 0; j < w->p; j++) {
                        eta_i += w->Xs[i * w->p + j] * w->beta[j];
                    }
                    w->eta[i] = eta_i;
                    w->mu[i] = linkinv_fn(w->link, eta_i);
                    double dmu = dmu_deta(w->link, eta_i);
                    double var = variance_fn(w->family, w->mu[i], w->tweedie_p, w->nb_theta);
                    double wi = dmu * dmu / var;
                    if (w->sample_weight) wi *= w->sample_weight[i];
                    w->w[i] = fmax(wi, 1e-10);
                    double z_i = eta_i + (w->y[i] - w->mu[i]) / fmax(dmu, 1e-10);
                    if (w->offset) z_i -= w->offset[i];
                    double fitted = w->intercept;
                    for (int32_t j = 0; j < w->p; j++) {
                        fitted += w->Xs[i * w->p + j] * w->beta[j];
                    }
                    w->r[i] = z_i - fitted;
                }
                /* Update weighted x^2 sums */
                for (int32_t j = 0; j < w->p; j++) {
                    double s = 0.0;
                    for (int32_t i = 0; i < w->n; i++) {
                        double xij = w->Xs[i * w->p + j];
                        s += xij * xij * w->w[i];
                    }
                    w->xw_sq[j] = s / w->n;
                }
            }

            /* Inner PGD loop */
            double prev_obj = 1e30;
            for (int32_t inner = 0; inner < max_iter; inner++) {
                int changed = slope_pgd_pass(w, lambda);
                iter++;

                /* Check convergence: objective decrease */
                double obj = 0.0;
                for (int32_t i = 0; i < w->n; i++) {
                    obj += w->r[i] * w->r[i] * w->w[i];
                }
                obj /= (2.0 * w->n);

                if (fabs(prev_obj - obj) / (fabs(obj) + 1e-10) < w->tol) break;
                if (!changed) break;
                prev_obj = obj;
            }

            if (n_outer == 1) break;  /* Gaussian: single outer iteration */

            /* IRLS convergence check */
            for (int32_t i = 0; i < w->n; i++) {
                double eta_i = w->intercept;
                if (w->offset) eta_i += w->offset[i];
                for (int32_t j = 0; j < w->p; j++) {
                    eta_i += w->Xs[i * w->p + j] * w->beta[j];
                }
                w->eta[i] = eta_i;
                w->mu[i] = linkinv_fn(w->link, eta_i);
            }
            double cur_dev = 0.0;
            for (int32_t i = 0; i < w->n; i++) {
                cur_dev += deviance_unit(w->family, w->y[i], w->mu[i],
                                         w->tweedie_p, w->nb_theta);
            }
            /* Simple convergence: absolute deviance is small enough */
            if (outer > 0 && cur_dev < w->tol) break;
        }
    } else if (w->family == GAM_FAMILY_HUBER || w->family == GAM_FAMILY_QUANTILE) {
        /* Robust regression via IRLS with Huber/quantile weights.
         * Outer: recompute weights from residuals. Inner: standard CD on weighted LS.
         * Ref: Yi & Huang (2017), "Semismooth Newton CD for Elastic-Net Penalized
         *      Huber Loss and Quantile Regression." */
        if (screening && lambda_prev > 0) {
            strong_screen(w, lambda, lambda_prev);
        } else {
            for (int32_t j = 0; j < w->p; j++) w->active[j] = 1;
        }

        double prev_obj = 1e30;
        for (int32_t outer = 0; outer < max_inner; outer++) {
            /* Update IRLS weights from current residuals */
            update_robust_weights(w);

            /* Inner CD loop on the weighted LS subproblem */
            for (int32_t inner = 0; inner < max_iter / max_inner; inner++) {
                double change = cd_pass(w, lambda, 1);
                iter++;
                if (change < w->tol) {
                    double full_change = cd_pass(w, lambda, 0);
                    iter++;
                    if (screening) kkt_check(w, lambda);
                    if (full_change < w->tol) break;
                }
            }

            /* Check convergence via objective change */
            double cur_obj = (w->family == GAM_FAMILY_HUBER)
                ? huber_objective(w, lambda)
                : quantile_objective(w, lambda);
            if (outer > 0 && fabs(prev_obj - cur_obj) / (fabs(cur_obj) + 0.1) < w->tol) break;
            prev_obj = cur_obj;
        }
    } else if (w->family == GAM_FAMILY_GAUSSIAN && w->link == GAM_LINK_IDENTITY) {
        /* Pure coordinate descent for Gaussian + identity, with Anderson acceleration
         * and GAP Safe screening */
        if (screening && lambda_prev > 0) {
            strong_screen(w, lambda, lambda_prev);
            /* Tighten with GAP Safe if available */
            gap_safe_screen(w, lambda);
        } else {
            for (int32_t j = 0; j < w->p; j++) w->active[j] = 1;
        }

        /* Anderson acceleration state */
        aa_state_t *aa = (w->penalty != GAM_PENALTY_SLOPE &&
                          w->penalty != GAM_PENALTY_GROUP_L1 &&
                          w->penalty != GAM_PENALTY_SGL) ? aa_create(w->p) : NULL;
        double *beta_save = aa ? (double *)malloc((size_t)w->p * sizeof(double)) : NULL;
        double *beta_aa   = aa ? (double *)malloc((size_t)w->p * sizeof(double)) : NULL;

        for (iter = 0; iter < max_iter; iter++) {
            /* Save state before CD pass (for AA) */
            double intercept_before = w->intercept;
            if (aa && (iter % AA_K == 0)) {
                memcpy(beta_save, w->beta, (size_t)w->p * sizeof(double));
            }

            /* Active set pass */
            double change = cd_pass(w, lambda, 1);

            /* Store AA iterate: (x_before, G(x)_after) */
            if (aa && (iter % AA_K == 0)) {
                aa_store(aa, beta_save, intercept_before, w->beta, w->intercept, w->p);
            }

            /* Attempt Anderson extrapolation every AA_K iterations */
            if (aa && beta_aa && (iter % AA_K == AA_K - 1) && iter >= 2 * AA_K) {
                double intercept_aa = 0.0;
                if (aa_extrapolate(aa, beta_aa, &intercept_aa, w->p)) {
                    /* Safeguard: accept only if objective decreases */
                    double obj_before = gaussian_objective(w, lambda);
                    /* Save current state */
                    double *beta_cur = beta_save;  /* reuse buffer */
                    memcpy(beta_cur, w->beta, (size_t)w->p * sizeof(double));
                    double intercept_cur = w->intercept;
                    /* Try AA iterate */
                    memcpy(w->beta, beta_aa, (size_t)w->p * sizeof(double));
                    w->intercept = intercept_aa;
                    recompute_residuals(w);
                    double obj_after = gaussian_objective(w, lambda);
                    if (obj_after < obj_before) {
                        /* Accept: update active set */
                        for (int32_t j = 0; j < w->p; j++) {
                            if (w->beta[j] != 0.0) {
                                w->active[j] = 1;
                                w->ever_active[j] = 1;
                            }
                        }
                    } else {
                        /* Reject: restore */
                        memcpy(w->beta, beta_cur, (size_t)w->p * sizeof(double));
                        w->intercept = intercept_cur;
                        recompute_residuals(w);
                    }
                }
            }

            /* Convergence: max weighted coordinate change < tol */
            if (change < w->tol) {
                /* Full pass to check all coordinates */
                double full_change = cd_pass(w, lambda, 0);
                if (screening && !kkt_check(w, lambda)) {
                    continue;  /* KKT violated, re-iterate */
                }
                if (full_change < w->tol) break;
            } else if (iter % 3 == 2) {
                /* Periodic full pass + GAP Safe re-screening */
                cd_pass(w, lambda, 0);
                if (screening && !kkt_check(w, lambda)) {
                    continue;
                }
                /* Tighten active set with GAP Safe */
                if (screening) gap_safe_screen(w, lambda);
            }
        }
        aa_free(aa);
        free(beta_save);
        free(beta_aa);
    } else {
        /* Non-Gaussian: IRLS (proximal Newton) */
        /* Initialize eta, mu from current beta */
        for (int32_t i = 0; i < w->n; i++) {
            double eta_i = w->intercept;
            if (w->offset) eta_i += w->offset[i];
            for (int32_t j = 0; j < w->p; j++) {
                eta_i += w->Xs[i * w->p + j] * w->beta[j];
            }
            w->eta[i] = eta_i;
            w->mu[i] = linkinv_fn(w->link, eta_i);
        }

        for (int32_t outer = 0; outer < max_inner; outer++) {
            /* Compute working response and weights */
            for (int32_t i = 0; i < w->n; i++) {
                double dmu = dmu_deta(w->link, w->eta[i]);
                double var = variance_fn(w->family, w->mu[i], w->tweedie_p, w->nb_theta);
                double wi = dmu * dmu / var;
                if (w->sample_weight) wi *= w->sample_weight[i];
                w->w[i] = fmax(wi, 1e-10);

                /* Working response: z = eta + (y - mu) / dmu */
                double z_i = w->eta[i] + (w->y[i] - w->mu[i]) / fmax(dmu, 1e-10);
                if (w->offset) z_i -= w->offset[i];

                /* Residual = z - X*beta - intercept */
                double fitted = w->intercept;
                for (int32_t j = 0; j < w->p; j++) {
                    fitted += w->Xs[i * w->p + j] * w->beta[j];
                }
                w->r[i] = z_i - fitted;
            }

            /* Update weighted x^2 sums */
            for (int32_t j = 0; j < w->p; j++) {
                double s = 0.0;
                for (int32_t i = 0; i < w->n; i++) {
                    double xij = w->Xs[i * w->p + j];
                    s += xij * xij * w->w[i];
                }
                w->xw_sq[j] = s / w->n;
            }

            /* Run CD passes on the weighted LS subproblem */
            if (screening && lambda_prev > 0) {
                strong_screen(w, lambda, lambda_prev);
            } else {
                for (int32_t j = 0; j < w->p; j++) w->active[j] = 1;
            }

            double prev_dev = 0.0;
            for (int32_t i = 0; i < w->n; i++) {
                prev_dev += deviance_unit(w->family, w->y[i], w->mu[i],
                                          w->tweedie_p, w->nb_theta);
            }

            for (int32_t inner = 0; inner < max_iter / max_inner; inner++) {
                double change = cd_pass(w, lambda, 1);
                if (change < w->tol) {
                    cd_pass(w, lambda, 0);
                    if (screening) kkt_check(w, lambda);
                    break;
                }
                iter++;
            }

            /* Update eta and mu */
            for (int32_t i = 0; i < w->n; i++) {
                double eta_i = w->intercept;
                if (w->offset) eta_i += w->offset[i];
                for (int32_t j = 0; j < w->p; j++) {
                    eta_i += w->Xs[i * w->p + j] * w->beta[j];
                }
                w->eta[i] = eta_i;
                w->mu[i] = linkinv_fn(w->link, eta_i);
            }

            /* Check IRLS convergence via deviance change */
            double cur_dev = 0.0;
            for (int32_t i = 0; i < w->n; i++) {
                cur_dev += deviance_unit(w->family, w->y[i], w->mu[i],
                                         w->tweedie_p, w->nb_theta);
            }
            if (fabs(cur_dev - prev_dev) / (fabs(cur_dev) + 0.1) < w->tol) break;
        }
    }

    *out_iter = iter;
    return 0;
}

/* Compute deviance for current fit */
static double compute_deviance(const cd_work_t *w) {
    double dev = 0.0;
    for (int32_t i = 0; i < w->n; i++) {
        double eta_i = w->intercept;
        if (w->offset) eta_i += w->offset[i];
        for (int32_t j = 0; j < w->p; j++) {
            eta_i += w->Xs[i * w->p + j] * w->beta[j];
        }
        double mu = linkinv_fn(w->link, eta_i);
        double wi = w->sample_weight ? w->sample_weight[i] : 1.0;
        dev += wi * deviance_unit(w->family, w->y[i], mu, w->tweedie_p, w->nb_theta);
    }
    return dev;
}

/* ========== GAM: basis expansion ========== */

/* Build full knot vector from interior knots, needed for smoothness penalty.
 * Returns allocated array of length n_basis + degree + 1. */
static double *build_full_knots(
    const double *x, int32_t n,
    const double *knots, int32_t n_knots, int32_t degree,
    int32_t n_basis
) {
    int32_t n_full = n_basis + degree + 1;
    double *t = (double *)malloc((size_t)n_full * sizeof(double));
    if (!t) return NULL;

    double xmin = x[0], xmax = x[0];
    for (int32_t i = 1; i < n; i++) {
        if (x[i] < xmin) xmin = x[i];
        if (x[i] > xmax) xmax = x[i];
    }

    for (int32_t i = 0; i <= degree; i++) t[i] = xmin;
    for (int32_t i = 0; i < n_knots; i++) t[degree + 1 + i] = knots[i];
    for (int32_t i = 0; i <= degree; i++) t[n_knots + degree + 1 + i] = xmax;
    return t;
}

/* Expand X with B-spline bases for smooth and tensor product terms.
 * Returns new matrix (caller frees) and sets *out_p to new column count.
 * basis_map[j] = original feature index for column j.
 * Optionally builds penalty blocks for smoothness penalties.
 * out_penalty_blocks and out_n_blocks can be NULL if not needed. */
static double *expand_basis(
    const double *X, int32_t n, int32_t p,
    const gam_smooth_t *smooths, int32_t n_smooths,
    const gam_tensor_t *tensors, int32_t n_tensors,
    int32_t *out_p, int32_t **out_basis_map,
    penalty_block_t **out_penalty_blocks, int32_t *out_n_blocks
) {
    if (n_smooths == 0 && n_tensors == 0) {
        /* No smooth terms: just copy X */
        double *Xout = (double *)malloc((size_t)n * (size_t)p * sizeof(double));
        if (!Xout) return NULL;
        memcpy(Xout, X, (size_t)n * (size_t)p * sizeof(double));
        *out_p = p;
        int32_t *bmap = (int32_t *)malloc((size_t)p * sizeof(int32_t));
        if (!bmap) { free(Xout); return NULL; }
        for (int32_t j = 0; j < p; j++) bmap[j] = j;
        *out_basis_map = bmap;
        if (out_penalty_blocks) { *out_penalty_blocks = NULL; *out_n_blocks = 0; }
        return Xout;
    }

    /* Determine which features have smooth terms */
    int32_t *is_smooth = (int32_t *)calloc((size_t)p, sizeof(int32_t));
    if (!is_smooth) return NULL;
    for (int32_t s = 0; s < n_smooths; s++) {
        if (smooths[s].feature >= 0 && smooths[s].feature < p)
            is_smooth[smooths[s].feature] = 1;
    }
    /* Also mark tensor product features as smooth (remove from linear) */
    for (int32_t t = 0; t < n_tensors; t++) {
        for (int32_t m = 0; m < tensors[t].n_margins && m < 4; m++) {
            int32_t f = tensors[t].features[m];
            if (f >= 0 && f < p) is_smooth[f] = 1;
        }
    }

    /* Count total output columns */
    int32_t total_cols = 0;
    for (int32_t j = 0; j < p; j++) {
        if (!is_smooth[j]) total_cols++;
    }

    /* Smooth term basis counts */
    int32_t *n_basis_per_smooth = n_smooths > 0
        ? (int32_t *)malloc((size_t)n_smooths * sizeof(int32_t)) : NULL;
    for (int32_t s = 0; s < n_smooths; s++) {
        int32_t nk = smooths[s].n_knots > 0 ? smooths[s].n_knots : 20;
        int32_t deg = smooths[s].degree > 0 ? smooths[s].degree : 3;
        n_basis_per_smooth[s] = nk + deg + 1;
        total_cols += n_basis_per_smooth[s];
    }

    /* Tensor product basis counts */
    int32_t *n_basis_per_tensor = n_tensors > 0
        ? (int32_t *)malloc((size_t)n_tensors * sizeof(int32_t)) : NULL;
    for (int32_t t = 0; t < n_tensors; t++) {
        int32_t nb_total = 1;
        for (int32_t m = 0; m < tensors[t].n_margins && m < 4; m++) {
            int32_t nk = tensors[t].n_knots[m] > 0 ? tensors[t].n_knots[m] : 5;
            int32_t deg = tensors[t].degree[m] > 0 ? tensors[t].degree[m] : 3;
            nb_total *= (nk + deg + 1);
        }
        n_basis_per_tensor[t] = nb_total;
        total_cols += nb_total;
    }

    /* Allocate output matrix and basis map */
    double *Xout = (double *)calloc((size_t)n * (size_t)total_cols, sizeof(double));
    int32_t *bmap = (int32_t *)malloc((size_t)total_cols * sizeof(int32_t));
    if (!Xout || !bmap) {
        free(Xout); free(bmap); free(is_smooth);
        free(n_basis_per_smooth); free(n_basis_per_tensor);
        return NULL;
    }

    /* Penalty blocks: at most n_smooths + n_tensors * max_margins */
    int32_t max_blocks = n_smooths + n_tensors * 4;
    penalty_block_t *pblocks = NULL;
    int32_t n_pblocks = 0;
    if (out_penalty_blocks && max_blocks > 0) {
        pblocks = (penalty_block_t *)calloc((size_t)max_blocks, sizeof(penalty_block_t));
    }

    int32_t col = 0;

    /* Copy linear terms */
    for (int32_t j = 0; j < p; j++) {
        if (is_smooth[j]) continue;
        for (int32_t i = 0; i < n; i++)
            Xout[i * total_cols + col] = X[i * p + j];
        bmap[col] = j;
        col++;
    }

    /* Compute basis expansions for univariate smooth terms */
    for (int32_t s = 0; s < n_smooths; s++) {
        int32_t feat = smooths[s].feature;
        int32_t nk = smooths[s].n_knots > 0 ? smooths[s].n_knots : 20;
        int32_t deg = smooths[s].degree > 0 ? smooths[s].degree : 3;
        int32_t nb = n_basis_per_smooth[s];

        double *xcol = (double *)malloc((size_t)n * sizeof(double));
        double *knots = (double *)malloc((size_t)nk * sizeof(double));
        double *basis = (double *)calloc((size_t)n * (size_t)nb, sizeof(double));
        if (!xcol || !knots || !basis) {
            free(xcol); free(knots); free(basis);
            free(Xout); free(bmap); free(is_smooth);
            free(n_basis_per_smooth); free(n_basis_per_tensor); free(pblocks);
            return NULL;
        }

        for (int32_t i = 0; i < n; i++) xcol[i] = X[i * p + feat];
        gam_quantile_knots(xcol, n, nk, knots);
        gam_bspline_basis(xcol, n, knots, nk, deg, basis);

        int32_t col_start = col;
        for (int32_t b = 0; b < nb; b++) {
            for (int32_t i = 0; i < n; i++)
                Xout[i * total_cols + col] = basis[i * nb + b];
            bmap[col] = feat;
            col++;
        }

        /* Build smoothness penalty block if lambda_smooth > 0 */
        if (smooths[s].lambda_smooth > 0.0 && pblocks) {
            double *full_knots = build_full_knots(xcol, n, knots, nk, deg, nb);
            if (full_knots) {
                double *S = (double *)calloc((size_t)nb * (size_t)nb, sizeof(double));
                if (S) {
                    gam_smoothness_penalty(nb, deg, full_knots, S);
                    pblocks[n_pblocks].col_start = col_start;
                    pblocks[n_pblocks].block_size = nb;
                    pblocks[n_pblocks].S = S;
                    pblocks[n_pblocks].lambda = smooths[s].lambda_smooth;
                    n_pblocks++;
                }
                free(full_knots);
            }
        }

        free(basis);
        free(knots);
        free(xcol);
    }

    /* Compute tensor product basis expansions */
    for (int32_t t = 0; t < n_tensors; t++) {
        int32_t nm = tensors[t].n_margins;
        if (nm < 2 || nm > 4) continue;

        /* For now, support 2D tensor products (most common case) */
        /* Higher dimensions: chain Kronecker products */
        int32_t nb_total = n_basis_per_tensor[t];
        int32_t col_start = col;

        /* Compute marginal bases and knots */
        int32_t nb_marginal[4] = {0};
        double *marginal_basis[4] = {NULL};
        double *marginal_knots[4] = {NULL};
        double *marginal_xcol[4] = {NULL};
        int32_t marginal_nk[4] = {0};
        int32_t marginal_deg[4] = {0};

        int ok = 1;
        for (int32_t m = 0; m < nm; m++) {
            int32_t f = tensors[t].features[m];
            int32_t nk = tensors[t].n_knots[m] > 0 ? tensors[t].n_knots[m] : 5;
            int32_t deg = tensors[t].degree[m] > 0 ? tensors[t].degree[m] : 3;
            int32_t nb = nk + deg + 1;
            marginal_nk[m] = nk;
            marginal_deg[m] = deg;
            nb_marginal[m] = nb;

            marginal_xcol[m] = (double *)malloc((size_t)n * sizeof(double));
            marginal_knots[m] = (double *)malloc((size_t)nk * sizeof(double));
            marginal_basis[m] = (double *)calloc((size_t)n * (size_t)nb, sizeof(double));
            if (!marginal_xcol[m] || !marginal_knots[m] || !marginal_basis[m]) {
                ok = 0; break;
            }

            for (int32_t i = 0; i < n; i++) marginal_xcol[m][i] = X[i * p + f];
            gam_quantile_knots(marginal_xcol[m], n, nk, marginal_knots[m]);
            gam_bspline_basis(marginal_xcol[m], n, marginal_knots[m], nk, deg, marginal_basis[m]);
        }

        if (ok) {
            /* Build tensor product via chained row-wise Kronecker products */
            /* Start with margin 0, successively Kronecker with margins 1, 2, ... */
            int32_t cur_nb = nb_marginal[0];
            double *cur_basis = (double *)malloc((size_t)n * (size_t)cur_nb * sizeof(double));
            if (cur_basis) {
                memcpy(cur_basis, marginal_basis[0], (size_t)n * (size_t)cur_nb * sizeof(double));

                for (int32_t m = 1; m < nm; m++) {
                    int32_t next_nb = cur_nb * nb_marginal[m];
                    double *next_basis = (double *)calloc((size_t)n * (size_t)next_nb, sizeof(double));
                    if (!next_basis) { free(cur_basis); cur_basis = NULL; break; }

                    for (int32_t i = 0; i < n; i++) {
                        for (int32_t j1 = 0; j1 < cur_nb; j1++) {
                            for (int32_t j2 = 0; j2 < nb_marginal[m]; j2++) {
                                next_basis[i * next_nb + j1 * nb_marginal[m] + j2] =
                                    cur_basis[i * cur_nb + j1] * marginal_basis[m][i * nb_marginal[m] + j2];
                            }
                        }
                    }

                    free(cur_basis);
                    cur_basis = next_basis;
                    cur_nb = next_nb;
                }

                if (cur_basis) {
                    /* Copy tensor columns to output */
                    for (int32_t b = 0; b < nb_total; b++) {
                        for (int32_t i = 0; i < n; i++)
                            Xout[i * total_cols + col] = cur_basis[i * nb_total + b];
                        bmap[col] = tensors[t].features[0];  /* map to first feature */
                        col++;
                    }
                    free(cur_basis);
                }
            }

            /* Build tensor product penalty blocks */
            if (pblocks) {
                for (int32_t m = 0; m < nm; m++) {
                    if (tensors[t].lambda_smooth[m] <= 0.0) continue;

                    int32_t nk = marginal_nk[m];
                    int32_t deg = marginal_deg[m];
                    int32_t nb = nb_marginal[m];

                    double *full_knots = build_full_knots(
                        marginal_xcol[m], n, marginal_knots[m], nk, deg, nb);
                    if (!full_knots) continue;

                    /* Compute marginal smoothness penalty */
                    double *Sm = (double *)calloc((size_t)nb * (size_t)nb, sizeof(double));
                    if (!Sm) { free(full_knots); continue; }
                    gam_smoothness_penalty(nb, deg, full_knots, Sm);
                    free(full_knots);

                    /* Build tensor penalty: S_m (x) I_other
                     * For margin m: P = I_{pre} (x) S_m (x) I_{post}
                     * where pre = product of nb for margins < m, post = product for margins > m */
                    int32_t pre = 1, post = 1;
                    for (int32_t mm = 0; mm < m; mm++) pre *= nb_marginal[mm];
                    for (int32_t mm = m + 1; mm < nm; mm++) post *= nb_marginal[mm];

                    double *S_tensor = (double *)calloc((size_t)nb_total * (size_t)nb_total, sizeof(double));
                    if (!S_tensor) { free(Sm); continue; }

                    /* I_pre (x) S_m (x) I_post */
                    for (int32_t i_pre = 0; i_pre < pre; i_pre++) {
                        for (int32_t j_pre = 0; j_pre < pre; j_pre++) {
                            if (i_pre != j_pre) continue;  /* I_pre is diagonal */
                            for (int32_t i_m = 0; i_m < nb; i_m++) {
                                for (int32_t j_m = 0; j_m < nb; j_m++) {
                                    double s_val = Sm[i_m * nb + j_m];
                                    if (s_val == 0.0) continue;
                                    for (int32_t i_post = 0; i_post < post; i_post++) {
                                        /* I_post is diagonal: j_post == i_post */
                                        int32_t row = i_pre * nb * post + i_m * post + i_post;
                                        int32_t c = j_pre * nb * post + j_m * post + i_post;
                                        S_tensor[row * nb_total + c] = s_val;
                                    }
                                }
                            }
                        }
                    }

                    pblocks[n_pblocks].col_start = col_start;
                    pblocks[n_pblocks].block_size = nb_total;
                    pblocks[n_pblocks].S = S_tensor;
                    pblocks[n_pblocks].lambda = tensors[t].lambda_smooth[m];
                    n_pblocks++;

                    free(Sm);
                }
            }
        }

        /* Free marginal data */
        for (int32_t m = 0; m < nm; m++) {
            free(marginal_xcol[m]);
            free(marginal_knots[m]);
            free(marginal_basis[m]);
        }
    }

    free(is_smooth);
    free(n_basis_per_smooth);
    free(n_basis_per_tensor);
    *out_p = total_cols;
    *out_basis_map = bmap;
    if (out_penalty_blocks) {
        *out_penalty_blocks = pblocks;
        *out_n_blocks = n_pblocks;
    }
    return Xout;
}

/* ========== Cross-validation ========== */

/* Simple LCG for fold assignment */
static uint32_t cv_rng_next(uint32_t *state) {
    *state = (*state * 1664525u + 1013904223u) & 0x7FFFFFFFu;
    return *state;
}

/* Compute CV error for a given lambda using fold assignments */
static void compute_cv(
    const double *X, int32_t n, int32_t p,
    const double *y,
    const gam_params_t *params,
    const int32_t *fold_ids, int32_t n_folds,
    double lambda,
    double *out_mean, double *out_se
) {
    double *errors = (double *)calloc((size_t)n_folds, sizeof(double));
    double *counts = (double *)calloc((size_t)n_folds, sizeof(double));
    if (!errors || !counts) {
        free(errors); free(counts);
        *out_mean = NAN; *out_se = NAN;
        return;
    }

    int32_t link = params->link >= 0 ? params->link : gam_canonical_link(params->family);

    for (int32_t fold = 0; fold < n_folds; fold++) {
        /* Count train/test sizes */
        int32_t n_test = 0, n_train = 0;
        for (int32_t i = 0; i < n; i++) {
            if (fold_ids[i] == fold) n_test++;
            else n_train++;
        }
        if (n_train == 0 || n_test == 0) continue;

        /* Build train/test sets */
        double *X_train = (double *)malloc((size_t)n_train * (size_t)p * sizeof(double));
        double *y_train = (double *)malloc((size_t)n_train * sizeof(double));
        double *X_test = (double *)malloc((size_t)n_test * (size_t)p * sizeof(double));
        double *y_test = (double *)malloc((size_t)n_test * sizeof(double));
        if (!X_train || !y_train || !X_test || !y_test) {
            free(X_train); free(y_train); free(X_test); free(y_test);
            continue;
        }

        int32_t tr = 0, te = 0;
        for (int32_t i = 0; i < n; i++) {
            if (fold_ids[i] == fold) {
                memcpy(X_test + te * p, X + i * p, (size_t)p * sizeof(double));
                y_test[te] = y[i];
                te++;
            } else {
                memcpy(X_train + tr * p, X + i * p, (size_t)p * sizeof(double));
                y_train[tr] = y[i];
                tr++;
            }
        }

        /* Fit on train with single lambda */
        gam_params_t cv_params = *params;
        cv_params.n_folds = 0;  /* no nested CV */
        cv_params.n_lambda = 1;
        double lam_arr[1] = { lambda };
        cv_params.lambda = lam_arr;
        cv_params.n_lambda_user = 1;
        cv_params.smooths = NULL;  /* CV on already-expanded X */
        cv_params.n_smooths = 0;

        gam_path_t *cv_path = gam_fit(X_train, n_train, p, y_train, &cv_params);
        if (cv_path && cv_path->n_fits > 0) {
            /* Compute test deviance */
            double fold_dev = 0.0;
            for (int32_t i = 0; i < n_test; i++) {
                double eta_i = cv_path->fits[0].beta[0];  /* intercept */
                for (int32_t j = 0; j < p; j++) {
                    eta_i += X_test[i * p + j] * cv_path->fits[0].beta[j + 1];
                }
                double mu_i = linkinv_fn(link, eta_i);
                fold_dev += deviance_unit(params->family, y_test[i], mu_i,
                                          params->tweedie_power, params->neg_binom_theta);
            }
            errors[fold] = fold_dev / n_test;
            counts[fold] = 1.0;
            gam_free(cv_path);
        }

        free(X_train); free(y_train); free(X_test); free(y_test);
    }

    /* Compute mean and SE */
    int32_t valid = 0;
    double mean = 0.0;
    for (int32_t f = 0; f < n_folds; f++) {
        if (counts[f] > 0) { mean += errors[f]; valid++; }
    }
    if (valid > 0) {
        mean /= valid;
        double var = 0.0;
        for (int32_t f = 0; f < n_folds; f++) {
            if (counts[f] > 0) {
                double d = errors[f] - mean;
                var += d * d;
            }
        }
        var /= fmax(valid - 1, 1);
        *out_mean = mean;
        *out_se = sqrt(var / fmax(valid, 1));
    } else {
        *out_mean = NAN;
        *out_se = NAN;
    }

    free(errors);
    free(counts);
}

/* ========== Main fit function ========== */

gam_path_t *gam_fit(
    const double *X, int32_t nrow, int32_t ncol,
    const double *y,
    const gam_params_t *params
) {
    if (!X || !y || nrow < 1 || ncol < 1) {
        gam_set_error("gam_fit: invalid input");
        return NULL;
    }

    int32_t family = params->family;
    int32_t link = params->link >= 0 ? params->link : gam_canonical_link(family);
    int32_t penalty = params->penalty;
    double alpha = params->alpha;

    /* ---- GAM basis expansion ---- */
    int32_t p_expanded = ncol;
    int32_t *basis_map = NULL;
    double *X_expanded = NULL;
    penalty_block_t *smooth_pblocks = NULL;
    int32_t n_smooth_pblocks = 0;

    int has_smooths = (params->n_smooths > 0 && params->smooths);
    int has_tensors = (params->n_tensors > 0 && params->tensors);

    if (has_smooths || has_tensors) {
        X_expanded = expand_basis(
            X, nrow, ncol,
            params->smooths, has_smooths ? params->n_smooths : 0,
            params->tensors, has_tensors ? params->n_tensors : 0,
            &p_expanded, &basis_map,
            &smooth_pblocks, &n_smooth_pblocks);
        if (!X_expanded) {
            gam_set_error("gam_fit: basis expansion failed");
            return NULL;
        }
    } else {
        /* No expansion, work directly with X */
        X_expanded = (double *)malloc((size_t)nrow * (size_t)ncol * sizeof(double));
        if (!X_expanded) { gam_set_error("gam_fit: alloc failed"); return NULL; }
        memcpy(X_expanded, X, (size_t)nrow * (size_t)ncol * sizeof(double));
        basis_map = (int32_t *)malloc((size_t)ncol * sizeof(int32_t));
        if (!basis_map) { free(X_expanded); gam_set_error("gam_fit: alloc failed"); return NULL; }
        for (int32_t j = 0; j < ncol; j++) basis_map[j] = j;
    }

    int32_t p = p_expanded;

    /* ---- Standardize ---- */
    double *x_mean = (double *)calloc((size_t)p, sizeof(double));
    double *x_sd = (double *)malloc((size_t)p * sizeof(double));
    if (!x_mean || !x_sd) {
        free(X_expanded); free(basis_map); free(x_mean); free(x_sd);
        gam_set_error("gam_fit: alloc failed");
        return NULL;
    }

    double y_mean = 0.0, y_sd = 1.0;

    if (params->standardize) {
        for (int32_t j = 0; j < p; j++) {
            double sum = 0.0;
            for (int32_t i = 0; i < nrow; i++) sum += X_expanded[i * p + j];
            x_mean[j] = sum / nrow;
        }
        for (int32_t j = 0; j < p; j++) {
            double ss = 0.0;
            for (int32_t i = 0; i < nrow; i++) {
                double d = X_expanded[i * p + j] - x_mean[j];
                ss += d * d;
            }
            x_sd[j] = sqrt(ss / nrow);
            if (x_sd[j] < 1e-10) x_sd[j] = 1.0;  /* constant column */
        }
        /* Center and scale */
        for (int32_t i = 0; i < nrow; i++) {
            for (int32_t j = 0; j < p; j++) {
                X_expanded[i * p + j] = (X_expanded[i * p + j] - x_mean[j]) / x_sd[j];
            }
        }
    } else {
        for (int32_t j = 0; j < p; j++) {
            x_mean[j] = 0.0;
            x_sd[j] = 1.0;
        }
    }

    /* Center y for Gaussian */
    double *y_work = (double *)malloc((size_t)nrow * sizeof(double));
    if (!y_work) {
        free(X_expanded); free(basis_map); free(x_mean); free(x_sd);
        gam_set_error("gam_fit: alloc failed");
        return NULL;
    }
    memcpy(y_work, y, (size_t)nrow * sizeof(double));

    if ((family == GAM_FAMILY_GAUSSIAN || family == GAM_FAMILY_HUBER ||
         family == GAM_FAMILY_QUANTILE) && link == GAM_LINK_IDENTITY && params->standardize) {
        double sum = 0.0;
        for (int32_t i = 0; i < nrow; i++) sum += y[i];
        y_mean = sum / nrow;
        double ss = 0.0;
        for (int32_t i = 0; i < nrow; i++) {
            double d = y[i] - y_mean;
            ss += d * d;
        }
        y_sd = sqrt(ss / nrow);
        if (y_sd < 1e-10) y_sd = 1.0;
        for (int32_t i = 0; i < nrow; i++) {
            y_work[i] = (y[i] - y_mean) / y_sd;
        }
    }

    /* ---- Setup working data ---- */
    cd_work_t w;
    memset(&w, 0, sizeof(w));
    w.n = nrow;
    w.p = p;
    w.Xs = X_expanded;
    w.y = y_work;
    w.family = family;
    w.link = link;
    w.penalty = penalty;
    w.alpha = alpha;
    w.gamma_mcp = params->gamma_mcp;
    w.gamma_scad = params->gamma_scad;
    w.tweedie_p = params->tweedie_power;
    w.nb_theta = params->neg_binom_theta;
    w.fit_intercept = params->fit_intercept;
    w.tol = params->tol;
    w.sample_weight = params->sample_weight;
    w.offset = params->offset;
    w.fused_order = (params->fused_order > 0) ? params->fused_order : 1;
    w.huber_gamma = (params->huber_gamma > 0) ? params->huber_gamma : 1.345;
    /* Scale gamma to standardized y scale */
    if (family == GAM_FAMILY_HUBER && params->standardize && y_sd > 1e-10)
        w.huber_gamma /= y_sd;
    w.quantile_tau = (params->quantile_tau > 0 && params->quantile_tau < 1)
                     ? params->quantile_tau : 0.5;

    /* Allocate working arrays */
    w.w = (double *)malloc((size_t)nrow * sizeof(double));
    w.r = (double *)malloc((size_t)nrow * sizeof(double));
    w.eta = (double *)malloc((size_t)nrow * sizeof(double));
    w.mu = (double *)malloc((size_t)nrow * sizeof(double));
    w.beta = (double *)calloc((size_t)p, sizeof(double));
    w.xw_sq = (double *)malloc((size_t)p * sizeof(double));
    w.pf = (double *)malloc((size_t)p * sizeof(double));
    w.lb = (double *)malloc((size_t)p * sizeof(double));
    w.ub = (double *)malloc((size_t)p * sizeof(double));
    w.active = (int32_t *)calloc((size_t)p, sizeof(int32_t));
    w.ever_active = (int32_t *)calloc((size_t)p, sizeof(int32_t));
    w.x_mean = x_mean;
    w.x_sd = x_sd;

    if (!w.w || !w.r || !w.eta || !w.mu || !w.beta || !w.xw_sq ||
        !w.pf || !w.lb || !w.ub || !w.active || !w.ever_active) {
        gam_set_error("gam_fit: alloc failed");
        goto cleanup_work;
    }

    /* SLOPE: allocate work buffers and generate BH sequence */
    if (penalty == GAM_PENALTY_SLOPE) {
        w.slope_seq = (double *)malloc((size_t)p * sizeof(double));
        w.slope_pairs = (slope_pair_t *)malloc((size_t)p * sizeof(slope_pair_t));
        w.slope_work = (double *)malloc((size_t)p * sizeof(double));
        w.slope_abs = (double *)malloc((size_t)p * sizeof(double));
        if (!w.slope_seq || !w.slope_pairs || !w.slope_work || !w.slope_abs) {
            gam_set_error("gam_fit: SLOPE alloc failed");
            goto cleanup_work;
        }
        if (params->slope_lambda && params->slope_n_lambda >= p) {
            /* User-supplied SLOPE sequence */
            memcpy(w.slope_seq, params->slope_lambda, (size_t)p * sizeof(double));
        } else {
            /* Generate BH sequence */
            slope_bh_sequence(w.slope_seq, p, params->slope_q);
        }
    }

    /* Group penalty: setup group structure */
    if ((penalty == GAM_PENALTY_GROUP_L1 || penalty == GAM_PENALTY_SGL) && params->groups) {
        w.groups = params->groups;
        w.n_groups = params->n_groups;
        w.sgl_alpha = (penalty == GAM_PENALTY_SGL) ? params->alpha : 0.0;

        w.group_start = (int32_t *)malloc((size_t)w.n_groups * sizeof(int32_t));
        w.group_size = (int32_t *)calloc((size_t)w.n_groups, sizeof(int32_t));
        if (!w.group_start || !w.group_size) {
            gam_set_error("gam_fit: group alloc failed");
            goto cleanup_work;
        }

        /* Count group sizes */
        for (int32_t j = 0; j < p; j++) {
            int32_t g = params->groups[j];
            if (g >= 0 && g < w.n_groups) w.group_size[g]++;
        }
        /* Compute start indices (groups packed contiguously -- feature order assumed) */
        w.group_start[0] = 0;
        for (int32_t g = 1; g < w.n_groups; g++) {
            w.group_start[g] = w.group_start[g - 1] + w.group_size[g - 1];
        }
    } else if (penalty == GAM_PENALTY_GROUP_L1 || penalty == GAM_PENALTY_SGL) {
        /* No groups provided: treat each feature as its own group */
        w.n_groups = p;
        w.groups = (int32_t *)malloc((size_t)p * sizeof(int32_t));
        w.group_start = (int32_t *)malloc((size_t)p * sizeof(int32_t));
        w.group_size = (int32_t *)malloc((size_t)p * sizeof(int32_t));
        if (!w.groups || !w.group_start || !w.group_size) {
            gam_set_error("gam_fit: group alloc failed");
            goto cleanup_work;
        }
        for (int32_t j = 0; j < p; j++) {
            w.groups[j] = j;
            w.group_start[j] = j;
            w.group_size[j] = 1;
        }
        w.sgl_alpha = (penalty == GAM_PENALTY_SGL) ? params->alpha : 0.0;
    }

    /* Wire smoothness penalty blocks into CD solver */
    if (n_smooth_pblocks > 0 && smooth_pblocks) {
        w.n_penalty_blocks = n_smooth_pblocks;
        w.penalty_blocks = smooth_pblocks;
    }

    /* Initialize weights, penalty factors, bounds */
    for (int32_t i = 0; i < nrow; i++) w.w[i] = 1.0;
    for (int32_t j = 0; j < p; j++) {
        w.pf[j] = (params->penalty_factor && j < ncol) ? params->penalty_factor[j] : 1.0;
        w.lb[j] = (params->lower_bounds && j < ncol) ? params->lower_bounds[j] : -1e30;
        w.ub[j] = (params->upper_bounds && j < ncol) ? params->upper_bounds[j] : 1e30;
    }

    /* Initialize intercept */
    if (params->fit_intercept) {
        if ((family == GAM_FAMILY_GAUSSIAN || family == GAM_FAMILY_HUBER ||
             family == GAM_FAMILY_QUANTILE) && link == GAM_LINK_IDENTITY) {
            w.intercept = 0.0;  /* y is already centered */
        } else if (family == GAM_FAMILY_BINOMIAL) {
            double ysum = 0.0;
            for (int32_t i = 0; i < nrow; i++) ysum += y_work[i];
            double p_bar = ysum / nrow;
            p_bar = clamp(p_bar, 0.01, 0.99);
            w.intercept = log(p_bar / (1.0 - p_bar));
        } else if (family == GAM_FAMILY_POISSON || family == GAM_FAMILY_GAMMA ||
                   family == GAM_FAMILY_NEG_BINOM || family == GAM_FAMILY_TWEEDIE) {
            double ysum = 0.0;
            for (int32_t i = 0; i < nrow; i++) ysum += y_work[i];
            double y_bar = fmax(ysum / nrow, 1e-5);
            w.intercept = log(y_bar);
        } else {
            w.intercept = 0.0;
        }
    }

    /* Initialize residuals (for Gaussian) */
    for (int32_t i = 0; i < nrow; i++) {
        w.r[i] = y_work[i] - w.intercept;
    }

    /* Compute weighted x^2 sums (initial, unit weights for Gaussian) */
    for (int32_t j = 0; j < p; j++) {
        double s = 0.0;
        for (int32_t i = 0; i < nrow; i++) {
            double xij = X_expanded[i * p + j];
            s += xij * xij * w.w[i];
        }
        w.xw_sq[j] = s / nrow;
    }

    /* For Huber/quantile: compute robust weights before lambda_max */
    if (family == GAM_FAMILY_HUBER || family == GAM_FAMILY_QUANTILE) {
        update_robust_weights(&w);
    }

    /* ---- Compute lambda sequence ---- */
    double lambda_max = compute_lambda_max(&w);
    double lmr = params->lambda_min_ratio;
    if (lmr <= 0) lmr = (nrow >= p) ? 1e-4 : 1e-2;

    int32_t n_lambda;
    double *lambda_seq;

    if (params->lambda && params->n_lambda_user > 0) {
        /* User-supplied lambda */
        n_lambda = params->n_lambda_user;
        lambda_seq = (double *)malloc((size_t)n_lambda * sizeof(double));
        if (!lambda_seq) { gam_set_error("gam_fit: alloc failed"); goto cleanup_work; }
        memcpy(lambda_seq, params->lambda, (size_t)n_lambda * sizeof(double));
    } else {
        n_lambda = params->n_lambda > 0 ? params->n_lambda : 100;
        lambda_seq = (double *)malloc((size_t)n_lambda * sizeof(double));
        if (!lambda_seq) { gam_set_error("gam_fit: alloc failed"); goto cleanup_work; }
        double log_max = log(fmax(lambda_max, 1e-15));
        double log_min = log(fmax(lambda_max * lmr, 1e-15));
        for (int32_t k = 0; k < n_lambda; k++) {
            double frac = (n_lambda > 1) ? (double)k / (n_lambda - 1) : 0.0;
            lambda_seq[k] = exp(log_max + frac * (log_min - log_max));
        }
    }

    /* ---- Cross-validation setup ---- */
    int32_t n_folds = params->n_folds;
    int32_t *fold_ids = NULL;
    if (n_folds > 1) {
        fold_ids = (int32_t *)malloc((size_t)nrow * sizeof(int32_t));
        if (fold_ids) {
            /* Simple random fold assignment */
            uint32_t rng_state = params->seed;
            for (int32_t i = 0; i < nrow; i++) {
                fold_ids[i] = (int32_t)(cv_rng_next(&rng_state) % (uint32_t)n_folds);
            }
        }
    }

    /* ---- Null deviance ---- */
    double null_dev = 0.0;
    {
        double y_bar = 0.0;
        for (int32_t i = 0; i < nrow; i++) y_bar += y_work[i];
        y_bar /= nrow;
        if (family == GAM_FAMILY_HUBER) {
            double gamma = w.huber_gamma;
            for (int32_t i = 0; i < nrow; i++) {
                double ri = y_work[i] - y_bar;
                double ari = fabs(ri);
                null_dev += (ari <= gamma) ? ri * ri : 2.0 * gamma * ari - gamma * gamma;
            }
        } else if (family == GAM_FAMILY_QUANTILE) {
            double tau = w.quantile_tau;
            for (int32_t i = 0; i < nrow; i++) {
                double ri = y_work[i] - y_bar;
                null_dev += 2.0 * ri * (tau - (ri < 0 ? 1.0 : 0.0));
            }
        } else {
            double mu_null;
            if ((family == GAM_FAMILY_GAUSSIAN) && link == GAM_LINK_IDENTITY) {
                mu_null = y_bar;
            } else {
                mu_null = linkinv_fn(link, w.intercept);
            }
            for (int32_t i = 0; i < nrow; i++) {
                null_dev += deviance_unit(family, y_work[i], mu_null,
                                           params->tweedie_power, params->neg_binom_theta);
            }
        }
    }

    /* ---- Allocate result ---- */
    gam_path_t *path = (gam_path_t *)calloc(1, sizeof(gam_path_t));
    if (!path) { gam_set_error("gam_fit: alloc failed"); free(lambda_seq); goto cleanup_work; }
    path->fits = (gam_fit_t *)calloc((size_t)n_lambda, sizeof(gam_fit_t));
    if (!path->fits) {
        free(path); free(lambda_seq);
        gam_set_error("gam_fit: alloc failed");
        goto cleanup_work;
    }

    path->n_features = ncol;
    path->n_coefs = p + 1;  /* including intercept */
    path->family = family;
    path->link = link;
    path->penalty = penalty;
    path->alpha = alpha;
    path->n_basis_total = p;
    path->x_mean = x_mean;
    path->x_sd = x_sd;
    path->y_mean = y_mean;
    path->y_sd = y_sd;
    path->basis_map = basis_map;
    path->idx_min = -1;
    path->idx_1se = -1;

    /* Store smooth/tensor specs for prediction-time basis expansion */
    path->n_smooths = 0;
    path->smooths = NULL;
    path->n_tensors = 0;
    path->tensors = NULL;
    if (has_smooths && params->n_smooths > 0) {
        path->smooths = (gam_smooth_t *)malloc(
            (size_t)params->n_smooths * sizeof(gam_smooth_t));
        if (path->smooths) {
            memcpy(path->smooths, params->smooths,
                   (size_t)params->n_smooths * sizeof(gam_smooth_t));
            path->n_smooths = params->n_smooths;
        }
    }
    if (has_tensors && params->n_tensors > 0) {
        path->tensors = (gam_tensor_t *)malloc(
            (size_t)params->n_tensors * sizeof(gam_tensor_t));
        if (path->tensors) {
            memcpy(path->tensors, params->tensors,
                   (size_t)params->n_tensors * sizeof(gam_tensor_t));
            path->n_tensors = params->n_tensors;
        }
    }

    /* ---- Fit along the path ---- */
    double lambda_prev = 0.0;
    int32_t n_fits = 0;

    for (int32_t k = 0; k < n_lambda; k++) {
        double lambda = lambda_seq[k];
        int32_t iter = 0;

        fit_single_lambda(&w, lambda, lambda_prev, params->max_iter,
                          params->max_inner, params->screening, &iter);

        /* Store result */
        gam_fit_t *fit = &path->fits[n_fits];
        fit->beta = (double *)malloc((size_t)(p + 1) * sizeof(double));
        if (!fit->beta) break;

        /* Unstandardize coefficients */
        if ((family == GAM_FAMILY_GAUSSIAN || family == GAM_FAMILY_HUBER ||
                 family == GAM_FAMILY_QUANTILE) && link == GAM_LINK_IDENTITY && params->standardize) {
            /* beta_orig = beta_std * y_sd / x_sd */
            fit->beta[0] = w.intercept * y_sd + y_mean;
            for (int32_t j = 0; j < p; j++) {
                fit->beta[j + 1] = w.beta[j] * y_sd / x_sd[j];
                fit->beta[0] -= fit->beta[j + 1] * x_mean[j];
            }
        } else if (params->standardize) {
            /* For non-Gaussian, beta_orig = beta_std / x_sd, intercept adjusted */
            fit->beta[0] = w.intercept;
            for (int32_t j = 0; j < p; j++) {
                fit->beta[j + 1] = w.beta[j] / x_sd[j];
                fit->beta[0] -= fit->beta[j + 1] * x_mean[j];
            }
        } else {
            fit->beta[0] = w.intercept;
            for (int32_t j = 0; j < p; j++) {
                fit->beta[j + 1] = w.beta[j];
            }
        }

        fit->n_coefs = p + 1;
        fit->lambda = lambda;
        fit->n_iter = iter;

        /* Count degrees of freedom */
        fit->df = 0;
        for (int32_t j = 0; j < p; j++) {
            if (w.beta[j] != 0.0) fit->df++;
        }

        /* Deviance (on standardized scale for simplicity, then store) */
        if (family == GAM_FAMILY_HUBER) {
            /* Huber deviance: 2 * sum(rho_gamma(r_i)), scaled back */
            double gamma = w.huber_gamma;
            fit->deviance = 0.0;
            for (int32_t i = 0; i < nrow; i++) {
                double ri = w.r[i];
                double ari = fabs(ri);
                if (ari <= gamma)
                    fit->deviance += ri * ri;
                else
                    fit->deviance += 2.0 * gamma * ari - gamma * gamma;
            }
            fit->deviance *= y_sd * y_sd;
        } else if (family == GAM_FAMILY_QUANTILE) {
            /* Quantile deviance: 2 * sum(rho_tau(r_i)), scaled back */
            double tau = w.quantile_tau;
            fit->deviance = 0.0;
            for (int32_t i = 0; i < nrow; i++) {
                double ri = w.r[i];
                fit->deviance += 2.0 * ri * (tau - (ri < 0 ? 1.0 : 0.0));
            }
            fit->deviance *= y_sd * y_sd;
        } else if (family == GAM_FAMILY_GAUSSIAN && link == GAM_LINK_IDENTITY) {
            fit->deviance = 0.0;
            for (int32_t i = 0; i < nrow; i++) {
                fit->deviance += w.r[i] * w.r[i] * y_sd * y_sd;
            }
        } else {
            fit->deviance = compute_deviance(&w);
        }
        fit->null_deviance = null_dev;
        if ((family == GAM_FAMILY_GAUSSIAN || family == GAM_FAMILY_HUBER ||
                 family == GAM_FAMILY_QUANTILE) && link == GAM_LINK_IDENTITY && params->standardize) {
            fit->null_deviance = null_dev * y_sd * y_sd;
        }

        /* CV */
        fit->cv_mean = NAN;
        fit->cv_se = NAN;
        if (n_folds > 1 && fold_ids) {
            compute_cv(X_expanded, nrow, p, y_work, params,
                       fold_ids, n_folds, lambda,
                       &fit->cv_mean, &fit->cv_se);
        }

        n_fits++;
        lambda_prev = lambda;

        /* Early stopping: if deviance ratio is very close to 1 */
        double dev_ratio = 1.0 - fit->deviance / fmax(fit->null_deviance, 1e-15);
        if (dev_ratio > 0.999) break;
    }

    path->n_fits = n_fits;

    /* Find CV min and 1se */
    if (n_folds > 1) {
        double min_cv = 1e30;
        for (int32_t k = 0; k < n_fits; k++) {
            if (!isnan(path->fits[k].cv_mean) && path->fits[k].cv_mean < min_cv) {
                min_cv = path->fits[k].cv_mean;
                path->idx_min = k;
            }
        }
        if (path->idx_min >= 0) {
            double threshold = min_cv + path->fits[path->idx_min].cv_se;
            path->idx_1se = path->idx_min;
            for (int32_t k = 0; k < path->idx_min; k++) {
                if (!isnan(path->fits[k].cv_mean) && path->fits[k].cv_mean <= threshold) {
                    path->idx_1se = k;
                    break;
                }
            }
        }
    }

    /* ---- Relaxed fits ---- */
    if (params->relax && n_fits > 0) {
        path->relaxed_fits = (gam_fit_t *)calloc((size_t)n_fits, sizeof(gam_fit_t));
        if (path->relaxed_fits) {
            for (int32_t k = 0; k < n_fits; k++) {
                gam_fit_t *orig = &path->fits[k];
                gam_fit_t *relaxed = &path->relaxed_fits[k];

                /* Count active features */
                int32_t n_active = 0;
                for (int32_t j = 0; j < p; j++) {
                    if (orig->beta[j + 1] != 0.0) n_active++;
                }

                if (n_active == 0 || n_active >= p) {
                    /* Nothing to relax */
                    relaxed->beta = (double *)malloc((size_t)(p + 1) * sizeof(double));
                    if (relaxed->beta) {
                        memcpy(relaxed->beta, orig->beta, (size_t)(p + 1) * sizeof(double));
                    }
                    relaxed->n_coefs = orig->n_coefs;
                    relaxed->lambda = orig->lambda;
                    relaxed->deviance = orig->deviance;
                    relaxed->null_deviance = orig->null_deviance;
                    relaxed->df = orig->df;
                    relaxed->cv_mean = NAN;
                    relaxed->cv_se = NAN;
                    continue;
                }

                /* Build reduced X with only active features */
                double *X_active = (double *)malloc((size_t)nrow * (size_t)n_active * sizeof(double));
                if (!X_active) continue;

                int32_t col_idx = 0;
                int32_t *active_cols = (int32_t *)malloc((size_t)n_active * sizeof(int32_t));
                if (!active_cols) { free(X_active); continue; }

                for (int32_t j = 0; j < p; j++) {
                    if (orig->beta[j + 1] != 0.0) {
                        for (int32_t i = 0; i < nrow; i++) {
                            X_active[i * n_active + col_idx] = X_expanded[i * p + j];
                        }
                        active_cols[col_idx] = j;
                        col_idx++;
                    }
                }

                /* Fit unpenalized on active set */
                gam_params_t relax_params;
                gam_params_init(&relax_params);
                relax_params.family = family;
                relax_params.link = link;
                relax_params.penalty = GAM_PENALTY_NONE;
                relax_params.standardize = 0;  /* already standardized */
                relax_params.fit_intercept = params->fit_intercept;
                relax_params.n_lambda = 1;
                double lam0 = 0.0;
                relax_params.lambda = &lam0;
                relax_params.n_lambda_user = 1;

                gam_path_t *rpath = gam_fit(X_active, nrow, n_active, y_work, &relax_params);
                if (rpath && rpath->n_fits > 0) {
                    relaxed->beta = (double *)calloc((size_t)(p + 1), sizeof(double));
                    if (relaxed->beta) {
                        relaxed->beta[0] = rpath->fits[0].beta[0];
                        for (int32_t a = 0; a < n_active; a++) {
                            relaxed->beta[active_cols[a] + 1] = rpath->fits[0].beta[a + 1];
                        }
                        /* Unstandardize if needed */
                        if ((family == GAM_FAMILY_GAUSSIAN || family == GAM_FAMILY_HUBER ||
                 family == GAM_FAMILY_QUANTILE) && link == GAM_LINK_IDENTITY && params->standardize) {
                            double b0 = relaxed->beta[0] * y_sd + y_mean;
                            for (int32_t j = 0; j < p; j++) {
                                if (relaxed->beta[j + 1] != 0.0) {
                                    relaxed->beta[j + 1] *= y_sd / x_sd[j];
                                    b0 -= relaxed->beta[j + 1] * x_mean[j];
                                }
                            }
                            relaxed->beta[0] = b0;
                        } else if (params->standardize) {
                            for (int32_t j = 0; j < p; j++) {
                                if (relaxed->beta[j + 1] != 0.0) {
                                    relaxed->beta[j + 1] /= x_sd[j];
                                    relaxed->beta[0] -= relaxed->beta[j + 1] * x_mean[j];
                                }
                            }
                        }
                    }
                    relaxed->n_coefs = p + 1;
                    relaxed->lambda = orig->lambda;
                    relaxed->df = n_active;
                    relaxed->deviance = rpath->fits[0].deviance;
                    relaxed->null_deviance = orig->null_deviance;
                    relaxed->cv_mean = NAN;
                    relaxed->cv_se = NAN;
                    gam_free(rpath);
                }

                free(X_active);
                free(active_cols);
            }
        }
    }

    /* Cleanup work arrays (but keep x_mean, x_sd, basis_map in path) */
    free(w.w); free(w.r); free(w.eta); free(w.mu);
    free(w.beta); free(w.xw_sq); free(w.pf);
    free(w.lb); free(w.ub); free(w.active); free(w.ever_active);
    free(w.slope_seq); free(w.slope_pairs); free(w.slope_work); free(w.slope_abs);
    if (!params->groups) { free(w.groups); }  /* only free if we allocated it */
    free(w.group_start); free(w.group_size);
    /* Free penalty block S matrices */
    if (smooth_pblocks) {
        for (int32_t b = 0; b < n_smooth_pblocks; b++)
            free(smooth_pblocks[b].S);
        free(smooth_pblocks);
    }
    free(X_expanded); free(y_work); free(lambda_seq); free(fold_ids);

    return path;

cleanup_work:
    free(w.w); free(w.r); free(w.eta); free(w.mu);
    free(w.beta); free(w.xw_sq); free(w.pf);
    free(w.lb); free(w.ub); free(w.active); free(w.ever_active);
    free(w.slope_seq); free(w.slope_pairs); free(w.slope_work); free(w.slope_abs);
    if (!params->groups) { free(w.groups); }  /* only free if we allocated it */
    free(w.group_start); free(w.group_size);
    /* Free penalty block S matrices */
    if (smooth_pblocks) {
        for (int32_t b = 0; b < n_smooth_pblocks; b++)
            free(smooth_pblocks[b].S);
        free(smooth_pblocks);
    }
    free(X_expanded); free(basis_map); free(x_mean); free(x_sd);
    free(y_work);
    return NULL;
}

/* ========== Prediction ========== */

int gam_predict_eta(
    const gam_path_t *path, int32_t fit_idx,
    const double *X, int32_t nrow, int32_t ncol,
    double *out
) {
    if (!path || fit_idx < 0 || fit_idx >= path->n_fits) {
        gam_set_error("predict_eta: invalid path or index");
        return -1;
    }

    const gam_fit_t *fit = &path->fits[fit_idx];
    int32_t n_coefs = fit->n_coefs;
    int32_t n_feats = n_coefs - 1;

    /* Check if basis expansion is needed */
    int has_smooths = (path->n_smooths > 0 && path->smooths);
    int has_tensors = (path->n_tensors > 0 && path->tensors);

    if ((has_smooths || has_tensors) && n_feats > ncol) {
        /* GAM case: expand basis for new data, then predict.
         * Stored betas are already unstandardized, so just X_expanded * beta. */
        int32_t p_expanded = 0;
        int32_t *bmap = NULL;
        double *X_exp = expand_basis(
            X, nrow, ncol,
            path->smooths, path->n_smooths,
            path->tensors, path->n_tensors,
            &p_expanded, &bmap, NULL, NULL);
        if (!X_exp) {
            gam_set_error("predict_eta: basis expansion failed");
            return -1;
        }

        for (int32_t i = 0; i < nrow; i++) {
            double eta = fit->beta[0];
            for (int32_t j = 0; j < p_expanded && j < n_feats; j++) {
                eta += X_exp[i * p_expanded + j] * fit->beta[j + 1];
            }
            out[i] = eta;
        }

        free(X_exp);
        free(bmap);
        return 0;
    }

    /* GLM case: coefficients map directly to features */
    for (int32_t i = 0; i < nrow; i++) {
        double eta = fit->beta[0];  /* intercept */
        for (int32_t j = 0; j < n_feats && j < ncol; j++) {
            eta += X[i * ncol + j] * fit->beta[j + 1];
        }
        out[i] = eta;
    }

    return 0;
}

int gam_predict(
    const gam_path_t *path, int32_t fit_idx,
    const double *X, int32_t nrow, int32_t ncol,
    double *out
) {
    int ret = gam_predict_eta(path, fit_idx, X, nrow, ncol, out);
    if (ret != 0) return ret;

    int32_t link = path->link;

    for (int32_t i = 0; i < nrow; i++) {
        out[i] = linkinv_fn(link, out[i]);
    }
    return 0;
}

int gam_predict_proba(
    const gam_path_t *path, int32_t fit_idx,
    const double *X, int32_t nrow, int32_t ncol,
    double *out
) {
    if (path->family == GAM_FAMILY_MULTINOMIAL) {
        return gam_predict_multinomial(path, fit_idx, X, nrow, ncol, out);
    }
    if (path->family != GAM_FAMILY_BINOMIAL) {
        gam_set_error("predict_proba: only for binomial/multinomial family");
        return -1;
    }
    return gam_predict(path, fit_idx, X, nrow, ncol, out);
}

/* ========== Diagnostics ========== */

double gam_deviance(const gam_path_t *path, int32_t fit_idx) {
    if (!path || fit_idx < 0 || fit_idx >= path->n_fits) return NAN;
    return path->fits[fit_idx].deviance;
}

double gam_aic(const gam_path_t *path, int32_t fit_idx, int32_t nrow) {
    if (!path || fit_idx < 0 || fit_idx >= path->n_fits) return NAN;
    const gam_fit_t *fit = &path->fits[fit_idx];
    return fit->deviance + 2.0 * (fit->df + 1);  /* +1 for intercept */
    (void)nrow;
}

double gam_bic(const gam_path_t *path, int32_t fit_idx, int32_t nrow) {
    if (!path || fit_idx < 0 || fit_idx >= path->n_fits) return NAN;
    const gam_fit_t *fit = &path->fits[fit_idx];
    return fit->deviance + log((double)nrow) * (fit->df + 1);
}

/* ========== Cox PH: penalized partial likelihood via coordinate descent ========== */
/*
 * Reference: Simon, Friedman, Hastie, Tibshirani (2011).
 * "Regularization Paths for Cox's PH Model via Coordinate Descent."
 * Journal of Statistical Software 39(5).
 *
 * Partial log-likelihood (Breslow approximation for ties):
 *   l(beta) = sum_{i: d_i=1} [eta_i - log(sum_{j in R_i} exp(eta_j))]
 * where R_i = {j : t_j >= t_i} is the risk set at time t_i.
 *
 * Gradient for feature j:
 *   g_j = (1/n) * sum_{i: d_i=1} [x_ij - (sum_{k in R_i} x_kj * w_k) / S0_i]
 * where w_k = exp(eta_k), S0_i = sum_{k in R_i} w_k
 *
 * Hessian diagonal for feature j:
 *   h_j = (1/n) * sum_{i: d_i=1} [(sum_{k in R_i} x_kj^2 * w_k)/S0_i
 *                                   - ((sum_{k in R_i} x_kj * w_k)/S0_i)^2]
 *
 * Coordinate descent update: beta_j <- prox(z, v, lambda)
 * where z = g_j + h_j * beta_j (partial residual), v = h_j
 */

/* Sort indices by time descending (for risk set accumulation) */
typedef struct { double time; double status; int32_t idx; } cox_obs_t;

static int cox_cmp_desc(const void *a, const void *b) {
    const cox_obs_t *oa = (const cox_obs_t *)a;
    const cox_obs_t *ob = (const cox_obs_t *)b;
    if (ob->time > oa->time) return 1;
    if (ob->time < oa->time) return -1;
    /* Ties: events before censored (so risk set includes events at tie times) */
    if (oa->status > ob->status) return -1;
    if (oa->status < ob->status) return 1;
    return 0;
}

/* Cox CD pass: compute gradient and Hessian for each feature,
 * then apply penalized update.
 * Data must be sorted by decreasing time. */
static int cox_cd_pass(
    int32_t n, int32_t p,
    const double *Xs,        /* standardized X, n*p (sorted order) */
    const double *status_s,  /* event indicator (sorted order) */
    double *beta, double *eta, double *exp_eta,
    int32_t *active, int32_t *ever_active,
    const double *pf, const double *lb, const double *ub,
    double lambda, double alpha,
    int32_t penalty, double gamma_mcp, double gamma_scad,
    double tol,
    int active_only
) {
    int any_changed = 0;

    /* Recompute exp(eta) */
    for (int32_t i = 0; i < n; i++) {
        exp_eta[i] = exp(eta[i]);
    }

    for (int32_t j = 0; j < p; j++) {
        if (active_only && !active[j]) continue;

        double old_beta = beta[j];

        /* Compute gradient g_j and Hessian h_j using cumulative sums.
         * We traverse i from 0 to n-1 (decreasing time order).
         * Cumulative sums accumulate the risk set from the end (largest times first).
         *
         * S0 = cumsum of exp_eta[k]
         * S1_j = cumsum of Xs[k,j] * exp_eta[k]
         * S2_j = cumsum of Xs[k,j]^2 * exp_eta[k]
         */
        double S0 = 0.0, S1 = 0.0, S2 = 0.0;
        double grad = 0.0, hess = 0.0;

        for (int32_t i = 0; i < n; i++) {
            double xij = Xs[i * p + j];
            double wi = exp_eta[i];

            S0 += wi;
            S1 += xij * wi;
            S2 += xij * xij * wi;

            if (status_s[i] > 0.5) {  /* event */
                double mean_x = S1 / fmax(S0, 1e-30);
                double var_x = S2 / fmax(S0, 1e-30) - mean_x * mean_x;

                grad += xij - mean_x;
                hess += fmax(var_x, 0.0);
            }
        }

        grad /= n;   /* negative log-PL gradient: we want to minimize */
        hess /= n;

        /* z = gradient + hessian * old_beta (partial residual form) */
        double z = grad + hess * old_beta;
        double v = hess;

        if (v < 1e-15) continue;

        double new_beta;
        if (pf[j] == 0.0) {
            new_beta = z / v;
        } else {
            new_beta = penalized_update(
                z, v, lambda, alpha,
                penalty, gamma_mcp, gamma_scad,
                pf[j], lb[j], ub[j]
            );
        }

        if (fabs(new_beta - old_beta) > tol * fmax(fabs(old_beta), 1.0)) {
            double delta = new_beta - old_beta;
            beta[j] = new_beta;
            /* Update eta */
            for (int32_t i = 0; i < n; i++) {
                eta[i] += delta * Xs[i * p + j];
            }
            any_changed = 1;
            if (new_beta != 0.0) {
                active[j] = 1;
                ever_active[j] = 1;
            }
        }
    }

    return any_changed;
}

/* Cox partial log-likelihood (for convergence checking) */
static double cox_partial_loglik(
    int32_t n, const double *status_s, const double *eta
) {
    double loglik = 0.0;
    /* Traverse from end (smallest time) to start (largest time) */
    /* to compute cumsum of exp(eta) for risk sets */
    double cum_exp = 0.0;
    for (int32_t i = n - 1; i >= 0; i--) {
        cum_exp += exp(eta[i]);
    }
    /* Now traverse forward (decreasing time): cum_exp starts as full sum */
    double removed = 0.0;
    for (int32_t i = 0; i < n; i++) {
        double S0 = cum_exp - removed;
        if (status_s[i] > 0.5) {
            loglik += eta[i] - log(fmax(S0, 1e-30));
        }
        removed += exp(eta[i]);
    }
    return loglik;
}

/* Compute lambda_max for Cox (max |gradient_j| at beta=0) */
static double cox_lambda_max(
    int32_t n, int32_t p,
    const double *Xs, const double *status_s,
    const double *pf, double alpha
) {
    /* At beta=0, exp(eta_i)=1 for all i, so S0 = cumulative count.
     * gradient_j = (1/n) * sum_{i: event} [x_ij - S1_j/S0]
     * where S1_j/S0 = (cumsum of x_kj) / (cumsum of 1) */
    double max_grad = 0.0;

    for (int32_t j = 0; j < p; j++) {
        if (pf[j] <= 0.0) continue;

        double S0 = 0.0, S1 = 0.0;
        double grad = 0.0;

        for (int32_t i = 0; i < n; i++) {
            double xij = Xs[i * p + j];
            S0 += 1.0;
            S1 += xij;
            if (status_s[i] > 0.5) {
                grad += xij - S1 / S0;
            }
        }
        grad /= n;
        double abs_grad = fabs(grad) / pf[j];
        if (abs_grad > max_grad) max_grad = abs_grad;
    }

    return (alpha > 0.0) ? max_grad / alpha : max_grad;
}

gam_path_t *gam_fit_cox(
    const double *X, int32_t nrow, int32_t ncol,
    const double *time, const double *status,
    const gam_params_t *params
) {
    if (!X || !time || !status || !params || nrow < 2 || ncol < 1) {
        gam_set_error("gam_fit_cox: invalid arguments");
        return NULL;
    }

    int32_t penalty = params->penalty;
    double alpha = params->alpha;
    if (penalty == GAM_PENALTY_NONE) alpha = 0.0;
    if (penalty == GAM_PENALTY_L1) alpha = 1.0;
    if (penalty == GAM_PENALTY_L2) alpha = 0.0;

    /* Sort observations by decreasing time */
    cox_obs_t *obs = (cox_obs_t *)malloc((size_t)nrow * sizeof(cox_obs_t));
    if (!obs) { gam_set_error("gam_fit_cox: alloc failed"); return NULL; }

    for (int32_t i = 0; i < nrow; i++) {
        obs[i].time = time[i];
        obs[i].status = status[i];
        obs[i].idx = i;
    }
    qsort(obs, (size_t)nrow, sizeof(cox_obs_t), cox_cmp_desc);

    /* Count events */
    int32_t n_events = 0;
    for (int32_t i = 0; i < nrow; i++) {
        if (status[i] > 0.5) n_events++;
    }
    if (n_events == 0) {
        free(obs);
        gam_set_error("gam_fit_cox: no events in data");
        return NULL;
    }

    /* Create sorted X and status arrays */
    double *Xs = (double *)malloc((size_t)nrow * (size_t)ncol * sizeof(double));
    double *status_s = (double *)malloc((size_t)nrow * sizeof(double));
    double *x_mean = (double *)calloc((size_t)ncol, sizeof(double));
    double *x_sd = (double *)malloc((size_t)ncol * sizeof(double));
    double *beta = (double *)calloc((size_t)ncol, sizeof(double));
    double *eta = (double *)calloc((size_t)nrow, sizeof(double));
    double *exp_eta = (double *)malloc((size_t)nrow * sizeof(double));
    int32_t *active = (int32_t *)calloc((size_t)ncol, sizeof(int32_t));
    int32_t *ever_active = (int32_t *)calloc((size_t)ncol, sizeof(int32_t));
    double *pf = (double *)malloc((size_t)ncol * sizeof(double));
    double *lb = (double *)malloc((size_t)ncol * sizeof(double));
    double *ub = (double *)malloc((size_t)ncol * sizeof(double));

    if (!Xs || !status_s || !x_mean || !x_sd || !beta || !eta || !exp_eta ||
        !active || !ever_active || !pf || !lb || !ub) {
        gam_set_error("gam_fit_cox: alloc failed");
        goto cox_cleanup;
    }

    /* Fill sorted arrays */
    for (int32_t i = 0; i < nrow; i++) {
        int32_t orig = obs[i].idx;
        status_s[i] = obs[i].status;
        for (int32_t j = 0; j < ncol; j++) {
            Xs[i * ncol + j] = X[orig * ncol + j];
        }
    }

    /* Standardize features */
    if (params->standardize) {
        for (int32_t j = 0; j < ncol; j++) {
            double m = 0.0;
            for (int32_t i = 0; i < nrow; i++) m += Xs[i * ncol + j];
            m /= nrow;
            x_mean[j] = m;

            double ss = 0.0;
            for (int32_t i = 0; i < nrow; i++) {
                double d = Xs[i * ncol + j] - m;
                ss += d * d;
            }
            x_sd[j] = (ss > 0.0) ? sqrt(ss / nrow) : 1.0;

            for (int32_t i = 0; i < nrow; i++) {
                Xs[i * ncol + j] = (Xs[i * ncol + j] - m) / x_sd[j];
            }
        }
    } else {
        for (int32_t j = 0; j < ncol; j++) {
            x_mean[j] = 0.0;
            x_sd[j] = 1.0;
        }
    }

    /* Penalty factors and bounds */
    for (int32_t j = 0; j < ncol; j++) {
        pf[j] = (params->penalty_factor) ? params->penalty_factor[j] : 1.0;
        lb[j] = (params->lower_bounds) ? params->lower_bounds[j] : -1e30;
        ub[j] = (params->upper_bounds) ? params->upper_bounds[j] : 1e30;
    }

    /* Lambda sequence */
    double lmax = cox_lambda_max(nrow, ncol, Xs, status_s, pf, alpha);
    double lmr = params->lambda_min_ratio;
    if (lmr <= 0) lmr = (nrow >= ncol) ? 1e-4 : 1e-2;

    int32_t n_lambda;
    double *lambda_seq;

    if (params->lambda && params->n_lambda_user > 0) {
        n_lambda = params->n_lambda_user;
        lambda_seq = (double *)malloc((size_t)n_lambda * sizeof(double));
        if (!lambda_seq) { gam_set_error("gam_fit_cox: alloc failed"); goto cox_cleanup; }
        memcpy(lambda_seq, params->lambda, (size_t)n_lambda * sizeof(double));
    } else {
        n_lambda = params->n_lambda > 0 ? params->n_lambda : 100;
        lambda_seq = (double *)malloc((size_t)n_lambda * sizeof(double));
        if (!lambda_seq) { gam_set_error("gam_fit_cox: alloc failed"); goto cox_cleanup; }
        double log_max = log(fmax(lmax, 1e-15));
        double log_min = log(fmax(lmax * lmr, 1e-15));
        for (int32_t k = 0; k < n_lambda; k++) {
            double frac = (n_lambda > 1) ? (double)k / (n_lambda - 1) : 0.0;
            lambda_seq[k] = exp(log_max + frac * (log_min - log_max));
        }
    }

    /* Allocate result */
    gam_path_t *path = (gam_path_t *)calloc(1, sizeof(gam_path_t));
    if (!path) { free(lambda_seq); gam_set_error("gam_fit_cox: alloc failed"); goto cox_cleanup; }
    path->fits = (gam_fit_t *)calloc((size_t)n_lambda, sizeof(gam_fit_t));
    if (!path->fits) { free(path); free(lambda_seq); gam_set_error("gam_fit_cox: alloc failed"); goto cox_cleanup; }

    path->n_features = ncol;
    path->n_coefs = ncol + 1;  /* intercept slot (always 0 for Cox) + features */
    path->family = GAM_FAMILY_COX;
    path->link = GAM_LINK_LOG;  /* Cox uses log-hazard = X*beta */
    path->penalty = penalty;
    path->alpha = alpha;
    path->n_basis_total = ncol;
    path->x_mean = x_mean;
    path->x_sd = x_sd;
    path->y_mean = 0.0;
    path->y_sd = 1.0;
    path->basis_map = NULL;
    path->idx_min = -1;
    path->idx_1se = -1;

    /* Null partial log-likelihood (beta=0) */
    double null_pll = cox_partial_loglik(nrow, status_s, eta);

    /* Fit along the path */
    int32_t n_fits = 0;
    for (int32_t k = 0; k < n_lambda; k++) {
        double lambda = lambda_seq[k];

        /* Coordinate descent with active set strategy */
        int32_t iter = 0;
        for (int32_t outer = 0; outer < params->max_iter; outer++) {
            /* Full pass first */
            int changed = cox_cd_pass(
                nrow, ncol, Xs, status_s,
                beta, eta, exp_eta,
                active, ever_active, pf, lb, ub,
                lambda, alpha, penalty,
                params->gamma_mcp, params->gamma_scad,
                params->tol, 0
            );
            iter++;

            if (!changed) break;

            /* Active set passes until convergence */
            for (int32_t inner = 0; inner < params->max_iter; inner++) {
                int ch2 = cox_cd_pass(
                    nrow, ncol, Xs, status_s,
                    beta, eta, exp_eta,
                    active, ever_active, pf, lb, ub,
                    lambda, alpha, penalty,
                    params->gamma_mcp, params->gamma_scad,
                    params->tol, 1
                );
                iter++;
                if (!ch2) break;
            }

            /* One more full pass to check KKT */
            int viol = cox_cd_pass(
                nrow, ncol, Xs, status_s,
                beta, eta, exp_eta,
                active, ever_active, pf, lb, ub,
                lambda, alpha, penalty,
                params->gamma_mcp, params->gamma_scad,
                params->tol, 0
            );
            iter++;
            if (!viol) break;
        }

        /* Store result */
        gam_fit_t *fit = &path->fits[n_fits];
        fit->beta = (double *)malloc((size_t)(ncol + 1) * sizeof(double));
        if (!fit->beta) break;

        /* Unstandardize: beta_orig = beta_std / x_sd */
        fit->beta[0] = 0.0;  /* Cox has no intercept (absorbed into baseline hazard) */
        for (int32_t j = 0; j < ncol; j++) {
            fit->beta[j + 1] = params->standardize ? beta[j] / x_sd[j] : beta[j];
        }

        fit->n_coefs = ncol + 1;
        fit->lambda = lambda;
        fit->n_iter = iter;

        /* Count nonzero */
        int32_t df = 0;
        for (int32_t j = 0; j < ncol; j++) {
            if (fabs(beta[j]) > 1e-15) df++;
        }
        fit->df = df;

        /* Deviance = -2 * (partial log-likelihood - null) */
        double pll = cox_partial_loglik(nrow, status_s, eta);
        fit->deviance = -2.0 * (pll - null_pll);
        fit->null_deviance = 0.0;  /* by definition */
        fit->cv_mean = NAN;
        fit->cv_se = NAN;

        n_fits++;
    }

    path->n_fits = n_fits;
    free(lambda_seq);
    free(obs);
    free(Xs);
    free(status_s);
    free(beta);
    free(eta);
    free(exp_eta);
    free(active);
    free(ever_active);
    free(pf);
    free(lb);
    free(ub);
    /* x_mean and x_sd are owned by path */
    return path;

cox_cleanup:
    free(obs);
    free(Xs);
    free(status_s);
    free(x_mean);
    free(x_sd);
    free(beta);
    free(eta);
    free(exp_eta);
    free(active);
    free(ever_active);
    free(pf);
    free(lb);
    free(ub);
    return NULL;
}

/* ========== Serialization ========== */

/* Binary format: GAM1 magic + header + fits */
/* ========== Multi-task Lasso / Elastic Net ========== */
/* L1/L2 mixed norm: lambda * alpha * sum_j ||beta_j||_2 + lambda * (1-alpha)/2 * sum_j ||beta_j||_2^2
 * where beta_j is the coefficient vector for feature j across all tasks.
 * Ref: Obozinski, Taskar, Jordan (2010). Joint covariate selection and joint
 *      subspace selection for multiple classification problems. Statistics and
 *      Computing 20(2):231-252.
 *      Yuan & Lin (2006). Model selection and estimation in regression with
 *      grouped variables. JRSS-B 68(1):49-67. */

gam_path_t *gam_fit_multi(
    const double *X, int32_t nrow, int32_t ncol,
    const double *Y, int32_t n_tasks,
    const gam_params_t *params
) {
    if (!X || !Y || nrow < 1 || ncol < 1 || n_tasks < 2) {
        gam_set_error("gam_fit_multi: invalid input (need n_tasks >= 2)");
        return NULL;
    }

    int32_t penalty = params->penalty;
    double alpha = params->alpha;
    int32_t p = ncol;

    /* Standardize X */
    double *Xs = (double *)malloc((size_t)nrow * (size_t)p * sizeof(double));
    double *x_mean = (double *)calloc((size_t)p, sizeof(double));
    double *x_sd = (double *)malloc((size_t)p * sizeof(double));
    if (!Xs || !x_mean || !x_sd) {
        free(Xs); free(x_mean); free(x_sd);
        gam_set_error("gam_fit_multi: alloc failed");
        return NULL;
    }
    memcpy(Xs, X, (size_t)nrow * (size_t)p * sizeof(double));

    if (params->standardize) {
        for (int32_t j = 0; j < p; j++) {
            double sum = 0.0;
            for (int32_t i = 0; i < nrow; i++) sum += Xs[i * p + j];
            x_mean[j] = sum / nrow;
        }
        for (int32_t j = 0; j < p; j++) {
            double ss = 0.0;
            for (int32_t i = 0; i < nrow; i++) {
                double d = Xs[i * p + j] - x_mean[j];
                ss += d * d;
            }
            x_sd[j] = sqrt(ss / nrow);
            if (x_sd[j] < 1e-10) x_sd[j] = 1.0;
        }
        for (int32_t i = 0; i < nrow; i++) {
            for (int32_t j = 0; j < p; j++) {
                Xs[i * p + j] = (Xs[i * p + j] - x_mean[j]) / x_sd[j];
            }
        }
    } else {
        for (int32_t j = 0; j < p; j++) { x_mean[j] = 0.0; x_sd[j] = 1.0; }
    }

    /* Center and scale Y per task */
    double *Yw = (double *)malloc((size_t)nrow * (size_t)n_tasks * sizeof(double));
    double *y_mean_arr = (double *)calloc((size_t)n_tasks, sizeof(double));
    double *y_sd_arr = (double *)malloc((size_t)n_tasks * sizeof(double));
    if (!Yw || !y_mean_arr || !y_sd_arr) {
        free(Xs); free(x_mean); free(x_sd); free(Yw); free(y_mean_arr); free(y_sd_arr);
        gam_set_error("gam_fit_multi: alloc failed");
        return NULL;
    }

    for (int32_t t = 0; t < n_tasks; t++) {
        if (params->standardize) {
            double sum = 0.0;
            for (int32_t i = 0; i < nrow; i++) sum += Y[i * n_tasks + t];
            y_mean_arr[t] = sum / nrow;
            double ss = 0.0;
            for (int32_t i = 0; i < nrow; i++) {
                double d = Y[i * n_tasks + t] - y_mean_arr[t];
                ss += d * d;
            }
            y_sd_arr[t] = sqrt(ss / nrow);
            if (y_sd_arr[t] < 1e-10) y_sd_arr[t] = 1.0;
            for (int32_t i = 0; i < nrow; i++) {
                Yw[i * n_tasks + t] = (Y[i * n_tasks + t] - y_mean_arr[t]) / y_sd_arr[t];
            }
        } else {
            y_mean_arr[t] = 0.0;
            y_sd_arr[t] = 1.0;
            for (int32_t i = 0; i < nrow; i++) {
                Yw[i * n_tasks + t] = Y[i * n_tasks + t];
            }
        }
    }

    /* Allocate working arrays */
    double *beta = (double *)calloc((size_t)p * (size_t)n_tasks, sizeof(double));
    double *r = (double *)malloc((size_t)nrow * (size_t)n_tasks * sizeof(double));
    double *intercept = (double *)calloc((size_t)n_tasks, sizeof(double));
    double *xw_sq = (double *)malloc((size_t)p * sizeof(double));
    double *pf = (double *)malloc((size_t)p * sizeof(double));
    int32_t *active = (int32_t *)calloc((size_t)p, sizeof(int32_t));
    if (!beta || !r || !intercept || !xw_sq || !pf || !active) {
        gam_set_error("gam_fit_multi: alloc failed");
        goto cleanup_multi;
    }

    /* Initialize penalty factors */
    for (int32_t j = 0; j < p; j++) {
        pf[j] = (params->penalty_factor && j < ncol) ? params->penalty_factor[j] : 1.0;
    }

    /* Initialize residuals: r = Y - intercept (beta = 0) */
    for (int32_t i = 0; i < nrow; i++) {
        for (int32_t t = 0; t < n_tasks; t++) {
            r[i * n_tasks + t] = Yw[i * n_tasks + t];
        }
    }

    /* Compute x^2 sums (unit weights, Gaussian) */
    for (int32_t j = 0; j < p; j++) {
        double s = 0.0;
        for (int32_t i = 0; i < nrow; i++) {
            double xij = Xs[i * p + j];
            s += xij * xij;
        }
        xw_sq[j] = s / nrow;
    }

    /* Compute lambda_max for multi-task:
     * lambda_max = max_j (1/alpha) * ||grad_j||_2 / pf_j
     * where grad_j = (1/n) * X_j^T * Y for task vector */
    double lambda_max = 0.0;
    {
        double *z_tmp = (double *)malloc((size_t)n_tasks * sizeof(double));
        for (int32_t j = 0; j < p; j++) {
            if (pf[j] == 0.0) continue;
            double norm_sq = 0.0;
            for (int32_t t = 0; t < n_tasks; t++) {
                double g = 0.0;
                for (int32_t i = 0; i < nrow; i++) {
                    g += Xs[i * p + j] * r[i * n_tasks + t];
                }
                z_tmp[t] = g / nrow;
                norm_sq += z_tmp[t] * z_tmp[t];
            }
            double scaled = sqrt(norm_sq) / fmax(pf[j], 1e-15);
            if (scaled > lambda_max) lambda_max = scaled;
        }
        free(z_tmp);
    }
    if (alpha > 0) lambda_max /= fmax(alpha, 1e-10);

    /* Lambda sequence */
    double lmr = params->lambda_min_ratio;
    if (lmr <= 0) lmr = (nrow >= p) ? 1e-4 : 1e-2;
    int32_t n_lambda = params->n_lambda > 0 ? params->n_lambda : 100;
    double *lambda_seq = (double *)malloc((size_t)n_lambda * sizeof(double));
    if (!lambda_seq) { gam_set_error("gam_fit_multi: alloc"); goto cleanup_multi; }

    if (params->lambda && params->n_lambda_user > 0) {
        n_lambda = params->n_lambda_user;
        free(lambda_seq);
        lambda_seq = (double *)malloc((size_t)n_lambda * sizeof(double));
        memcpy(lambda_seq, params->lambda, (size_t)n_lambda * sizeof(double));
    } else {
        double log_max = log(fmax(lambda_max, 1e-15));
        double log_min = log(fmax(lambda_max * lmr, 1e-15));
        for (int32_t k = 0; k < n_lambda; k++) {
            double frac = (n_lambda > 1) ? (double)k / (n_lambda - 1) : 0.0;
            lambda_seq[k] = exp(log_max + frac * (log_min - log_max));
        }
    }

    /* Null deviance: sum of squared residuals across all tasks */
    double null_dev = 0.0;
    for (int32_t i = 0; i < nrow; i++) {
        for (int32_t t = 0; t < n_tasks; t++) {
            null_dev += Yw[i * n_tasks + t] * Yw[i * n_tasks + t];
        }
    }

    /* Allocate result */
    int32_t n_coefs_total = (p + 1) * n_tasks;
    gam_path_t *path = (gam_path_t *)calloc(1, sizeof(gam_path_t));
    if (!path) { free(lambda_seq); gam_set_error("gam_fit_multi: alloc"); goto cleanup_multi; }
    path->fits = (gam_fit_t *)calloc((size_t)n_lambda, sizeof(gam_fit_t));
    if (!path->fits) { free(path); free(lambda_seq); gam_set_error("gam_fit_multi: alloc"); goto cleanup_multi; }

    path->n_features = ncol;
    path->n_coefs = n_coefs_total;
    path->family = GAM_FAMILY_GAUSSIAN;
    path->link = GAM_LINK_IDENTITY;
    path->penalty = penalty;
    path->alpha = alpha;
    path->n_tasks = n_tasks;
    path->n_basis_total = p;
    path->x_mean = x_mean; x_mean = NULL;
    path->x_sd = x_sd; x_sd = NULL;
    path->y_mean = 0.0;  /* not used for multi-task, per-task means in unstd */
    path->y_sd = 1.0;
    path->idx_min = -1;
    path->idx_1se = -1;

    /* Temporary buffer for z_j vector across tasks */
    double *z_buf = (double *)malloc((size_t)n_tasks * sizeof(double));
    if (!z_buf) { gam_free(path); free(lambda_seq); gam_set_error("gam_fit_multi: alloc"); goto cleanup_multi; }

    /* ---- Fit along the path ---- */
    int32_t n_fits = 0;
    double tol = params->tol > 0 ? params->tol : 1e-7;
    int32_t max_iter = params->max_iter > 0 ? params->max_iter : 10000;

    for (int32_t k = 0; k < n_lambda; k++) {
        double lambda = lambda_seq[k];
        int32_t iter = 0;

        /* Strong screening (simple: activate based on gradient magnitude) */
        if (k > 0) {
            for (int32_t j = 0; j < p; j++) {
                double norm_sq = 0.0;
                for (int32_t t = 0; t < n_tasks; t++) {
                    double g = 0.0;
                    for (int32_t i = 0; i < nrow; i++) {
                        g += Xs[i * p + j] * r[i * n_tasks + t];
                    }
                    g /= nrow;
                    norm_sq += g * g;
                }
                double grad_norm = sqrt(norm_sq);
                double thresh = alpha * lambda * pf[j];
                /* Strong rule: feature is active if gradient > 2*lambda - lambda_prev */
                active[j] = (grad_norm >= fmax(thresh - (lambda_seq[k-1] - lambda) * alpha * pf[j], 0.0)) ? 1 : 0;
                /* Always keep already-nonzero features active */
                for (int32_t t = 0; t < n_tasks; t++) {
                    if (beta[j * n_tasks + t] != 0.0) { active[j] = 1; break; }
                }
            }
        } else {
            for (int32_t j = 0; j < p; j++) active[j] = 1;
        }

        /* Multi-task coordinate descent */
        for (iter = 0; iter < max_iter; iter++) {
            double max_change = 0.0;

            for (int32_t j = 0; j < p; j++) {
                if (!active[j]) continue;

                /* Compute z_j vector across tasks */
                double z_norm_sq = 0.0;
                for (int32_t t = 0; t < n_tasks; t++) {
                    double zt = 0.0;
                    for (int32_t i = 0; i < nrow; i++) {
                        zt += Xs[i * p + j] * r[i * n_tasks + t];
                    }
                    zt = zt / nrow + xw_sq[j] * beta[j * n_tasks + t];
                    z_buf[t] = zt;
                    z_norm_sq += zt * zt;
                }

                double v = xw_sq[j];
                if (v < 1e-15) continue;

                double z_norm = sqrt(z_norm_sq);
                double l1 = lambda * alpha * pf[j];
                double l2 = lambda * (1.0 - alpha) * pf[j];

                /* Group soft-threshold: if ||z|| > l1, shrink; else zero */
                double shrink;
                if (pf[j] == 0.0) {
                    shrink = 1.0 / v;
                } else if (z_norm > l1) {
                    shrink = (1.0 - l1 / z_norm) / (v + l2);
                } else {
                    shrink = 0.0;
                }

                for (int32_t t = 0; t < n_tasks; t++) {
                    double old_b = beta[j * n_tasks + t];
                    double new_b = shrink * z_buf[t];
                    if (new_b != old_b) {
                        double delta = new_b - old_b;
                        beta[j * n_tasks + t] = new_b;
                        for (int32_t i = 0; i < nrow; i++) {
                            r[i * n_tasks + t] -= delta * Xs[i * p + j];
                        }
                        double wchange = fabs(delta) * sqrt(v);
                        if (wchange > max_change) max_change = wchange;
                    }
                }
            }

            /* Update intercepts per task */
            if (params->fit_intercept) {
                for (int32_t t = 0; t < n_tasks; t++) {
                    double sum_r = 0.0;
                    for (int32_t i = 0; i < nrow; i++) {
                        sum_r += r[i * n_tasks + t];
                    }
                    double delta = sum_r / nrow;
                    if (fabs(delta) > 1e-15) {
                        intercept[t] += delta;
                        for (int32_t i = 0; i < nrow; i++) {
                            r[i * n_tasks + t] -= delta;
                        }
                        if (fabs(delta) > max_change) max_change = fabs(delta);
                    }
                }
            }

            /* Convergence */
            if (max_change < tol) {
                /* Full pass (all features) */
                double full_change = 0.0;
                for (int32_t j = 0; j < p; j++) {
                    if (active[j]) continue;
                    double z_norm_sq2 = 0.0;
                    for (int32_t t = 0; t < n_tasks; t++) {
                        double zt = 0.0;
                        for (int32_t i = 0; i < nrow; i++) {
                            zt += Xs[i * p + j] * r[i * n_tasks + t];
                        }
                        zt = zt / nrow;
                        z_norm_sq2 += zt * zt;
                    }
                    double z_norm2 = sqrt(z_norm_sq2);
                    double l1_2 = lambda * alpha * pf[j];
                    if (z_norm2 > l1_2) {
                        active[j] = 1;
                        full_change = 1.0;
                    }
                }
                if (full_change < tol) break;
            }
        }

        /* Store result */
        gam_fit_t *fit = &path->fits[n_fits];
        fit->beta = (double *)malloc((size_t)n_coefs_total * sizeof(double));
        if (!fit->beta) break;

        /* Unstandardize coefficients */
        for (int32_t t = 0; t < n_tasks; t++) {
            int32_t base = t * (p + 1);
            if (params->standardize) {
                fit->beta[base] = intercept[t] * y_sd_arr[t] + y_mean_arr[t];
                for (int32_t j = 0; j < p; j++) {
                    fit->beta[base + j + 1] = beta[j * n_tasks + t] * y_sd_arr[t] / path->x_sd[j];
                    fit->beta[base] -= fit->beta[base + j + 1] * path->x_mean[j];
                }
            } else {
                fit->beta[base] = intercept[t];
                for (int32_t j = 0; j < p; j++) {
                    fit->beta[base + j + 1] = beta[j * n_tasks + t];
                }
            }
        }

        fit->n_coefs = n_coefs_total;
        fit->lambda = lambda;
        fit->n_iter = iter;

        /* Deviance: sum of squared residuals scaled back */
        fit->deviance = 0.0;
        for (int32_t i = 0; i < nrow; i++) {
            for (int32_t t = 0; t < n_tasks; t++) {
                double ri = r[i * n_tasks + t] * y_sd_arr[t];
                fit->deviance += ri * ri;
            }
        }
        fit->null_deviance = null_dev;
        if (params->standardize) {
            /* Scale null deviance back */
            double nd = 0.0;
            for (int32_t t = 0; t < n_tasks; t++) {
                nd += null_dev * y_sd_arr[t] * y_sd_arr[t] / n_tasks;
            }
            fit->null_deviance = nd;
        }

        /* df: count features with any nonzero coefficient across tasks */
        fit->df = 0;
        for (int32_t j = 0; j < p; j++) {
            for (int32_t t = 0; t < n_tasks; t++) {
                if (beta[j * n_tasks + t] != 0.0) { fit->df++; break; }
            }
        }

        fit->cv_mean = NAN;
        fit->cv_se = NAN;

        n_fits++;
    }

    path->n_fits = n_fits;
    free(z_buf);
    free(lambda_seq);
    free(beta);
    free(r);
    free(intercept);
    free(xw_sq);
    free(pf);
    free(active);
    free(Xs);
    free(x_mean);  /* may be NULL if transferred to path */
    free(x_sd);
    free(Yw);
    free(y_mean_arr);
    free(y_sd_arr);
    return path;

cleanup_multi:
    free(beta); free(r); free(intercept); free(xw_sq); free(pf); free(active);
    free(Xs); free(x_mean); free(x_sd); free(Yw); free(y_mean_arr); free(y_sd_arr);
    return NULL;
}

/* Predict for multi-task model.
 * out: float64 array of length nrow * n_tasks, row-major. */
int gam_predict_multi(
    const gam_path_t *path, int32_t fit_idx,
    const double *X, int32_t nrow, int32_t ncol,
    double *out
) {
    if (!path || !X || !out) return -1;
    if (fit_idx < 0 || fit_idx >= path->n_fits) return -1;
    int32_t n_tasks = path->n_tasks;
    if (n_tasks < 2) return -1;
    int32_t p = path->n_features;
    if (ncol != p) return -1;

    const double *beta = path->fits[fit_idx].beta;

    for (int32_t i = 0; i < nrow; i++) {
        for (int32_t t = 0; t < n_tasks; t++) {
            int32_t base = t * (p + 1);
            double pred = beta[base];  /* intercept */
            for (int32_t j = 0; j < p; j++) {
                pred += beta[base + j + 1] * X[i * ncol + j];
            }
            out[i * n_tasks + t] = pred;
        }
    }
    return 0;
}

/* ========== Multinomial logistic regression ========== */
/*
 * Reference: Friedman, Hastie, Tibshirani (2010), Section 5.
 * "Regularization Paths for GLMs via Coordinate Descent."
 *
 * Symmetric multinomial logistic: K classes, p features.
 * Linear predictors: eta_k = beta_k0 + sum_j beta_kj * x_j
 * Probabilities: P(Y=k|x) = exp(eta_k) / sum_l exp(eta_l)
 * Loss: -sum_i sum_k y_ik * log(P_ik)
 *
 * Penalty (grouped across classes):
 *   sum_j pf_j * [alpha * ||beta_.j||_1 + (1-alpha)/2 * ||beta_.j||_2^2]
 * where beta_.j = (beta_1j, ..., beta_Kj) is the coefficient vector for
 * feature j across all K classes.
 *
 * Solver: outer Newton (IRLS) with inner coordinate descent.
 * For each class k and feature j, the quadratic approximation gives:
 *   z_kj = (1/n) sum_i x_ij * (y_ik - P_ik) + xw_sq_kj * beta_kj
 *   v_kj = (1/n) sum_i x_ij^2 * P_ik * (1 - P_ik)
 * Update: beta_kj = soft_threshold(z_kj, lambda*alpha*pf_j) / (v_kj + lambda*(1-alpha)*pf_j)
 *
 * Coefficient layout: K blocks of (p+1) values each.
 *   beta[k*(p+1) + 0] = intercept for class k
 *   beta[k*(p+1) + j+1] = coefficient for feature j, class k
 */

gam_path_t *gam_fit_multinomial(
    const double *X, int32_t nrow, int32_t ncol,
    const double *y, int32_t n_classes,
    const gam_params_t *params
) {
    if (!X || !y || !params || nrow <= 0 || ncol <= 0 || n_classes < 3) {
        gam_set_error("gam_fit_multinomial: invalid input (need n_classes >= 3)");
        return NULL;
    }

    int32_t n = nrow, p = ncol, K = n_classes;
    double alpha = (params->alpha >= 0.0 && params->alpha <= 1.0) ? params->alpha : 1.0;
    int32_t n_lambda = (params->n_lambda > 0) ? params->n_lambda : 100;
    int32_t max_iter = (params->max_iter > 0) ? params->max_iter : 10000;
    int32_t max_inner = (params->max_inner > 0) ? params->max_inner : 25;
    double tol = (params->tol > 0) ? params->tol : 1e-7;

    /* Encode y as indicator matrix Y: n x K */
    double *Y = (double *)calloc((size_t)n * K, sizeof(double));
    if (!Y) { gam_set_error("gam_fit_multinomial: alloc"); return NULL; }
    for (int32_t i = 0; i < n; i++) {
        int32_t cls = (int32_t)y[i];
        if (cls >= 0 && cls < K) Y[i * K + cls] = 1.0;
    }

    /* Standardize X */
    double *Xs = (double *)malloc((size_t)n * p * sizeof(double));
    double *x_mean = (double *)calloc(p, sizeof(double));
    double *x_sd = (double *)malloc((size_t)p * sizeof(double));
    if (!Xs || !x_mean || !x_sd) {
        free(Y); free(Xs); free(x_mean); free(x_sd);
        gam_set_error("gam_fit_multinomial: alloc"); return NULL;
    }

    if (params->standardize) {
        for (int32_t j = 0; j < p; j++) {
            double sum = 0.0, ss = 0.0;
            for (int32_t i = 0; i < n; i++) sum += X[i * p + j];
            x_mean[j] = sum / n;
            for (int32_t i = 0; i < n; i++) {
                double d = X[i * p + j] - x_mean[j];
                ss += d * d;
            }
            x_sd[j] = sqrt(ss / n);
            if (x_sd[j] < 1e-10) x_sd[j] = 1.0;
            for (int32_t i = 0; i < n; i++)
                Xs[i * p + j] = (X[i * p + j] - x_mean[j]) / x_sd[j];
        }
    } else {
        memcpy(Xs, X, (size_t)n * p * sizeof(double));
        for (int32_t j = 0; j < p; j++) { x_mean[j] = 0.0; x_sd[j] = 1.0; }
    }

    /* Penalty factors */
    double *pf = (double *)malloc((size_t)p * sizeof(double));
    if (!pf) {
        free(Y); free(Xs); free(x_mean); free(x_sd);
        gam_set_error("gam_fit_multinomial: alloc"); return NULL;
    }
    for (int32_t j = 0; j < p; j++)
        pf[j] = (params->penalty_factor && j < ncol) ? params->penalty_factor[j] : 1.0;

    /* Allocate coefficient and working arrays */
    double *beta = (double *)calloc((size_t)p * K, sizeof(double));  /* p x K, row-major */
    double *intercept = (double *)calloc(K, sizeof(double));
    double *prob = (double *)malloc((size_t)n * K * sizeof(double));  /* P(Y=k|x) */
    double *eta = (double *)calloc((size_t)n * K, sizeof(double));
    if (!beta || !intercept || !prob || !eta) {
        free(Y); free(Xs); free(x_mean); free(x_sd); free(pf);
        free(beta); free(intercept); free(prob); free(eta);
        gam_set_error("gam_fit_multinomial: alloc"); return NULL;
    }

    /* Initialize intercepts from class proportions */
    for (int32_t k = 0; k < K; k++) {
        double count = 0.0;
        for (int32_t i = 0; i < n; i++) count += Y[i * K + k];
        double pk = clamp(count / n, 0.01, 0.99);
        intercept[k] = log(pk);  /* log(p_k) */
    }
    /* Center intercepts (sum to zero constraint for identifiability) */
    double int_mean = 0.0;
    for (int32_t k = 0; k < K; k++) int_mean += intercept[k];
    int_mean /= K;
    for (int32_t k = 0; k < K; k++) intercept[k] -= int_mean;

    /* Compute initial eta and probabilities */
    for (int32_t i = 0; i < n; i++)
        for (int32_t k = 0; k < K; k++)
            eta[i * K + k] = intercept[k];

    /* Softmax helper: compute prob from eta (in-place numerically stable) */
    #define SOFTMAX_ROWS() do { \
        for (int32_t i = 0; i < n; i++) { \
            double max_eta = eta[i * K]; \
            for (int32_t k = 1; k < K; k++) \
                if (eta[i * K + k] > max_eta) max_eta = eta[i * K + k]; \
            double sum_exp = 0.0; \
            for (int32_t k = 0; k < K; k++) { \
                prob[i * K + k] = exp(eta[i * K + k] - max_eta); \
                sum_exp += prob[i * K + k]; \
            } \
            for (int32_t k = 0; k < K; k++) \
                prob[i * K + k] /= sum_exp; \
        } \
    } while(0)

    SOFTMAX_ROWS();

    /* Converge intercept-only model before computing lambda_max */
    if (params->fit_intercept) {
        for (int32_t it = 0; it < 100; it++) {
            double max_d = 0.0;
            for (int32_t k = 0; k < K; k++) {
                double grad = 0.0, hess = 0.0;
                for (int32_t i = 0; i < n; i++) {
                    grad += (Y[i * K + k] - prob[i * K + k]);
                    hess += prob[i * K + k] * (1.0 - prob[i * K + k]);
                }
                grad /= n; hess /= n;
                if (hess < 1e-12) continue;
                double d = grad / hess;
                intercept[k] += d;
                for (int32_t i = 0; i < n; i++) eta[i * K + k] += d;
                if (fabs(d) > max_d) max_d = fabs(d);
            }
            /* Re-center */
            double imean = 0.0;
            for (int32_t k = 0; k < K; k++) imean += intercept[k];
            imean /= K;
            for (int32_t k = 0; k < K; k++) {
                intercept[k] -= imean;
                for (int32_t i = 0; i < n; i++) eta[i * K + k] -= imean;
            }
            SOFTMAX_ROWS();
            if (max_d < 1e-10) break;
        }
    }

    /* Compute lambda_max: max over features of ||gradient_j||_inf / alpha */
    double lambda_max = 0.0;
    for (int32_t j = 0; j < p; j++) {
        if (pf[j] <= 0.0) continue;
        for (int32_t k = 0; k < K; k++) {
            double grad = 0.0;
            for (int32_t i = 0; i < n; i++)
                grad += Xs[i * p + j] * (Y[i * K + k] - prob[i * K + k]);
            grad /= n;
            double ag = fabs(grad) / pf[j];
            if (ag > lambda_max) lambda_max = ag;
        }
    }
    if (alpha > 0.0) lambda_max /= alpha;
    else lambda_max = 1.0;

    /* Generate lambda sequence */
    double lmr = (params->lambda_min_ratio > 0) ? params->lambda_min_ratio
                 : (n >= p ? 1e-4 : 1e-2);
    double *lambda_seq = (double *)malloc((size_t)n_lambda * sizeof(double));
    if (!lambda_seq) {
        gam_set_error("gam_fit_multinomial: alloc");
        goto cleanup_mn;
    }
    if (params->lambda && params->n_lambda_user > 0) {
        n_lambda = params->n_lambda_user;
        free(lambda_seq);
        lambda_seq = (double *)malloc((size_t)n_lambda * sizeof(double));
        memcpy(lambda_seq, params->lambda, (size_t)n_lambda * sizeof(double));
    } else {
        double log_lmax = log(lambda_max);
        double log_lmin = log(lambda_max * lmr);
        for (int32_t k = 0; k < n_lambda; k++)
            lambda_seq[k] = exp(log_lmax - (double)k / (n_lambda - 1) * (log_lmax - log_lmin));
    }

    /* Allocate path */
    int32_t n_coefs_total = K * (p + 1);
    gam_path_t *path = (gam_path_t *)calloc(1, sizeof(gam_path_t));
    if (!path) { free(lambda_seq); gam_set_error("gam_fit_multinomial: alloc"); goto cleanup_mn; }
    path->fits = (gam_fit_t *)calloc((size_t)n_lambda, sizeof(gam_fit_t));
    if (!path->fits) { free(path); free(lambda_seq); gam_set_error("gam_fit_multinomial: alloc"); goto cleanup_mn; }

    path->n_features = p;
    path->n_coefs = n_coefs_total;
    path->family = GAM_FAMILY_MULTINOMIAL;
    path->link = GAM_LINK_LOGIT;
    path->penalty = params->penalty;
    path->alpha = alpha;
    path->n_tasks = K;  /* reuse n_tasks to store n_classes */
    path->n_basis_total = p;

    /* ---- Fit along the path ---- */
    int32_t n_fits = 0;

    for (int32_t lk = 0; lk < n_lambda; lk++) {
        double lambda = lambda_seq[lk];
        int32_t iter = 0;

        /* Outer IRLS loop */
        for (int32_t outer = 0; outer < max_inner; outer++) {
            /* Compute eta and probabilities */
            for (int32_t i = 0; i < n; i++) {
                for (int32_t k = 0; k < K; k++) {
                    double e = intercept[k];
                    for (int32_t j = 0; j < p; j++)
                        e += Xs[i * p + j] * beta[j * K + k];
                    eta[i * K + k] = e;
                }
            }
            SOFTMAX_ROWS();

            /* Inner CD loop */
            double max_change = 0.0;
            for (int32_t cd_iter = 0; cd_iter < max_iter; cd_iter++) {
                max_change = 0.0;
                iter++;

                /* Update intercepts */
                if (params->fit_intercept) {
                    for (int32_t k = 0; k < K; k++) {
                        double grad = 0.0;
                        for (int32_t i = 0; i < n; i++)
                            grad += (Y[i * K + k] - prob[i * K + k]);
                        grad /= n;
                        double hess = 0.0;
                        for (int32_t i = 0; i < n; i++)
                            hess += prob[i * K + k] * (1.0 - prob[i * K + k]);
                        hess /= n;
                        if (hess < 1e-12) continue;
                        double delta = grad / hess;
                        intercept[k] += delta;
                        /* Update eta and prob */
                        for (int32_t i = 0; i < n; i++)
                            eta[i * K + k] += delta;
                        double ad = fabs(delta);
                        if (ad > max_change) max_change = ad;
                    }
                    /* Re-center intercepts */
                    double imean = 0.0;
                    for (int32_t k = 0; k < K; k++) imean += intercept[k];
                    imean /= K;
                    for (int32_t k = 0; k < K; k++) {
                        double delta = -imean;
                        intercept[k] += delta;
                        for (int32_t i = 0; i < n; i++)
                            eta[i * K + k] += delta;
                    }
                    SOFTMAX_ROWS();
                }

                /* Update coefficients feature by feature */
                for (int32_t j = 0; j < p; j++) {
                    for (int32_t k = 0; k < K; k++) {
                        double old_b = beta[j * K + k];

                        /* Compute gradient and Hessian for (j, k) */
                        double grad = 0.0, hess = 0.0;
                        for (int32_t i = 0; i < n; i++) {
                            double xij = Xs[i * p + j];
                            double r_ik = Y[i * K + k] - prob[i * K + k];
                            double w_ik = prob[i * K + k] * (1.0 - prob[i * K + k]);
                            grad += xij * r_ik;
                            hess += xij * xij * w_ik;
                        }
                        grad /= n;
                        hess /= n;

                        double z = grad + hess * old_b;
                        double v = hess;

                        if (v < 1e-15) continue;

                        /* Elastic net penalty per coordinate */
                        double new_b;
                        double lam_l1 = lambda * alpha * pf[j];
                        double lam_l2 = lambda * (1.0 - alpha) * pf[j];

                        double s = (z > 0) ? z - lam_l1 : z + lam_l1;
                        if (fabs(z) <= lam_l1) {
                            new_b = 0.0;
                        } else {
                            new_b = s / (v + lam_l2);
                        }

                        if (new_b != old_b) {
                            double delta = new_b - old_b;
                            beta[j * K + k] = new_b;
                            /* Update eta */
                            for (int32_t i = 0; i < n; i++)
                                eta[i * K + k] += Xs[i * p + j] * delta;
                            double ad = fabs(delta);
                            if (ad > max_change) max_change = ad;
                        }
                    }
                    /* Recompute probabilities after each feature update */
                    SOFTMAX_ROWS();
                }

                if (max_change < tol) break;
            }

            /* Check outer convergence */
            if (max_change < tol) break;
        }

        /* Store result */
        gam_fit_t *fit = &path->fits[n_fits];
        fit->beta = (double *)malloc((size_t)n_coefs_total * sizeof(double));
        if (!fit->beta) break;

        /* Unstandardize coefficients */
        for (int32_t k = 0; k < K; k++) {
            int32_t base = k * (p + 1);
            fit->beta[base] = intercept[k];
            for (int32_t j = 0; j < p; j++) {
                if (params->standardize) {
                    fit->beta[base + j + 1] = beta[j * K + k] / x_sd[j];
                    fit->beta[base] -= fit->beta[base + j + 1] * x_mean[j];
                } else {
                    fit->beta[base + j + 1] = beta[j * K + k];
                }
            }
        }

        fit->n_coefs = n_coefs_total;
        fit->lambda = lambda;
        fit->n_iter = iter;

        /* Deviance: -2 * log-likelihood */
        fit->deviance = 0.0;
        for (int32_t i = 0; i < n; i++) {
            for (int32_t k = 0; k < K; k++) {
                if (Y[i * K + k] > 0.5)
                    fit->deviance -= 2.0 * log(fmax(prob[i * K + k], 1e-15));
            }
        }

        /* Null deviance: -2 * log-likelihood under intercept-only model */
        if (n_fits == 0) {
            double nd = 0.0;
            for (int32_t i = 0; i < n; i++) {
                for (int32_t k = 0; k < K; k++) {
                    if (Y[i * K + k] > 0.5) {
                        double pk = 0.0;
                        for (int32_t ii = 0; ii < n; ii++) pk += Y[ii * K + k];
                        pk /= n;
                        nd -= 2.0 * log(fmax(pk, 1e-15));
                    }
                }
            }
            fit->null_deviance = nd;
        } else {
            fit->null_deviance = path->fits[0].null_deviance;
        }

        /* Count active features (nonzero across any class) */
        fit->df = 0;
        for (int32_t j = 0; j < p; j++) {
            int any_nz = 0;
            for (int32_t k = 0; k < K; k++) {
                if (beta[j * K + k] != 0.0) { any_nz = 1; break; }
            }
            fit->df += any_nz;
        }

        fit->cv_mean = NAN;
        fit->cv_se = NAN;

        n_fits++;
    }

    path->n_fits = n_fits;
    path->x_mean = x_mean; x_mean = NULL;
    path->x_sd = x_sd; x_sd = NULL;
    path->idx_min = -1;
    path->idx_1se = -1;

    free(lambda_seq);
    free(beta); free(intercept); free(prob); free(eta);
    free(Y); free(Xs); free(x_mean); free(x_sd); free(pf);
    return path;

cleanup_mn:
    free(beta); free(intercept); free(prob); free(eta);
    free(Y); free(Xs); free(x_mean); free(x_sd); free(pf);
    return NULL;
}

#undef SOFTMAX_ROWS

/* Predict class probabilities for multinomial model. */
int gam_predict_multinomial(
    const gam_path_t *path, int32_t fit_idx,
    const double *X, int32_t nrow, int32_t ncol,
    double *out
) {
    if (!path || !X || !out) return -1;
    if (fit_idx < 0 || fit_idx >= path->n_fits) return -1;
    if (path->family != GAM_FAMILY_MULTINOMIAL) {
        gam_set_error("predict_multinomial: not a multinomial model");
        return -1;
    }
    int32_t K = path->n_tasks;
    int32_t p = path->n_features;
    if (K < 3 || ncol != p) return -1;

    const double *beta = path->fits[fit_idx].beta;

    /* Compute linear predictors and softmax */
    for (int32_t i = 0; i < nrow; i++) {
        double max_eta = -1e30;
        for (int32_t k = 0; k < K; k++) {
            int32_t base = k * (p + 1);
            double e = beta[base];  /* intercept */
            for (int32_t j = 0; j < p; j++)
                e += beta[base + j + 1] * X[i * ncol + j];
            out[i * K + k] = e;
            if (e > max_eta) max_eta = e;
        }
        /* Softmax */
        double sum_exp = 0.0;
        for (int32_t k = 0; k < K; k++) {
            out[i * K + k] = exp(out[i * K + k] - max_eta);
            sum_exp += out[i * K + k];
        }
        for (int32_t k = 0; k < K; k++)
            out[i * K + k] /= sum_exp;
    }
    return 0;
}

/* ========== GAMLSS (distributional regression) ========== */
/*
 * Reference: Rigby & Stasinopoulos (2005), "Generalized additive models
 * for location, scale and shape." JRSS-C, 54(3):507-554.
 *
 * RS algorithm: alternating IRLS for each distributional parameter.
 * For a 2-parameter distribution (mu, sigma), at each lambda:
 *   1. Fix sigma, update mu via penalized WLS (CD)
 *   2. Fix mu, update sigma via penalized WLS (CD)
 *   3. Check deviance convergence
 *
 * Coefficient layout: 2 * (p+1) per fit.
 *   [0..p] = mu coefficients (intercept at [0])
 *   [p+1..2p+1] = sigma/phi coefficients (intercept at [p+1])
 */

/* --- Normal(mu, sigma) --- */

static void gamlss_normal_init(const double *y, int32_t n,
                                double *mu, double *sigma) {
    double sum = 0.0;
    for (int32_t i = 0; i < n; i++) sum += y[i];
    double m = sum / n;
    double ss = 0.0;
    for (int32_t i = 0; i < n; i++) {
        double d = y[i] - m;
        ss += d * d;
    }
    double s = sqrt(ss / n);
    if (s < 1e-10) s = 1.0;
    for (int32_t i = 0; i < n; i++) {
        mu[i] = m;
        sigma[i] = s;
    }
}

static void gamlss_normal_wz_mu(const double *y, int32_t n,
                                 const double *mu, const double *sigma,
                                 const double *eta_mu,
                                 double *z, double *w) {
    /* mu link = identity: dmu/deta = 1
     * w_mu = 1/sigma^2, z_mu = eta_mu + (y - mu) = y */
    for (int32_t i = 0; i < n; i++) {
        double s2 = sigma[i] * sigma[i];
        w[i] = 1.0 / fmax(s2, 1e-30);
        z[i] = y[i]; /* identity link: z = eta + (y-mu)/1 = mu + (y-mu) = y */
    }
    (void)mu; (void)eta_mu;
}

static void gamlss_normal_wz_sigma(const double *y, int32_t n,
                                    const double *mu, const double *sigma,
                                    const double *eta_sigma,
                                    double *z, double *w) {
    /* sigma link = log: eta_sigma = log(sigma), sigma = exp(eta_sigma)
     * dl/d(eta_sigma) = -1 + (y-mu)^2/sigma^2
     * d2l/d(eta_sigma)^2 = 2 (Fisher info)
     * z = eta_sigma + (dl/d(eta_sigma)) / d2l/d(eta_sigma)^2 */
    for (int32_t i = 0; i < n; i++) {
        double r = (y[i] - mu[i]) / fmax(sigma[i], 1e-30);
        double dl = clamp(-1.0 + r * r, -10.0, 10.0);
        w[i] = 2.0;
        z[i] = clamp(eta_sigma[i] + dl / 2.0, -10.0, 10.0);
    }
}

static double gamlss_normal_deviance(const double *y, int32_t n,
                                      const double *mu, const double *sigma) {
    /* -2 * loglik = sum((y-mu)^2/sigma^2 + 2*log(sigma) + log(2*pi)) */
    double dev = 0.0;
    for (int32_t i = 0; i < n; i++) {
        double r = (y[i] - mu[i]) / fmax(sigma[i], 1e-30);
        dev += r * r + 2.0 * log(fmax(sigma[i], 1e-30));
    }
    return dev;
}

/* --- Gamma(mu, sigma) --- */
/* sigma = CV (coefficient of variation), shape = 1/sigma^2, scale = mu*sigma^2 */

static void gamlss_gamma_init(const double *y, int32_t n,
                               double *mu, double *sigma) {
    double sum = 0.0;
    for (int32_t i = 0; i < n; i++) sum += fmax(y[i], 1e-10);
    double m = sum / n;
    double ss = 0.0;
    for (int32_t i = 0; i < n; i++) {
        double d = fmax(y[i], 1e-10) - m;
        ss += d * d;
    }
    double cv = sqrt(ss / n) / fmax(m, 1e-10);
    if (cv < 0.01) cv = 0.5;
    for (int32_t i = 0; i < n; i++) {
        mu[i] = m;
        sigma[i] = cv;
    }
}

static void gamlss_gamma_wz_mu(const double *y, int32_t n,
                                const double *mu, const double *sigma,
                                const double *eta_mu,
                                double *z, double *w) {
    /* mu link = log: eta_mu = log(mu), mu = exp(eta_mu)
     * dmu/deta = mu
     * Var(y) = mu^2 * sigma^2
     * Fisher w_mu = (dmu/deta)^2 / Var = mu^2 / (mu^2 * sigma^2) = 1/sigma^2
     * z = eta_mu + (y - mu) / (dmu/deta) = eta_mu + (y - mu) / mu */
    for (int32_t i = 0; i < n; i++) {
        double s2 = sigma[i] * sigma[i];
        w[i] = 1.0 / fmax(s2, 1e-30);
        double mu_i = fmax(mu[i], 1e-30);
        z[i] = eta_mu[i] + (fmax(y[i], 1e-10) - mu_i) / mu_i;
    }
}

static void gamlss_gamma_wz_sigma(const double *y, int32_t n,
                                   const double *mu, const double *sigma,
                                   const double *eta_sigma,
                                   double *z, double *w) {
    /* Gamma GAMLSS sigma step using the deviance-based approach.
     * For Gamma(shape=k, rate=k/mu): k = 1/sigma^2
     * dl/dsigma = (2/sigma^3)*[-digamma(k) - log(k) + log(y/mu) + 1 - y/mu + log(k)]
     * Simplified stable score for eta_sigma = log(sigma):
     * dl/d(eta_sigma) = 2*(-log(y/mu) + y/mu - 1) / sigma^2  -  2*(digamma(1/sigma^2) + log(sigma^2))/sigma^2
     *
     * Use an approach that avoids digamma:
     * The gamma deviance residual for observation i is d_i = 2*(-log(y/mu) + (y-mu)/mu)
     * and sigma^2 estimates the dispersion.
     * Working response: z = eta_sigma + (d_i/sigma^2 - 1) / 2
     * Working weight: 2 */
    for (int32_t i = 0; i < n; i++) {
        double yi = fmax(y[i], 1e-10);
        double mu_i = fmax(mu[i], 1e-30);
        double s2 = fmax(sigma[i] * sigma[i], 1e-30);
        double ratio = yi / mu_i;
        /* Gamma deviance residual component */
        double d_i = 2.0 * (-log(ratio) + ratio - 1.0);
        /* Score: dl/d(eta_sigma) = d_i/sigma^2 - 1 */
        double dl = clamp(d_i / s2 - 1.0, -5.0, 5.0);
        w[i] = 2.0;
        z[i] = eta_sigma[i] + dl / 2.0;
    }
}

static double gamlss_gamma_deviance(const double *y, int32_t n,
                                     const double *mu, const double *sigma) {
    /* -2 * loglik (up to constant) for Gamma(shape=1/sigma^2, scale=mu*sigma^2)
     * = sum_i [ 2*(-log(y_i/mu_i) + (y_i - mu_i)/mu_i) / sigma^2
     *           + 2*log(sigma^2) + 2*lgamma(1/sigma^2) - 2*(1/sigma^2)*log(1/sigma^2)
     *           + 2/sigma^2 * log(mu_i) + 2*log(y_i) ] */
    double dev = 0.0;
    for (int32_t i = 0; i < n; i++) {
        double yi = fmax(y[i], 1e-10);
        double mu_i = fmax(mu[i], 1e-30);
        double s2 = sigma[i] * sigma[i];
        double shape = 1.0 / fmax(s2, 1e-30);
        /* Gamma deviance contribution */
        double ratio = yi / mu_i;
        dev += 2.0 * (-log(ratio) + ratio - 1.0) / fmax(s2, 1e-30);
        dev += 2.0 * log(fmax(s2, 1e-30));
        dev += 2.0 * lgamma(shape) - 2.0 * shape * log(shape);
    }
    return dev;
}

/* --- Beta(mu, phi) --- */
/* mu = mean (0,1), phi = precision (>0), Var = mu*(1-mu)/(1+phi) */

static void gamlss_beta_init(const double *y, int32_t n,
                              double *mu, double *phi) {
    double sum = 0.0;
    for (int32_t i = 0; i < n; i++) sum += clamp(y[i], 0.001, 0.999);
    double m = sum / n;
    double ss = 0.0;
    for (int32_t i = 0; i < n; i++) {
        double d = clamp(y[i], 0.001, 0.999) - m;
        ss += d * d;
    }
    double var = ss / n;
    /* phi ~ mu*(1-mu)/var - 1 */
    double phi_init = fmax(m * (1.0 - m) / fmax(var, 1e-10) - 1.0, 1.0);
    for (int32_t i = 0; i < n; i++) {
        mu[i] = m;
        phi[i] = phi_init;
    }
}

static void gamlss_beta_wz_mu(const double *y, int32_t n,
                               const double *mu, const double *phi,
                               const double *eta_mu,
                               double *z, double *w) {
    /* mu link = logit: eta_mu = log(mu/(1-mu)), mu = 1/(1+exp(-eta))
     * dmu/deta = mu*(1-mu)
     * dl/dmu = phi * (digamma(mu*phi) - digamma((1-mu)*phi) - log(y) + log(1-y))
     *        But for scoring: use Fisher info.
     * Fisher info for mu: phi * (trigamma(mu*phi) + trigamma((1-mu)*phi)) - simplify
     * Use the working weight: w = phi * mu*(1-mu)
     * z = eta + (y* - mu) / (mu*(1-mu))
     * where y* = (log(y/(1-y)) - digamma(mu*phi) + digamma((1-mu)*phi)) / phi ... too complex.
     *
     * Simpler approach: standard IRLS for Beta regression with logit link.
     * w = phi * mu*(1-mu) [the expected Fisher info for logit]
     * Working response: z = eta + (y - mu) / (mu*(1-mu)) */
    for (int32_t i = 0; i < n; i++) {
        double mu_i = clamp(mu[i], 1e-6, 1.0 - 1e-6);
        double v = mu_i * (1.0 - mu_i);
        w[i] = fmax(phi[i], 1e-10) * v;
        z[i] = eta_mu[i] + (clamp(y[i], 0.001, 0.999) - mu_i) / v;
    }
}

static void gamlss_beta_wz_phi(const double *y, int32_t n,
                                const double *mu, const double *phi,
                                const double *eta_phi,
                                double *z, double *w) {
    /* phi link = log: eta_phi = log(phi), phi = exp(eta_phi)
     * dl/dphi = digamma(phi) - mu*digamma(mu*phi) - (1-mu)*digamma((1-mu)*phi)
     *         + mu*log(y) + (1-mu)*log(1-y)
     * dl/d(eta_phi) = phi * dl/dphi  (chain rule for log link)
     * Fisher info for log(phi): phi^2 * (trigamma(phi) - mu^2*trigamma(mu*phi)
     *                            - (1-mu)^2*trigamma((1-mu)*phi))
     *
     * Use an approximation for numerical stability:
     * d2l/d(eta_phi)^2 ~ phi^2 * trigamma(phi) as leading term */
    for (int32_t i = 0; i < n; i++) {
        double mu_i = clamp(mu[i], 1e-6, 1.0 - 1e-6);
        double phi_i = fmax(phi[i], 1e-10);
        double yi = clamp(y[i], 0.001, 0.999);

        /* dl/dphi using digamma */
        double a = mu_i * phi_i;
        double b = (1.0 - mu_i) * phi_i;
        /* Approximate digamma(x) ~ log(x) - 1/(2x) for large x, or use series */
        /* For simplicity, use a robust but approximate approach */
        double dl_dphi = log(phi_i) - mu_i * log(fmax(a, 1e-30)) -
                          (1.0 - mu_i) * log(fmax(b, 1e-30)) +
                          mu_i * log(fmax(yi, 1e-30)) +
                          (1.0 - mu_i) * log(fmax(1.0 - yi, 1e-30));
        /* Simpler approximation: Fisher weight for log(phi) */
        w[i] = 1.0;  /* approximate constant weight */
        double dl_deta = phi_i * dl_dphi;
        z[i] = clamp(eta_phi[i] + dl_deta / fmax(w[i], 1e-10), -10.0, 10.0);
    }
}

static double gamlss_beta_deviance(const double *y, int32_t n,
                                    const double *mu, const double *phi) {
    /* -2 * loglik = -2 * sum_i [lgamma(phi) - lgamma(mu*phi) - lgamma((1-mu)*phi)
     *               + (mu*phi-1)*log(y) + ((1-mu)*phi-1)*log(1-y)] */
    double dev = 0.0;
    for (int32_t i = 0; i < n; i++) {
        double mu_i = clamp(mu[i], 1e-6, 1.0 - 1e-6);
        double phi_i = fmax(phi[i], 1e-10);
        double yi = clamp(y[i], 0.001, 0.999);
        double a = mu_i * phi_i;
        double b = (1.0 - mu_i) * phi_i;
        double loglik = lgamma(phi_i) - lgamma(a) - lgamma(b)
                       + (a - 1.0) * log(yi) + (b - 1.0) * log(1.0 - yi);
        dev -= 2.0 * loglik;
    }
    return dev;
}

/* --- GAMLSS fitting --- */

gam_path_t *gam_fit_gamlss(
    const double *X, int32_t nrow, int32_t ncol,
    const double *y,
    int32_t distribution,
    const gam_params_t *params
) {
    if (!X || !y || !params || nrow <= 0 || ncol <= 0) {
        gam_set_error("gam_fit_gamlss: invalid input");
        return NULL;
    }
    if (distribution < 0 || distribution > 2) {
        gam_set_error("gam_fit_gamlss: invalid distribution (0=Normal, 1=Gamma, 2=Beta)");
        return NULL;
    }

    int32_t n = nrow, p = ncol;
    int32_t penalty = params->penalty;
    double alpha = params->alpha;
    double tol = params->tol > 0 ? params->tol : 1e-7;
    int32_t max_iter = params->max_iter > 0 ? params->max_iter : 10000;
    int32_t max_rs = 50;  /* RS outer iterations */

    /* Standardize X */
    double *Xs = (double *)malloc((size_t)n * (size_t)p * sizeof(double));
    double *x_mean = (double *)calloc((size_t)p, sizeof(double));
    double *x_sd = (double *)malloc((size_t)p * sizeof(double));
    if (!Xs || !x_mean || !x_sd) {
        free(Xs); free(x_mean); free(x_sd);
        gam_set_error("gam_fit_gamlss: alloc failed");
        return NULL;
    }
    memcpy(Xs, X, (size_t)n * (size_t)p * sizeof(double));

    if (params->standardize) {
        for (int32_t j = 0; j < p; j++) {
            double sum = 0.0;
            for (int32_t i = 0; i < n; i++) sum += Xs[i * p + j];
            x_mean[j] = sum / n;
        }
        for (int32_t j = 0; j < p; j++) {
            double ss = 0.0;
            for (int32_t i = 0; i < n; i++) {
                double d = Xs[i * p + j] - x_mean[j];
                ss += d * d;
            }
            x_sd[j] = sqrt(ss / n);
            if (x_sd[j] < 1e-10) x_sd[j] = 1.0;
        }
        for (int32_t i = 0; i < n; i++) {
            for (int32_t j = 0; j < p; j++) {
                Xs[i * p + j] = (Xs[i * p + j] - x_mean[j]) / x_sd[j];
            }
        }
    } else {
        for (int32_t j = 0; j < p; j++) { x_mean[j] = 0.0; x_sd[j] = 1.0; }
    }

    /* Allocate working arrays for both mu and sigma parameters */
    double *mu_val = (double *)malloc((size_t)n * sizeof(double));
    double *sigma_val = (double *)malloc((size_t)n * sizeof(double));
    double *eta_mu = (double *)malloc((size_t)n * sizeof(double));
    double *eta_sigma = (double *)malloc((size_t)n * sizeof(double));
    double *z_work = (double *)malloc((size_t)n * sizeof(double));
    double *w_work = (double *)malloc((size_t)n * sizeof(double));
    double *r_work = (double *)malloc((size_t)n * sizeof(double));
    double *beta_mu = (double *)calloc((size_t)p, sizeof(double));
    double *beta_sigma = (double *)calloc((size_t)p, sizeof(double));
    double *xw_sq = (double *)malloc((size_t)p * sizeof(double));
    double *pf = (double *)malloc((size_t)p * sizeof(double));
    int32_t *active_mu = (int32_t *)calloc((size_t)p, sizeof(int32_t));
    int32_t *active_sigma = (int32_t *)calloc((size_t)p, sizeof(int32_t));
    int32_t *ever_active = (int32_t *)calloc((size_t)p, sizeof(int32_t));
    double intercept_mu = 0.0, intercept_sigma = 0.0;

    if (!mu_val || !sigma_val || !eta_mu || !eta_sigma || !z_work || !w_work ||
        !r_work || !beta_mu || !beta_sigma || !xw_sq || !pf ||
        !active_mu || !active_sigma || !ever_active) {
        gam_set_error("gam_fit_gamlss: alloc failed");
        goto cleanup_gamlss;
    }

    /* Initialize penalty factors */
    for (int32_t j = 0; j < p; j++) {
        pf[j] = (params->penalty_factor && j < ncol) ? params->penalty_factor[j] : 1.0;
    }

    /* Initialize distributional parameters */
    if (distribution == GAMLSS_NORMAL) {
        gamlss_normal_init(y, n, mu_val, sigma_val);
    } else if (distribution == GAMLSS_GAMMA) {
        gamlss_gamma_init(y, n, mu_val, sigma_val);
    } else {
        gamlss_beta_init(y, n, mu_val, sigma_val);
    }

    /* Initialize eta from parameter values */
    /* mu link: Normal=identity, Gamma=log, Beta=logit */
    if (distribution == GAMLSS_NORMAL) {
        intercept_mu = mu_val[0];
        for (int32_t i = 0; i < n; i++) eta_mu[i] = mu_val[i];
    } else if (distribution == GAMLSS_GAMMA) {
        intercept_mu = log(fmax(mu_val[0], 1e-30));
        for (int32_t i = 0; i < n; i++) eta_mu[i] = log(fmax(mu_val[i], 1e-30));
    } else {
        double m = clamp(mu_val[0], 1e-6, 1.0 - 1e-6);
        intercept_mu = log(m / (1.0 - m));
        for (int32_t i = 0; i < n; i++) {
            m = clamp(mu_val[i], 1e-6, 1.0 - 1e-6);
            eta_mu[i] = log(m / (1.0 - m));
        }
    }
    /* sigma/phi link = log for all distributions */
    intercept_sigma = log(fmax(sigma_val[0], 1e-30));
    for (int32_t i = 0; i < n; i++)
        eta_sigma[i] = log(fmax(sigma_val[i], 1e-30));

    /* Set up cd_work_t (shared, will swap arrays between mu/sigma) */
    cd_work_t w;
    memset(&w, 0, sizeof(w));
    w.n = n;
    w.p = p;
    w.Xs = Xs;
    w.family = GAM_FAMILY_GAUSSIAN; /* not used directly; we compute our own z/w */
    w.link = GAM_LINK_IDENTITY;
    w.penalty = penalty;
    w.alpha = alpha;
    w.gamma_mcp = params->gamma_mcp;
    w.gamma_scad = params->gamma_scad;
    w.fit_intercept = params->fit_intercept;
    w.tol = tol;
    w.x_mean = x_mean;
    w.x_sd = x_sd;
    w.pf = pf;

    /* Allocate lb/ub for cd_work_t */
    w.lb = (double *)malloc((size_t)p * sizeof(double));
    w.ub = (double *)malloc((size_t)p * sizeof(double));
    if (!w.lb || !w.ub) {
        gam_set_error("gam_fit_gamlss: alloc failed");
        goto cleanup_gamlss;
    }
    for (int32_t j = 0; j < p; j++) { w.lb[j] = -1e30; w.ub[j] = 1e30; }

    /* Compute lambda_max from initial mu working response */
    if (distribution == GAMLSS_NORMAL) {
        gamlss_normal_wz_mu(y, n, mu_val, sigma_val, eta_mu, z_work, w_work);
    } else if (distribution == GAMLSS_GAMMA) {
        gamlss_gamma_wz_mu(y, n, mu_val, sigma_val, eta_mu, z_work, w_work);
    } else {
        gamlss_beta_wz_mu(y, n, mu_val, sigma_val, eta_mu, z_work, w_work);
    }

    /* Set cd_work_t for lambda_max computation */
    w.y = z_work;
    w.w = w_work;
    w.r = r_work;
    w.beta = beta_mu;
    w.intercept = intercept_mu;
    w.xw_sq = xw_sq;
    w.active = active_mu;
    w.ever_active = ever_active;
    w.eta = eta_mu;
    w.mu = mu_val;

    /* Compute r = z - intercept (beta = 0) */
    for (int32_t i = 0; i < n; i++)
        r_work[i] = z_work[i] - intercept_mu;

    /* Compute xw_sq = sum(w * x^2) / n */
    for (int32_t j = 0; j < p; j++) {
        double s = 0.0;
        for (int32_t i = 0; i < n; i++) {
            double xij = Xs[i * p + j];
            s += w_work[i] * xij * xij;
        }
        xw_sq[j] = s / n;
    }

    double lambda_max = compute_lambda_max(&w);

    /* Lambda sequence */
    double lmr = params->lambda_min_ratio;
    if (lmr <= 0) lmr = (n >= p) ? 1e-4 : 1e-2;
    int32_t n_lambda = params->n_lambda > 0 ? params->n_lambda : 100;
    double *lambda_seq = (double *)malloc((size_t)n_lambda * sizeof(double));
    if (!lambda_seq) { gam_set_error("gam_fit_gamlss: alloc"); goto cleanup_gamlss; }

    if (params->lambda && params->n_lambda_user > 0) {
        n_lambda = params->n_lambda_user;
        free(lambda_seq);
        lambda_seq = (double *)malloc((size_t)n_lambda * sizeof(double));
        memcpy(lambda_seq, params->lambda, (size_t)n_lambda * sizeof(double));
    } else {
        double log_lmax = log(fmax(lambda_max, 1e-30));
        double log_lmin = log_lmax + log(lmr);
        for (int32_t k = 0; k < n_lambda; k++) {
            lambda_seq[k] = exp(log_lmax - (double)k / (double)(n_lambda - 1) * (log_lmax - log_lmin));
        }
    }

    /* Allocate path result */
    int32_t n_coefs_total = 2 * (p + 1);
    gam_path_t *path = (gam_path_t *)calloc(1, sizeof(gam_path_t));
    if (!path) { free(lambda_seq); gam_set_error("gam_fit_gamlss: alloc"); goto cleanup_gamlss; }

    path->fits = (gam_fit_t *)calloc((size_t)n_lambda, sizeof(gam_fit_t));
    if (!path->fits) { free(path); free(lambda_seq); gam_set_error("gam_fit_gamlss: alloc"); goto cleanup_gamlss; }

    path->n_features = p;
    path->n_coefs = n_coefs_total;
    path->family = GAM_FAMILY_GAUSSIAN; /* base family (not used for GAMLSS dispatch) */
    path->link = GAM_LINK_IDENTITY;
    path->penalty = penalty;
    path->alpha = alpha;
    path->n_tasks = 2;
    path->family_gamlss = distribution + 1;  /* 1-indexed: 1=Normal, 2=Gamma, 3=Beta */
    path->idx_min = -1;
    path->idx_1se = -1;
    path->y_mean = 0.0;
    path->y_sd = 1.0;

    /* Transfer x_mean/x_sd to path (allocate copies) */
    path->x_mean = (double *)malloc((size_t)p * sizeof(double));
    path->x_sd = (double *)malloc((size_t)p * sizeof(double));
    if (path->x_mean && path->x_sd) {
        memcpy(path->x_mean, x_mean, (size_t)p * sizeof(double));
        memcpy(path->x_sd, x_sd, (size_t)p * sizeof(double));
    }

    /* ---- Fit along the path ---- */
    int32_t n_fits = 0;

    for (int32_t k = 0; k < n_lambda; k++) {
        double lambda = lambda_seq[k];

        /* RS outer loop */
        double dev_prev = 1e30;
        int32_t total_iter = 0;

        for (int32_t rs = 0; rs < max_rs; rs++) {
            /* ---- Step 1: Update mu ---- */
            if (distribution == GAMLSS_NORMAL) {
                gamlss_normal_wz_mu(y, n, mu_val, sigma_val, eta_mu, z_work, w_work);
            } else if (distribution == GAMLSS_GAMMA) {
                gamlss_gamma_wz_mu(y, n, mu_val, sigma_val, eta_mu, z_work, w_work);
            } else {
                gamlss_beta_wz_mu(y, n, mu_val, sigma_val, eta_mu, z_work, w_work);
            }

            /* Set up cd_work_t for mu */
            w.y = z_work;
            w.w = w_work;
            w.r = r_work;
            w.beta = beta_mu;
            w.intercept = intercept_mu;
            w.xw_sq = xw_sq;
            w.active = active_mu;
            w.ever_active = ever_active;

            /* Compute r = z - Xs*beta - intercept */
            for (int32_t i = 0; i < n; i++) {
                double pred = intercept_mu;
                for (int32_t j = 0; j < p; j++) {
                    pred += Xs[i * p + j] * beta_mu[j];
                }
                r_work[i] = z_work[i] - pred;
            }

            /* Compute xw_sq */
            for (int32_t j = 0; j < p; j++) {
                double s = 0.0;
                for (int32_t i = 0; i < n; i++) {
                    double xij = Xs[i * p + j];
                    s += w_work[i] * xij * xij;
                }
                xw_sq[j] = s / n;
            }

            /* CD passes for mu */
            for (int32_t cd_iter = 0; cd_iter < max_iter; cd_iter++) {
                double max_change = cd_pass(&w, lambda, 1);
                total_iter++;
                if (max_change < tol) {
                    /* Full pass */
                    double full_change = cd_pass(&w, lambda, 0);
                    total_iter++;
                    if (full_change < tol) break;
                }
            }

            intercept_mu = w.intercept;
            /* memcpy not needed -- beta_mu is w.beta pointer */

            /* Update eta_mu and mu_val */
            for (int32_t i = 0; i < n; i++) {
                double pred = intercept_mu;
                for (int32_t j = 0; j < p; j++) {
                    pred += Xs[i * p + j] * beta_mu[j];
                }
                eta_mu[i] = pred;
            }

            /* Apply mu inverse link (with clamping) */
            if (distribution == GAMLSS_NORMAL) {
                for (int32_t i = 0; i < n; i++) mu_val[i] = eta_mu[i];
            } else if (distribution == GAMLSS_GAMMA) {
                for (int32_t i = 0; i < n; i++) {
                    eta_mu[i] = clamp(eta_mu[i], -20.0, 20.0);
                    mu_val[i] = exp(eta_mu[i]);
                }
            } else {
                for (int32_t i = 0; i < n; i++) {
                    eta_mu[i] = clamp(eta_mu[i], -20.0, 20.0);
                    double e = exp(-eta_mu[i]);
                    mu_val[i] = 1.0 / (1.0 + e);
                }
            }

            /* ---- Step 2: Update sigma/phi ---- */
            if (distribution == GAMLSS_NORMAL) {
                gamlss_normal_wz_sigma(y, n, mu_val, sigma_val, eta_sigma, z_work, w_work);
            } else if (distribution == GAMLSS_GAMMA) {
                gamlss_gamma_wz_sigma(y, n, mu_val, sigma_val, eta_sigma, z_work, w_work);
            } else {
                gamlss_beta_wz_phi(y, n, mu_val, sigma_val, eta_sigma, z_work, w_work);
            }

            /* Set up cd_work_t for sigma */
            w.y = z_work;
            w.w = w_work;
            w.r = r_work;
            w.beta = beta_sigma;
            w.intercept = intercept_sigma;
            w.active = active_sigma;

            /* Compute r = z - Xs*beta_sigma - intercept_sigma */
            for (int32_t i = 0; i < n; i++) {
                double pred = intercept_sigma;
                for (int32_t j = 0; j < p; j++) {
                    pred += Xs[i * p + j] * beta_sigma[j];
                }
                r_work[i] = z_work[i] - pred;
            }

            /* Compute xw_sq for sigma */
            for (int32_t j = 0; j < p; j++) {
                double s = 0.0;
                for (int32_t i = 0; i < n; i++) {
                    double xij = Xs[i * p + j];
                    s += w_work[i] * xij * xij;
                }
                xw_sq[j] = s / n;
            }

            /* CD passes for sigma */
            for (int32_t cd_iter = 0; cd_iter < max_iter; cd_iter++) {
                double max_change = cd_pass(&w, lambda, 1);
                total_iter++;
                if (max_change < tol) {
                    double full_change = cd_pass(&w, lambda, 0);
                    total_iter++;
                    if (full_change < tol) break;
                }
            }

            intercept_sigma = clamp(w.intercept, -10.0, 10.0);

            /* Update eta_sigma and sigma_val (clamp to prevent overflow) */
            for (int32_t i = 0; i < n; i++) {
                double pred = intercept_sigma;
                for (int32_t j = 0; j < p; j++) {
                    pred += Xs[i * p + j] * beta_sigma[j];
                }
                eta_sigma[i] = clamp(pred, -10.0, 10.0);
                sigma_val[i] = exp(eta_sigma[i]);
            }

            /* ---- Step 3: Check convergence ---- */
            double dev;
            if (distribution == GAMLSS_NORMAL) {
                dev = gamlss_normal_deviance(y, n, mu_val, sigma_val);
            } else if (distribution == GAMLSS_GAMMA) {
                dev = gamlss_gamma_deviance(y, n, mu_val, sigma_val);
            } else {
                dev = gamlss_beta_deviance(y, n, mu_val, sigma_val);
            }

            double rel_change = fabs(dev - dev_prev) / (fabs(dev_prev) + 0.1);
            dev_prev = dev;
            if (rel_change < tol && rs > 0) break;
        }

        /* Store result */
        gam_fit_t *fit = &path->fits[n_fits];
        fit->beta = (double *)malloc((size_t)n_coefs_total * sizeof(double));
        if (!fit->beta) break;

        /* Unstandardize coefficients for mu */
        if (distribution == GAMLSS_NORMAL && params->standardize) {
            /* Normal with identity link: mu is on original scale */
            fit->beta[0] = intercept_mu;
            for (int32_t j = 0; j < p; j++) {
                fit->beta[j + 1] = beta_mu[j] / x_sd[j];
                fit->beta[0] -= fit->beta[j + 1] * x_mean[j];
            }
        } else if (params->standardize) {
            /* Gamma/Beta with log/logit link */
            fit->beta[0] = intercept_mu;
            for (int32_t j = 0; j < p; j++) {
                fit->beta[j + 1] = beta_mu[j] / x_sd[j];
                fit->beta[0] -= fit->beta[j + 1] * x_mean[j];
            }
        } else {
            fit->beta[0] = intercept_mu;
            for (int32_t j = 0; j < p; j++) fit->beta[j + 1] = beta_mu[j];
        }

        /* Unstandardize coefficients for sigma/phi (log link always) */
        int32_t base = p + 1;
        if (params->standardize) {
            fit->beta[base] = intercept_sigma;
            for (int32_t j = 0; j < p; j++) {
                fit->beta[base + j + 1] = beta_sigma[j] / x_sd[j];
                fit->beta[base] -= fit->beta[base + j + 1] * x_mean[j];
            }
        } else {
            fit->beta[base] = intercept_sigma;
            for (int32_t j = 0; j < p; j++) fit->beta[base + j + 1] = beta_sigma[j];
        }

        fit->n_coefs = n_coefs_total;
        fit->lambda = lambda;
        fit->n_iter = total_iter;
        fit->deviance = dev_prev;

        /* Null deviance (intercept-only model) */
        if (n_fits == 0) {
            double *mu0 = (double *)malloc((size_t)n * sizeof(double));
            double *s0 = (double *)malloc((size_t)n * sizeof(double));
            if (mu0 && s0) {
                if (distribution == GAMLSS_NORMAL) {
                    gamlss_normal_init(y, n, mu0, s0);
                    fit->null_deviance = gamlss_normal_deviance(y, n, mu0, s0);
                } else if (distribution == GAMLSS_GAMMA) {
                    gamlss_gamma_init(y, n, mu0, s0);
                    fit->null_deviance = gamlss_gamma_deviance(y, n, mu0, s0);
                } else {
                    gamlss_beta_init(y, n, mu0, s0);
                    fit->null_deviance = gamlss_beta_deviance(y, n, mu0, s0);
                }
            }
            free(mu0); free(s0);
        } else {
            fit->null_deviance = path->fits[0].null_deviance;
        }

        /* df: count nonzero coefficients (both mu and sigma) */
        fit->df = 0;
        for (int32_t j = 0; j < p; j++) {
            if (beta_mu[j] != 0.0) fit->df++;
            if (beta_sigma[j] != 0.0) fit->df++;
        }

        fit->cv_mean = NAN;
        fit->cv_se = NAN;

        n_fits++;
    }

    path->n_fits = n_fits;
    free(lambda_seq);

    /* Cleanup working arrays */
    free(Xs); free(x_mean); free(x_sd);
    free(mu_val); free(sigma_val); free(eta_mu); free(eta_sigma);
    free(z_work); free(w_work); free(r_work);
    free(beta_mu); free(beta_sigma);
    free(xw_sq); free(pf); free(active_mu); free(active_sigma); free(ever_active);
    free(w.lb); free(w.ub);
    return path;

cleanup_gamlss:
    free(Xs); free(x_mean); free(x_sd);
    free(mu_val); free(sigma_val); free(eta_mu); free(eta_sigma);
    free(z_work); free(w_work); free(r_work);
    free(beta_mu); free(beta_sigma);
    free(xw_sq); free(pf); free(active_mu); free(active_sigma); free(ever_active);
    free(w.lb); free(w.ub);
    return NULL;
}

/* --- GAMLSS prediction --- */

int gam_predict_gamlss(
    const gam_path_t *path, int32_t fit_idx,
    const double *X, int32_t nrow, int32_t ncol,
    double *out
) {
    if (!path || !X || !out) return -1;
    if (fit_idx < 0 || fit_idx >= path->n_fits) return -1;
    if (path->family_gamlss == 0) return -1;
    int32_t p = path->n_features;
    if (ncol != p) return -1;

    int32_t distribution = path->family_gamlss - 1;
    const double *beta = path->fits[fit_idx].beta;

    for (int32_t i = 0; i < nrow; i++) {
        /* mu prediction */
        double eta_mu = beta[0];
        for (int32_t j = 0; j < p; j++) {
            eta_mu += beta[j + 1] * X[i * ncol + j];
        }
        /* Apply mu inverse link */
        double mu;
        if (distribution == GAMLSS_NORMAL) {
            mu = eta_mu;
        } else if (distribution == GAMLSS_GAMMA) {
            mu = exp(eta_mu);
        } else {
            mu = 1.0 / (1.0 + exp(-eta_mu));
        }

        /* sigma/phi prediction (log link) */
        int32_t base = p + 1;
        double eta_sigma = beta[base];
        for (int32_t j = 0; j < p; j++) {
            eta_sigma += beta[base + j + 1] * X[i * ncol + j];
        }
        double sigma = exp(eta_sigma);

        out[i * 2 + 0] = mu;
        out[i * 2 + 1] = sigma;
    }
    return 0;
}

/* ========== Serialization ========== */

static const char GAM_MAGIC[4] = {'G', 'A', 'M', '1'};

int gam_save(const gam_path_t *path, char **out_buf, int32_t *out_len) {
    if (!path) { gam_set_error("gam_save: NULL path"); return -1; }

    /* Calculate buffer size */
    size_t size = 4;  /* magic */
    size += sizeof(int32_t) * 8;  /* n_fits, n_features, n_coefs, family, link, penalty, idx_min, idx_1se */
    size += sizeof(double) * 4;   /* alpha, y_mean, y_sd, reserved */
    /* smooth/tensor specs */
    size += sizeof(int32_t) * 2;  /* n_smooths, n_tensors */
    size += (size_t)path->n_smooths * sizeof(gam_smooth_t);
    size += (size_t)path->n_tensors * sizeof(gam_tensor_t);
    size += (size_t)path->n_features * sizeof(double) * 2;  /* x_mean, x_sd */

    for (int32_t k = 0; k < path->n_fits; k++) {
        size += sizeof(double) * 3;   /* lambda, deviance, null_deviance */
        size += sizeof(int32_t) * 2;  /* df, n_iter */
        size += sizeof(double) * 2;   /* cv_mean, cv_se */
        size += (size_t)path->fits[k].n_coefs * sizeof(double);  /* beta */
    }

    char *buf = (char *)malloc(size);
    if (!buf) { gam_set_error("gam_save: alloc failed"); return -1; }

    char *p = buf;

    /* Magic */
    memcpy(p, GAM_MAGIC, 4); p += 4;

    /* Header */
    #define WRITE_I32(v) do { int32_t _v = (v); memcpy(p, &_v, 4); p += 4; } while(0)
    #define WRITE_F64(v) do { double _v = (v); memcpy(p, &_v, 8); p += 8; } while(0)

    WRITE_I32(path->n_fits);
    WRITE_I32(path->n_features);
    WRITE_I32(path->n_coefs);
    WRITE_I32(path->family);
    WRITE_I32(path->link);
    WRITE_I32(path->penalty);
    WRITE_I32(path->idx_min);
    WRITE_I32(path->idx_1se);
    WRITE_F64(path->alpha);
    WRITE_F64(path->y_mean);
    WRITE_F64(path->y_sd);
    WRITE_I32(path->n_tasks);  /* n_tasks (0 or 1 = single-task) */
    WRITE_I32(path->family_gamlss);  /* GAMLSS distribution (0 = not GAMLSS) */

    /* Smooth/tensor term specs (for prediction-time basis expansion) */
    WRITE_I32(path->n_smooths);
    WRITE_I32(path->n_tensors);
    for (int32_t s = 0; s < path->n_smooths; s++) {
        WRITE_I32(path->smooths[s].feature);
        WRITE_I32(path->smooths[s].n_knots);
        WRITE_I32(path->smooths[s].degree);
        WRITE_F64(path->smooths[s].lambda_smooth);
    }
    for (int32_t t = 0; t < path->n_tensors; t++) {
        WRITE_I32(path->tensors[t].n_margins);
        for (int32_t m = 0; m < 4; m++) WRITE_I32(path->tensors[t].features[m]);
        for (int32_t m = 0; m < 4; m++) WRITE_I32(path->tensors[t].n_knots[m]);
        for (int32_t m = 0; m < 4; m++) WRITE_I32(path->tensors[t].degree[m]);
        for (int32_t m = 0; m < 4; m++) WRITE_F64(path->tensors[t].lambda_smooth[m]);
    }

    /* x_mean, x_sd (length n_features) */
    for (int32_t j = 0; j < path->n_features; j++) {
        WRITE_F64(path->x_mean ? path->x_mean[j] : 0.0);
    }
    for (int32_t j = 0; j < path->n_features; j++) {
        WRITE_F64(path->x_sd ? path->x_sd[j] : 1.0);
    }

    /* Fits */
    for (int32_t k = 0; k < path->n_fits; k++) {
        const gam_fit_t *fit = &path->fits[k];
        WRITE_F64(fit->lambda);
        WRITE_F64(fit->deviance);
        WRITE_F64(fit->null_deviance);
        WRITE_I32(fit->df);
        WRITE_I32(fit->n_iter);
        WRITE_F64(fit->cv_mean);
        WRITE_F64(fit->cv_se);
        for (int32_t j = 0; j < fit->n_coefs; j++) {
            WRITE_F64(fit->beta[j]);
        }
    }

    #undef WRITE_I32
    #undef WRITE_F64

    *out_buf = buf;
    *out_len = (int32_t)(p - buf);
    return 0;
}

gam_path_t *gam_load(const char *buf, int32_t len) {
    if (len < 4 || memcmp(buf, GAM_MAGIC, 4) != 0) {
        gam_set_error("gam_load: invalid magic");
        return NULL;
    }

    const char *p = buf + 4;
    const char *end = buf + len;

    #define READ_I32(dst) do { if (p + 4 > end) goto trunc; int32_t _v; memcpy(&_v, p, 4); p += 4; dst = _v; } while(0)
    #define READ_F64(dst) do { if (p + 8 > end) goto trunc; double _v; memcpy(&_v, p, 8); p += 8; dst = _v; } while(0)

    gam_path_t *path = (gam_path_t *)calloc(1, sizeof(gam_path_t));
    if (!path) { gam_set_error("gam_load: alloc failed"); return NULL; }

    int32_t n_fits;
    READ_I32(n_fits);
    READ_I32(path->n_features);
    READ_I32(path->n_coefs);
    READ_I32(path->family);
    READ_I32(path->link);
    READ_I32(path->penalty);
    READ_I32(path->idx_min);
    READ_I32(path->idx_1se);
    READ_F64(path->alpha);
    READ_F64(path->y_mean);
    READ_F64(path->y_sd);
    int32_t n_tasks_stored; READ_I32(n_tasks_stored);
    path->n_tasks = (n_tasks_stored > 1) ? n_tasks_stored : 0;
    int32_t family_gamlss_stored; READ_I32(family_gamlss_stored);
    path->family_gamlss = family_gamlss_stored;

    /* Read smooth/tensor term specs */
    READ_I32(path->n_smooths);
    READ_I32(path->n_tensors);
    if (path->n_smooths > 0) {
        path->smooths = (gam_smooth_t *)calloc((size_t)path->n_smooths, sizeof(gam_smooth_t));
        if (!path->smooths) { gam_free(path); gam_set_error("gam_load: alloc"); return NULL; }
        for (int32_t s = 0; s < path->n_smooths; s++) {
            READ_I32(path->smooths[s].feature);
            READ_I32(path->smooths[s].n_knots);
            READ_I32(path->smooths[s].degree);
            READ_F64(path->smooths[s].lambda_smooth);
        }
    }
    if (path->n_tensors > 0) {
        path->tensors = (gam_tensor_t *)calloc((size_t)path->n_tensors, sizeof(gam_tensor_t));
        if (!path->tensors) { gam_free(path); gam_set_error("gam_load: alloc"); return NULL; }
        for (int32_t t = 0; t < path->n_tensors; t++) {
            READ_I32(path->tensors[t].n_margins);
            for (int32_t m = 0; m < 4; m++) READ_I32(path->tensors[t].features[m]);
            for (int32_t m = 0; m < 4; m++) READ_I32(path->tensors[t].n_knots[m]);
            for (int32_t m = 0; m < 4; m++) READ_I32(path->tensors[t].degree[m]);
            for (int32_t m = 0; m < 4; m++) READ_F64(path->tensors[t].lambda_smooth[m]);
        }
    }

    int32_t p_feat = path->n_features;
    path->x_mean = (double *)malloc((size_t)p_feat * sizeof(double));
    path->x_sd = (double *)malloc((size_t)p_feat * sizeof(double));
    if (!path->x_mean || !path->x_sd) { gam_free(path); gam_set_error("gam_load: alloc"); return NULL; }

    for (int32_t j = 0; j < p_feat; j++) READ_F64(path->x_mean[j]);
    for (int32_t j = 0; j < p_feat; j++) READ_F64(path->x_sd[j]);

    path->fits = (gam_fit_t *)calloc((size_t)n_fits, sizeof(gam_fit_t));
    if (!path->fits) { gam_free(path); gam_set_error("gam_load: alloc"); return NULL; }
    path->n_fits = n_fits;

    for (int32_t k = 0; k < n_fits; k++) {
        gam_fit_t *fit = &path->fits[k];
        READ_F64(fit->lambda);
        READ_F64(fit->deviance);
        READ_F64(fit->null_deviance);
        READ_I32(fit->df);
        READ_I32(fit->n_iter);
        READ_F64(fit->cv_mean);
        READ_F64(fit->cv_se);

        fit->n_coefs = path->n_coefs;
        fit->beta = (double *)malloc((size_t)fit->n_coefs * sizeof(double));
        if (!fit->beta) { gam_free(path); gam_set_error("gam_load: alloc"); return NULL; }
        for (int32_t j = 0; j < fit->n_coefs; j++) READ_F64(fit->beta[j]);
    }

    #undef READ_I32
    #undef READ_F64
    return path;

trunc:
    gam_free(path);
    gam_set_error("gam_load: truncated data");
    return NULL;
}

/* ========== Free ========== */

void gam_free(gam_path_t *path) {
    if (!path) return;
    if (path->fits) {
        for (int32_t k = 0; k < path->n_fits; k++) {
            free(path->fits[k].beta);
        }
        free(path->fits);
    }
    if (path->relaxed_fits) {
        for (int32_t k = 0; k < path->n_fits; k++) {
            free(path->relaxed_fits[k].beta);
        }
        free(path->relaxed_fits);
    }
    free(path->x_mean);
    free(path->x_sd);
    free(path->basis_map);
    free(path->smooths);
    free(path->tensors);
    free(path);
}

void gam_free_buffer(void *ptr) {
    free(ptr);
}
