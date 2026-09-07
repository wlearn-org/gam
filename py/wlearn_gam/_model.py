import ctypes

import numpy as np
from wlearn.bundle import encode_bundle, decode_bundle, write_bundle_output
from wlearn.registry import register
from wlearn.errors import ValidationError, NotFittedError, DisposedError

from ._ffi import get_lib


TYPE_ID_CLASSIFIER = "wlearn.gam.classifier@1"
TYPE_ID_REGRESSOR = "wlearn.gam.regressor@1"
TYPE_ID_CLASSIFIER_V2 = "wlearn.gam.classifier@2"
TYPE_ID_REGRESSOR_V2 = "wlearn.gam.regressor@2"

FAMILY = {
    "gaussian": 0,
    "binomial": 1,
    "poisson": 2,
    "gamma": 3,
    "inverse_gaussian": 4,
    "negative_binomial": 5,
    "tweedie": 6,
    "multinomial": 7,
    "cox": 8,
    "huber": 10,
    "quantile": 11,
}

LINK = {
    "canonical": -1,
    "identity": 0,
    "log": 1,
    "logit": 2,
    "probit": 3,
    "cloglog": 4,
    "inverse": 5,
    "inverse_squared": 6,
    "sqrt": 7,
}

PENALTY = {
    "none": 0,
    "l1": 1,
    "lasso": 1,
    "l2": 2,
    "ridge": 2,
    "elasticnet": 3,
    "mcp": 4,
    "scad": 5,
    "group_l1": 6,
    "sparse_group": 7,
    "slope": 8,
    "fused": 9,
}

_DECLARED = False
_D = ctypes.c_double
_I = ctypes.c_int
_DP = ctypes.POINTER(ctypes.c_double)
_IP = ctypes.POINTER(ctypes.c_int)


def _declare(lib):
    global _DECLARED
    if _DECLARED:
        return

    lib.wl_gam_get_last_error.argtypes = []
    lib.wl_gam_get_last_error.restype = ctypes.c_char_p

    lib.wl_gam_fit.argtypes = [
        _DP, _I, _I, _DP,
        _I, _I, _I,
        _D, _I, _D,
        _D, _D,
        _D, _I, _I, _I,
        _I, _I, _I, _I,
        _D, _D,
        _D,
        _I,
        _D,
        _D,
    ]
    lib.wl_gam_fit.restype = ctypes.c_void_p

    lib.wl_gam_fit_groups.argtypes = [
        _DP, _I, _I, _DP,
        _I, _I, _I,
        _D, _I, _D,
        _IP, _I,
        _D, _I,
        _I, _I,
        _I,
    ]
    lib.wl_gam_fit_groups.restype = ctypes.c_void_p
    lib.wl_gam_fit_multinomial.argtypes = [
        _DP, _I, _I, _DP, _I, _I, _D, _I, _D, _D, _I, _I, _I, _I, _I,
    ]
    lib.wl_gam_fit_multinomial.restype = ctypes.c_void_p
    lib.wl_gam_get_n_tasks.argtypes = [ctypes.c_void_p]
    lib.wl_gam_get_n_tasks.restype = _I

    lib.wl_gam_predict.argtypes = [
        ctypes.c_void_p, _I, _DP, _I, _I, _DP,
    ]
    lib.wl_gam_predict.restype = _I
    lib.wl_gam_predict_multinomial.argtypes = lib.wl_gam_predict.argtypes
    lib.wl_gam_predict_multinomial.restype = _I
    lib.wl_gam_predict_relaxed.argtypes = lib.wl_gam_predict.argtypes
    lib.wl_gam_predict_relaxed.restype = _I
    lib.wl_gam_has_relaxed.argtypes = [ctypes.c_void_p]
    lib.wl_gam_has_relaxed.restype = _I
    lib.wl_gam_get_relaxed_coef.argtypes = [ctypes.c_void_p, _I, _I]
    lib.wl_gam_get_relaxed_coef.restype = _D
    lib.wl_gam_predict_eta.argtypes = lib.wl_gam_predict.argtypes
    lib.wl_gam_predict_eta.restype = _I
    lib.wl_gam_predict_proba.argtypes = lib.wl_gam_predict.argtypes
    lib.wl_gam_predict_proba.restype = _I

    lib.wl_gam_get_n_fits.argtypes = [ctypes.c_void_p]
    lib.wl_gam_get_n_fits.restype = _I
    lib.wl_gam_get_n_features.argtypes = [ctypes.c_void_p]
    lib.wl_gam_get_n_features.restype = _I
    lib.wl_gam_get_n_coefs.argtypes = [ctypes.c_void_p]
    lib.wl_gam_get_n_coefs.restype = _I
    lib.wl_gam_get_family.argtypes = [ctypes.c_void_p]
    lib.wl_gam_get_family.restype = _I
    lib.wl_gam_get_idx_min.argtypes = [ctypes.c_void_p]
    lib.wl_gam_get_idx_min.restype = _I
    lib.wl_gam_get_idx_1se.argtypes = [ctypes.c_void_p]
    lib.wl_gam_get_idx_1se.restype = _I
    lib.wl_gam_get_lambda.argtypes = [ctypes.c_void_p, _I]
    lib.wl_gam_get_lambda.restype = _D
    lib.wl_gam_get_deviance.argtypes = [ctypes.c_void_p, _I]
    lib.wl_gam_get_deviance.restype = _D
    lib.wl_gam_get_df.argtypes = [ctypes.c_void_p, _I]
    lib.wl_gam_get_df.restype = _I
    lib.wl_gam_get_cv_mean.argtypes = [ctypes.c_void_p, _I]
    lib.wl_gam_get_cv_mean.restype = _D
    lib.wl_gam_get_cv_se.argtypes = [ctypes.c_void_p, _I]
    lib.wl_gam_get_cv_se.restype = _D
    lib.wl_gam_get_coef.argtypes = [ctypes.c_void_p, _I, _I]
    lib.wl_gam_get_coef.restype = _D

    lib.wl_gam_save.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(_I),
    ]
    lib.wl_gam_save.restype = _I
    lib.wl_gam_load.argtypes = [ctypes.c_void_p, _I]
    lib.wl_gam_load.restype = ctypes.c_void_p
    lib.wl_gam_free.argtypes = [ctypes.c_void_p]
    lib.wl_gam_free.restype = None
    lib.wl_gam_free_buffer.argtypes = [ctypes.c_void_p]
    lib.wl_gam_free_buffer.restype = None

    _DECLARED = True


def _lib():
    lib = get_lib()
    _declare(lib)
    return lib


def _resolve_enum(mapping, value, fallback):
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return mapping.get(value.lower(), fallback)
    return fallback


def _as_matrix(X):
    arr = np.ascontiguousarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got {arr.ndim}")
    return arr


def _as_vector(y, name="y"):
    arr = np.ascontiguousarray(y, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got {arr.ndim}")
    return arr


def _last_error(lib):
    err = lib.wl_gam_get_last_error()
    return err.decode("utf-8", "replace") if err else "unknown error"


def _class_labels(values, expected=None):
    labels = np.asarray(values)
    if (labels.ndim != 1 or len(labels) < 2 or
            (expected is not None and len(labels) != expected) or
            labels.dtype.kind not in 'iuf' or not np.all(np.isfinite(labels)) or
            np.any(labels != np.floor(labels)) or np.any(labels < -(1 << 31)) or
            np.any(labels >= (1 << 31)) or len(np.unique(labels)) != len(labels)):
        raise ValidationError('GAM classes must be unique int32 labels matching the probability columns')
    return labels.astype(np.int32)


class GAMModel:
    """GLM/GAM estimator backed by the wlearn C11 core."""

    def __init__(self, params=None):
        self._params = dict(params or {})
        self._handle = None
        self._fitted = False
        self._disposed = False
        self._n_features = 0
        self._n_fits = 0
        self._classes = None
        self._family_inferred = False

    @classmethod
    def create(cls, params=None):
        return cls(params)

    def fit(self, X, y):
        if self._disposed:
            raise DisposedError("GAMModel has been disposed")

        lib = _lib()
        if self._params.get("relax") and self._params.get("groups") is not None:
            raise ValueError("relax is not supported by the group fitting ABI")
        self._free_handle()

        X = _as_matrix(X)
        y = _as_vector(y)
        nrow, ncol = X.shape
        if y.shape[0] != nrow:
            raise ValueError(f"y length ({y.shape[0]}) does not match X rows ({nrow})")

        if self._params.get('family') is None and self._params.get('task') is not None:
            task = self._params['task']
            if task not in ('classification', 'regression'):
                raise ValidationError('GAM task must be classification or regression')
            self._params['family'] = 'binomial' if task == 'classification' else 'gaussian'
            self._family_inferred = True
        family = _resolve_enum(FAMILY, self._params.get("family"), 0)
        self._classes = None
        if family in (1, 7):
            labels = _class_labels(np.unique(y))
            if family == 1 and self._family_inferred and len(labels) > 2:
                self._params['family'] = 'multinomial'
                family = 7
            if family == 1 and len(labels) != 2:
                raise ValidationError('Binomial GAM requires exactly two classes')
            self._classes = labels
            y = np.ascontiguousarray(np.searchsorted(labels, y), dtype=np.float64)
        link = _resolve_enum(LINK, self._params.get("link"), -1)
        penalty = _resolve_enum(PENALTY, self._params.get("penalty"), 3)
        groups = self._params.get("groups")

        if family == 7:
            if self._params.get('relax') or groups is not None:
                raise ValidationError('Multinomial GAM does not support relaxed or group fitting')
            handle = lib.wl_gam_fit_multinomial(
                X.ctypes.data_as(_DP), nrow, ncol, y.ctypes.data_as(_DP), len(self._classes),
                penalty, float(self._params.get('alpha', 1.0)),
                int(self._params.get('nLambda', self._params.get('n_lambda', 50))),
                float(self._params.get('lambdaMinRatio', self._params.get('lambda_min_ratio', 0.0))),
                float(self._params.get('tol', 1e-7)),
                int(self._params.get('maxIter', self._params.get('max_iter', 10000))),
                int(self._params.get('maxInner', self._params.get('max_inner', 25))),
                int(self._params.get('standardize', 1)),
                int(self._params.get('fitIntercept', self._params.get('fit_intercept', 1))),
                int(self._params.get('seed', 42)),
            )
        elif groups is not None and penalty in (6, 7):
            groups_arr = np.ascontiguousarray(groups, dtype=np.int32)
            if groups_arr.shape[0] != ncol:
                raise ValueError(
                    f"groups length ({groups_arr.shape[0]}) does not match X columns ({ncol})"
                )
            n_groups = int(self._params.get("nGroups", self._params.get("n_groups", int(groups_arr.max()) + 1)))
            handle = lib.wl_gam_fit_groups(
                X.ctypes.data_as(_DP), nrow, ncol,
                y.ctypes.data_as(_DP),
                family, link, penalty,
                float(self._params.get("alpha", 1.0)),
                int(self._params.get("nLambda", self._params.get("n_lambda", 100))),
                float(self._params.get("lambdaMinRatio", self._params.get("lambda_min_ratio", 0.0))),
                groups_arr.ctypes.data_as(_IP), n_groups,
                float(self._params.get("tol", 1e-7)),
                int(self._params.get("maxIter", self._params.get("max_iter", 10000))),
                int(self._params.get("standardize", 1)),
                int(self._params.get("fitIntercept", self._params.get("fit_intercept", 1))),
                int(self._params.get("seed", 42)),
            )
        else:
            handle = lib.wl_gam_fit(
                X.ctypes.data_as(_DP), nrow, ncol,
                y.ctypes.data_as(_DP),
                family, link, penalty,
                float(self._params.get("alpha", 1.0)),
                int(self._params.get("nLambda", self._params.get("n_lambda", 100))),
                float(self._params.get("lambdaMinRatio", self._params.get("lambda_min_ratio", 0.0))),
                float(self._params.get("gammaMcp", self._params.get("gamma_mcp", 3.0))),
                float(self._params.get("gammaScad", self._params.get("gamma_scad", 3.7))),
                float(self._params.get("tol", 1e-7)),
                int(self._params.get("maxIter", self._params.get("max_iter", 10000))),
                int(self._params.get("maxInner", self._params.get("max_inner", 25))),
                int(self._params.get("screening", 1)),
                int(self._params.get("nFolds", self._params.get("n_folds", 0))),
                int(self._params.get("standardize", 1)),
                int(self._params.get("fitIntercept", self._params.get("fit_intercept", 1))),
                int(self._params.get("relax", 0)),
                float(self._params.get("tweedieP", self._params.get("tweedie_power", 1.5))),
                float(self._params.get("nbTheta", self._params.get("neg_binom_theta", 0.0))),
                float(self._params.get("slopeQ", self._params.get("slope_q", 0.1))),
                int(self._params.get("seed", 42)),
                float(self._params.get("huberGamma", self._params.get("huber_gamma", 1.345))),
                float(self._params.get("quantileTau", self._params.get("quantile_tau", 0.5))),
            )

        if not handle:
            raise RuntimeError(f"GAM fit failed: {_last_error(lib)}")

        self._handle = handle
        self._artifact_media_type = 'application/vnd.wlearn.gam.raw'
        self._fitted = True
        self._n_features = ncol
        self._n_fits = lib.wl_gam_get_n_fits(handle)
        return self

    @property
    def classes(self):
        family = _resolve_enum(FAMILY, self._params.get('family'), 0)
        if not self.is_fitted or family not in (1, 7):
            return None
        if self._classes is not None:
            return self._classes.copy()
        count = 2 if family == 1 else _lib().wl_gam_get_n_tasks(self._handle)
        return np.arange(count, dtype=np.int32)

    def predict(self, X, fit_idx=None):
        classes = self.classes
        if classes is None:
            return self.predict_response(X, fit_idx)
        proba = self.predict_proba(X, fit_idx).reshape(-1, len(classes))
        return classes[proba.argmax(axis=1)]

    def predict_response(self, X, fit_idx=None):
        return self._predict_common(X, fit_idx, "wl_gam_predict", "predict")

    def predict_relaxed(self, X, fit_idx=None):
        return self._predict_common(X, fit_idx, "wl_gam_predict_relaxed", "predict_relaxed")

    @property
    def has_relaxed(self):
        return bool(self._handle and not self._disposed and _lib().wl_gam_has_relaxed(self._handle))

    def predict_eta(self, X, fit_idx=None):
        return self._predict_common(X, fit_idx, "wl_gam_predict_eta", "predict_eta")

    def predict_proba(self, X, fit_idx=None):
        self._ensure_fitted()
        family = _resolve_enum(FAMILY, self._params.get("family"), 0)
        if family == 7:
            return self._predict_common(X, fit_idx, 'wl_gam_predict_multinomial',
                                        'predict_proba', len(self.classes))
        if family != 1:
            raise ValidationError('predict_proba is only available for binomial/multinomial models')
        positive = self._predict_common(X, fit_idx, "wl_gam_predict_proba", "predict_proba")
        return np.column_stack((1 - positive, positive)).reshape(-1)

    def score(self, X, y, fit_idx=None):
        y = _as_vector(y)
        family = _resolve_enum(FAMILY, self._params.get("family"), 0)
        preds = self.predict(X, fit_idx)
        if family in (1, 7):
            return float(np.mean(preds == y))
        ss_res = float(np.sum((y - preds) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def get_coefs(self, fit_idx=None):
        return self._get_coefs(fit_idx, False)

    def get_relaxed_coefs(self, fit_idx=None):
        return self._get_coefs(fit_idx, True)

    def _get_coefs(self, fit_idx, relaxed):
        self._ensure_fitted()
        lib = _lib()
        idx = self._resolve_fit_idx(fit_idx)
        if relaxed and not self.has_relaxed:
            raise RuntimeError("Model has no relaxed fit")
        n_coefs = lib.wl_gam_get_n_coefs(self._handle)
        get_coef = lib.wl_gam_get_relaxed_coef if relaxed else lib.wl_gam_get_coef
        return np.array(
            [get_coef(self._handle, idx, i) for i in range(n_coefs)],
            dtype=np.float64,
        )

    def get_lambda(self, fit_idx=None):
        self._ensure_fitted()
        return float(_lib().wl_gam_get_lambda(self._handle, self._resolve_fit_idx(fit_idx)))

    def get_deviance(self, fit_idx=None):
        self._ensure_fitted()
        return float(_lib().wl_gam_get_deviance(self._handle, self._resolve_fit_idx(fit_idx)))

    def get_df(self, fit_idx=None):
        self._ensure_fitted()
        return int(_lib().wl_gam_get_df(self._handle, self._resolve_fit_idx(fit_idx)))

    def get_cv_mean(self, fit_idx=None):
        self._ensure_fitted()
        return float(_lib().wl_gam_get_cv_mean(self._handle, self._resolve_fit_idx(fit_idx)))

    def get_cv_se(self, fit_idx=None):
        self._ensure_fitted()
        return float(_lib().wl_gam_get_cv_se(self._handle, self._resolve_fit_idx(fit_idx)))

    def save(self, path=None):
        raw = self._save_raw()
        family = _resolve_enum(FAMILY, self._params.get("family"), 0)
        if self.has_relaxed:
            type_id = TYPE_ID_CLASSIFIER_V2 if family in (1, 7) else TYPE_ID_REGRESSOR_V2
        else:
            type_id = TYPE_ID_CLASSIFIER if family in (1, 7) else TYPE_ID_REGRESSOR
        metadata = {'nFeatures': self._n_features, 'nFits': self._n_fits}
        classes = self.classes
        if classes is not None and not np.array_equal(classes, np.arange(len(classes))):
            metadata['classes'] = classes.tolist()
        bundle = encode_bundle(
            {
                "typeId": type_id,
                "params": self.get_params(),
                "metadata": metadata,
            },
            [{
                "id": "model",
                "mediaType": getattr(self, '_artifact_media_type', 'application/vnd.wlearn.gam.raw'),
                "data": raw,
            }],
        )
        return write_bundle_output(bundle, path)

    @classmethod
    def load(cls, data):
        manifest, toc, blobs = decode_bundle(data)
        return cls._from_bundle(manifest, toc, blobs)

    @classmethod
    def _from_bundle(cls, manifest, toc, blobs):
        type_id = manifest.get("typeId")
        if type_id not in (TYPE_ID_CLASSIFIER, TYPE_ID_REGRESSOR, TYPE_ID_CLASSIFIER_V2, TYPE_ID_REGRESSOR_V2):
            raise ValueError(f"Unsupported GAM bundle typeId: {type_id}")
        entry = next((item for item in toc if item["id"] == "model"), None)
        if entry is None:
            raise ValueError('Bundle missing "model" artifact')
        raw = bytes(blobs[entry["offset"]:entry["offset"] + entry["length"]])
        if raw[:4] != f"GAM{type_id[-1]}".encode('ascii'):
            raise ValueError("GAM bundle typeId and model format disagree")
        params = dict(manifest.get("params") or {})
        if type_id in (TYPE_ID_CLASSIFIER, TYPE_ID_CLASSIFIER_V2):
            params.setdefault("family", "binomial")
        model = cls._load_raw(raw, params)
        try:
            labels = manifest.get('metadata', {}).get('classes')
            if labels is not None:
                expected = len(model.classes) if model.classes is not None else 0
                model._classes = _class_labels(labels, expected)
            model._artifact_media_type = entry['mediaType']
            return model
        except Exception:
            model.dispose()
            raise

    def _save_raw(self):
        self._ensure_fitted()
        lib = _lib()
        out_buf = ctypes.c_void_p()
        out_len = _I()
        ret = lib.wl_gam_save(self._handle, ctypes.byref(out_buf), ctypes.byref(out_len))
        if ret != 0:
            raise RuntimeError(f"GAM save failed: {_last_error(lib)}")
        try:
            return bytes((ctypes.c_ubyte * out_len.value).from_address(out_buf.value))
        finally:
            lib.wl_gam_free_buffer(out_buf)

    @classmethod
    def _load_raw(cls, data, params=None):
        lib = _lib()
        buf = ctypes.create_string_buffer(bytes(data))
        handle = lib.wl_gam_load(buf, len(data))
        if not handle:
            raise RuntimeError(f"GAM load failed: {_last_error(lib)}")
        obj = cls(params)
        obj._handle = handle
        obj._fitted = True
        obj._n_features = lib.wl_gam_get_n_features(handle)
        obj._n_fits = lib.wl_gam_get_n_fits(handle)
        return obj

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        self._free_handle()

    def get_params(self):
        return dict(self._params)

    def set_params(self, params=None, **kwargs):
        updates = {**(params or {}), **kwargs}
        if 'family' in updates:
            self._family_inferred = False
        elif 'task' in updates and self._family_inferred:
            self._params.pop('family', None)
            self._family_inferred = False
        self._params.update(updates)
        return self

    @property
    def is_fitted(self):
        return self._fitted and not self._disposed

    @property
    def n_features(self):
        return self._n_features

    @property
    def n_fits(self):
        return self._n_fits

    @property
    def idx_min(self):
        if not self._handle or self._disposed:
            return -1
        return int(_lib().wl_gam_get_idx_min(self._handle))

    @property
    def idx_1se(self):
        if not self._handle or self._disposed:
            return -1
        return int(_lib().wl_gam_get_idx_1se(self._handle))

    @property
    def capabilities(self):
        family = _resolve_enum(FAMILY, self._params.get("family"), 0)
        is_classifier = family in (1, 7)
        return {
            "classifier": is_classifier,
            "regressor": not is_classifier,
            "predictProba": is_classifier,
            "sampleWeight": False,
            "csr": False,
            "featureImportances": False,
        }

    @staticmethod
    def default_search_space(task=None):
        return {
            **({} if task else {"family": {"type": "categorical", "values": ["gaussian", "binomial", "poisson", "gamma"]}}),
            "penalty": {"type": "categorical", "values": ["elasticnet", "lasso", "ridge", "mcp", "scad", "slope"]},
            "alpha": {"type": "uniform", "low": 0.0, "high": 1.0},
            "nLambda": {"type": "categorical", "values": [50, 100]},
            "nFolds": {"type": "categorical", "values": [0, 5]},
        }

    defaultSearchSpace = default_search_space

    def _predict_common(self, X, fit_idx, fn_name, label, columns=1):
        self._ensure_fitted()
        lib = _lib()
        X = _as_matrix(X)
        nrow, ncol = X.shape
        out = np.zeros(nrow * columns, dtype=np.float64)
        fn = getattr(lib, fn_name)
        ret = fn(
            self._handle,
            self._resolve_fit_idx(fit_idx),
            X.ctypes.data_as(_DP),
            nrow,
            ncol,
            out.ctypes.data_as(_DP),
        )
        if ret != 0:
            raise RuntimeError(f"GAM {label} failed: {_last_error(lib)}")
        return out

    def _resolve_fit_idx(self, fit_idx):
        if fit_idx is not None:
            return int(fit_idx)
        idx = self.idx_min
        return idx if idx >= 0 else self._n_fits - 1

    def _ensure_fitted(self):
        if self._disposed:
            raise DisposedError("GAMModel has been disposed")
        if not self._fitted or not self._handle:
            raise NotFittedError("GAMModel is not fitted. Call fit() first")

    def _free_handle(self):
        if self._handle:
            _lib().wl_gam_free(self._handle)
            self._handle = None
        self._fitted = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.dispose()
        return False

    def __del__(self):
        if getattr(self, "_handle", None):
            try:
                self._free_handle()
            except Exception:
                pass


register(TYPE_ID_CLASSIFIER, GAMModel._from_bundle)
register(TYPE_ID_REGRESSOR, GAMModel._from_bundle)
register(TYPE_ID_CLASSIFIER_V2, GAMModel._from_bundle)
register(TYPE_ID_REGRESSOR_V2, GAMModel._from_bundle)
