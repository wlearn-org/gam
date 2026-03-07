import { getWasm, loadGAM } from './wasm.js'
import {
  normalizeX, normalizeY,
  encodeBundle, decodeBundle,
  register,
  DisposedError, NotFittedError
} from '@wlearn/core'

// Constants matching gam.h
const FAMILY = {
  gaussian: 0, binomial: 1, poisson: 2, gamma: 3,
  inverse_gaussian: 4, negative_binomial: 5, tweedie: 6,
  multinomial: 7, cox: 8,
  huber: 10, quantile: 11
}

const LINK = {
  canonical: -1, identity: 0, log: 1, logit: 2, probit: 3,
  cloglog: 4, inverse: 5, inverse_squared: 6, sqrt: 7
}

const PENALTY = {
  none: 0, l1: 1, lasso: 1, l2: 2, ridge: 2,
  elasticnet: 3, mcp: 4, scad: 5,
  group_l1: 6, sparse_group: 7, slope: 8,
  fused: 9
}

const GAMLSS_DIST = {
  normal: 0, gamma: 1, beta: 2
}

function resolveEnum(map, value, fallback) {
  if (typeof value === 'number') return value
  if (typeof value === 'string') {
    const v = map[value.toLowerCase()]
    if (v !== undefined) return v
  }
  return fallback
}

function getLastError() {
  const wasm = getWasm()
  return wasm.ccall('wl_gam_get_last_error', 'string', [], [])
}

// FinalizationRegistry safety net
const leakRegistry = typeof FinalizationRegistry !== 'undefined'
  ? new FinalizationRegistry(({ ptr, freeFn }) => {
    if (ptr[0]) {
      console.warn('@wlearn/gam: Model was not disposed -- calling free() automatically.')
      freeFn(ptr[0])
    }
  })
  : null

const LOAD_SENTINEL = Symbol('load')

export class GAMModel {
  #handle = null
  #freed = false
  #ptrRef = null
  #params = {}
  #fitted = false
  #nFeatures = 0
  #nFits = 0

  constructor(handle, params, extra) {
    if (handle === LOAD_SENTINEL) {
      this.#handle = params
      this.#params = extra.params || {}
      this.#nFeatures = extra.nFeatures || 0
      this.#nFits = extra.nFits || 0
      this.#fitted = true
    } else {
      this.#handle = null
      this.#params = handle || {}
    }

    this.#freed = false
    if (this.#handle) this.#registerLeak()
  }

  static async create(params = {}) {
    await loadGAM()
    return new GAMModel(params)
  }

  // --- Estimator interface ---

  fit(X, y) {
    this.#ensureFitted(false)
    const wasm = getWasm()

    if (this.#handle) {
      wasm._wl_gam_free(this.#handle)
      this.#handle = null
      if (this.#ptrRef) this.#ptrRef[0] = null
      if (leakRegistry) leakRegistry.unregister(this)
    }

    const { data: xData, rows, cols } = this.#normalizeX(X)
    const yNorm = normalizeY(y)
    const yData = yNorm instanceof Float64Array ? yNorm : new Float64Array(yNorm)

    if (yData.length !== rows) {
      throw new Error(`y length (${yData.length}) does not match X rows (${rows})`)
    }

    const family = resolveEnum(FAMILY, this.#params.family, 0)
    const link = resolveEnum(LINK, this.#params.link, -1)
    const penalty = resolveEnum(PENALTY, this.#params.penalty, 3)

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)
    const yPtr = wasm._malloc(yData.length * 8)
    wasm.HEAPF64.set(yData, yPtr / 8)

    let modelPtr
    const groups = this.#params.groups
    const isGroupPenalty = (penalty === 6 || penalty === 7) && groups

    if (isGroupPenalty) {
      const nGroups = this.#params.nGroups ?? (Math.max(...groups) + 1)
      const groupArr = new Int32Array(groups)
      const groupPtr = wasm._malloc(groupArr.length * 4)
      wasm.HEAP32.set(groupArr, groupPtr / 4)

      modelPtr = wasm._wl_gam_fit_groups(
        xPtr, rows, cols,
        yPtr,
        family, link, penalty,
        this.#params.alpha ?? 1.0,
        this.#params.nLambda ?? 100,
        this.#params.lambdaMinRatio ?? 0.0,
        groupPtr, nGroups,
        this.#params.tol ?? 1e-7,
        this.#params.maxIter ?? 10000,
        this.#params.standardize ?? 1,
        this.#params.fitIntercept ?? 1,
        this.#params.seed ?? 42
      )

      wasm._free(groupPtr)
    } else {
      modelPtr = wasm._wl_gam_fit(
        xPtr, rows, cols,
        yPtr,
        family, link, penalty,
        this.#params.alpha ?? 1.0,
        this.#params.nLambda ?? 100,
        this.#params.lambdaMinRatio ?? 0.0,
        this.#params.gammaMcp ?? 3.0,
        this.#params.gammaScad ?? 3.7,
        this.#params.tol ?? 1e-7,
        this.#params.maxIter ?? 10000,
        this.#params.maxInner ?? 25,
        this.#params.screening ?? 1,
        this.#params.nFolds ?? 0,
        this.#params.standardize ?? 1,
        this.#params.fitIntercept ?? 1,
        this.#params.relax ?? 0,
        this.#params.tweedieP ?? 1.5,
        this.#params.nbTheta ?? 0.0,
        this.#params.slopeQ ?? 0.1,
        this.#params.seed ?? 42,
        this.#params.huberGamma ?? 1.345,
        this.#params.quantileTau ?? 0.5
      )
    }

    wasm._free(xPtr)
    wasm._free(yPtr)

    if (!modelPtr) {
      throw new Error(`Training failed: ${getLastError()}`)
    }

    this.#handle = modelPtr
    this.#fitted = true
    this.#nFeatures = cols
    this.#nFits = wasm._wl_gam_get_n_fits(modelPtr)

    this.#registerLeak()
    return this
  }

  fitCox(X, time, status) {
    this.#ensureFitted(false)
    const wasm = getWasm()

    if (this.#handle) {
      wasm._wl_gam_free(this.#handle)
      this.#handle = null
      if (this.#ptrRef) this.#ptrRef[0] = null
      if (leakRegistry) leakRegistry.unregister(this)
    }

    const { data: xData, rows, cols } = this.#normalizeX(X)
    const timeData = new Float64Array(normalizeY(time))
    const statusData = new Float64Array(normalizeY(status))

    if (timeData.length !== rows || statusData.length !== rows) {
      throw new Error(`time/status length mismatch with X rows (${rows})`)
    }

    const penalty = resolveEnum(PENALTY, this.#params.penalty, 3)

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)
    const timePtr = wasm._malloc(timeData.length * 8)
    wasm.HEAPF64.set(timeData, timePtr / 8)
    const statusPtr = wasm._malloc(statusData.length * 8)
    wasm.HEAPF64.set(statusData, statusPtr / 8)

    const modelPtr = wasm._wl_gam_fit_cox(
      xPtr, rows, cols,
      timePtr, statusPtr,
      penalty,
      this.#params.alpha ?? 0.5,
      this.#params.nLambda ?? 100,
      this.#params.lambdaMinRatio ?? 0.0,
      this.#params.gammaMcp ?? 3.0,
      this.#params.gammaScad ?? 3.7,
      this.#params.tol ?? 1e-7,
      this.#params.maxIter ?? 5000,
      this.#params.standardize ?? 1,
      this.#params.seed ?? 42
    )

    wasm._free(xPtr)
    wasm._free(timePtr)
    wasm._free(statusPtr)

    if (!modelPtr) {
      throw new Error(`Cox training failed: ${getLastError()}`)
    }

    this.#handle = modelPtr
    this.#fitted = true
    this.#nFeatures = cols
    this.#nFits = wasm._wl_gam_get_n_fits(modelPtr)

    this.#registerLeak()
    return this
  }

  fitMulti(X, Y, nTasks) {
    this.#ensureFitted(false)
    const wasm = getWasm()

    if (this.#handle) {
      wasm._wl_gam_free(this.#handle)
      this.#handle = null
      if (this.#ptrRef) this.#ptrRef[0] = null
      if (leakRegistry) leakRegistry.unregister(this)
    }

    const { data: xData, rows, cols } = this.#normalizeX(X)
    const yData = Y instanceof Float64Array ? Y : new Float64Array(Y)

    if (yData.length !== rows * nTasks) {
      throw new Error(`Y length (${yData.length}) does not match rows*nTasks (${rows}*${nTasks})`)
    }

    const penalty = resolveEnum(PENALTY, this.#params.penalty, 3)

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)
    const yPtr = wasm._malloc(yData.length * 8)
    wasm.HEAPF64.set(yData, yPtr / 8)

    const modelPtr = wasm._wl_gam_fit_multi(
      xPtr, rows, cols,
      yPtr, nTasks,
      penalty,
      this.#params.alpha ?? 1.0,
      this.#params.nLambda ?? 100,
      this.#params.lambdaMinRatio ?? 0.0,
      this.#params.tol ?? 1e-7,
      this.#params.maxIter ?? 10000,
      this.#params.standardize ?? 1,
      this.#params.fitIntercept ?? 1,
      this.#params.seed ?? 42
    )

    wasm._free(xPtr)
    wasm._free(yPtr)

    if (!modelPtr) {
      throw new Error(`Multi-task training failed: ${getLastError()}`)
    }

    this.#handle = modelPtr
    this.#fitted = true
    this.#nFeatures = cols
    this.#nFits = wasm._wl_gam_get_n_fits(modelPtr)

    this.#registerLeak()
    return this
  }

  fitMultinomial(X, y, nClasses) {
    this.#ensureFitted(false)
    const wasm = getWasm()

    if (this.#handle) {
      wasm._wl_gam_free(this.#handle)
      this.#handle = null
      if (this.#ptrRef) this.#ptrRef[0] = null
      if (leakRegistry) leakRegistry.unregister(this)
    }

    const { data: xData, rows, cols } = this.#normalizeX(X)
    const yNorm = normalizeY(y)
    const yData = yNorm instanceof Float64Array ? yNorm : new Float64Array(yNorm)

    if (yData.length !== rows) {
      throw new Error(`y length (${yData.length}) does not match X rows (${rows})`)
    }

    const penalty = resolveEnum(PENALTY, this.#params.penalty, 3)

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)
    const yPtr = wasm._malloc(yData.length * 8)
    wasm.HEAPF64.set(yData, yPtr / 8)

    const modelPtr = wasm._wl_gam_fit_multinomial(
      xPtr, rows, cols,
      yPtr, nClasses,
      penalty,
      this.#params.alpha ?? 1.0,
      this.#params.nLambda ?? 50,
      this.#params.lambdaMinRatio ?? 0.0,
      this.#params.tol ?? 1e-7,
      this.#params.maxIter ?? 10000,
      this.#params.maxInner ?? 25,
      this.#params.standardize ?? 1,
      this.#params.fitIntercept ?? 1,
      this.#params.seed ?? 42
    )

    wasm._free(xPtr)
    wasm._free(yPtr)

    if (!modelPtr) {
      throw new Error(`Multinomial training failed: ${getLastError()}`)
    }

    this.#handle = modelPtr
    this.#fitted = true
    this.#nFeatures = cols
    this.#nFits = wasm._wl_gam_get_n_fits(modelPtr)
    this.#params.family = 'multinomial'

    this.#registerLeak()
    return this
  }

  fitGamlss(X, y, distribution = 'normal') {
    this.#ensureFitted(false)
    const wasm = getWasm()

    if (this.#handle) {
      wasm._wl_gam_free(this.#handle)
      this.#handle = null
      if (this.#ptrRef) this.#ptrRef[0] = null
      if (leakRegistry) leakRegistry.unregister(this)
    }

    const { data: xData, rows, cols } = this.#normalizeX(X)
    const yNorm = normalizeY(y)
    const yData = yNorm instanceof Float64Array ? yNorm : new Float64Array(yNorm)

    if (yData.length !== rows) {
      throw new Error(`y length (${yData.length}) does not match X rows (${rows})`)
    }

    const dist = resolveEnum(GAMLSS_DIST, distribution, 0)
    const penalty = resolveEnum(PENALTY, this.#params.penalty, 3)

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)
    const yPtr = wasm._malloc(yData.length * 8)
    wasm.HEAPF64.set(yData, yPtr / 8)

    const modelPtr = wasm._wl_gam_fit_gamlss(
      xPtr, rows, cols,
      yPtr, dist,
      penalty,
      this.#params.alpha ?? 1.0,
      this.#params.nLambda ?? 50,
      this.#params.lambdaMinRatio ?? 0.0,
      this.#params.tol ?? 1e-7,
      this.#params.maxIter ?? 10000,
      this.#params.standardize ?? 1,
      this.#params.fitIntercept ?? 1,
      this.#params.seed ?? 42
    )

    wasm._free(xPtr)
    wasm._free(yPtr)

    if (!modelPtr) {
      throw new Error(`GAMLSS training failed: ${getLastError()}`)
    }

    this.#handle = modelPtr
    this.#fitted = true
    this.#nFeatures = cols
    this.#nFits = wasm._wl_gam_get_n_fits(modelPtr)
    this.#params.family = 'gamlss'

    this.#registerLeak()
    return this
  }

  predictGamlss(X, fitIdx) {
    this.#ensureFitted()
    const idx = this.#resolveFitIdx(fitIdx)
    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)
    const outPtr = wasm._malloc(rows * 2 * 8)

    const ret = wasm._wl_gam_predict_gamlss(this.#handle, idx, xPtr, rows, cols, outPtr)

    wasm._free(xPtr)

    if (ret !== 0) {
      wasm._free(outPtr)
      throw new Error(`predictGamlss failed: ${getLastError()}`)
    }

    const result = new Float64Array(rows * 2)
    result.set(wasm.HEAPF64.subarray(outPtr / 8, outPtr / 8 + rows * 2))
    wasm._free(outPtr)
    return result
  }

  get familyGamlss() {
    if (!this.#handle || this.#freed) return 0
    return getWasm()._wl_gam_get_family_gamlss(this.#handle)
  }

  predictMultinomial(X, fitIdx) {
    this.#ensureFitted()
    const idx = this.#resolveFitIdx(fitIdx)
    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)
    const nClasses = wasm._wl_gam_get_n_tasks(this.#handle)

    if (nClasses < 2) {
      throw new Error('predictMultinomial requires a multinomial model')
    }

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)
    const outPtr = wasm._malloc(rows * nClasses * 8)

    const ret = wasm._wl_gam_predict_multinomial(this.#handle, idx, xPtr, rows, cols, outPtr)

    wasm._free(xPtr)

    if (ret !== 0) {
      wasm._free(outPtr)
      throw new Error(`predictMultinomial failed: ${getLastError()}`)
    }

    const result = new Float64Array(rows * nClasses)
    result.set(wasm.HEAPF64.subarray(outPtr / 8, outPtr / 8 + rows * nClasses))
    wasm._free(outPtr)
    return result
  }

  predictMulti(X, fitIdx) {
    this.#ensureFitted()
    const idx = this.#resolveFitIdx(fitIdx)
    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)
    const nTasks = wasm._wl_gam_get_n_tasks(this.#handle)

    if (nTasks < 2) {
      throw new Error('predictMulti requires a multi-task model')
    }

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)
    const outPtr = wasm._malloc(rows * nTasks * 8)

    const ret = wasm._wl_gam_predict_multi(this.#handle, idx, xPtr, rows, cols, outPtr)

    wasm._free(xPtr)

    if (ret !== 0) {
      wasm._free(outPtr)
      throw new Error(`predictMulti failed: ${getLastError()}`)
    }

    const result = new Float64Array(rows * nTasks)
    result.set(wasm.HEAPF64.subarray(outPtr / 8, outPtr / 8 + rows * nTasks))
    wasm._free(outPtr)
    return result
  }

  get nTasks() {
    if (!this.#handle || this.#freed) return 0
    return getWasm()._wl_gam_get_n_tasks(this.#handle)
  }

  predict(X, fitIdx) {
    this.#ensureFitted()
    const idx = this.#resolveFitIdx(fitIdx)
    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)
    const outPtr = wasm._malloc(rows * 8)

    const ret = wasm._wl_gam_predict(this.#handle, idx, xPtr, rows, cols, outPtr)

    wasm._free(xPtr)

    if (ret !== 0) {
      wasm._free(outPtr)
      throw new Error(`Predict failed: ${getLastError()}`)
    }

    const result = new Float64Array(rows)
    result.set(wasm.HEAPF64.subarray(outPtr / 8, outPtr / 8 + rows))
    wasm._free(outPtr)
    return result
  }

  predictProba(X, fitIdx) {
    this.#ensureFitted()
    const family = resolveEnum(FAMILY, this.#params.family, 0)
    if (family !== 1 && family !== 7) {
      throw new Error('predictProba is only available for binomial/multinomial')
    }

    // For multinomial, delegate to predictMultinomial
    if (family === 7) {
      return this.predictMultinomial(X, fitIdx)
    }

    const idx = this.#resolveFitIdx(fitIdx)
    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)
    const outPtr = wasm._malloc(rows * 8)

    const ret = wasm._wl_gam_predict_proba(this.#handle, idx, xPtr, rows, cols, outPtr)

    wasm._free(xPtr)

    if (ret !== 0) {
      wasm._free(outPtr)
      throw new Error(`predictProba failed: ${getLastError()}`)
    }

    const result = new Float64Array(rows)
    result.set(wasm.HEAPF64.subarray(outPtr / 8, outPtr / 8 + rows))
    wasm._free(outPtr)
    return result
  }

  predictEta(X, fitIdx) {
    this.#ensureFitted()
    const idx = this.#resolveFitIdx(fitIdx)
    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)
    const outPtr = wasm._malloc(rows * 8)

    const ret = wasm._wl_gam_predict_eta(this.#handle, idx, xPtr, rows, cols, outPtr)

    wasm._free(xPtr)

    if (ret !== 0) {
      wasm._free(outPtr)
      throw new Error(`predictEta failed: ${getLastError()}`)
    }

    const result = new Float64Array(rows)
    result.set(wasm.HEAPF64.subarray(outPtr / 8, outPtr / 8 + rows))
    wasm._free(outPtr)
    return result
  }

  score(X, y, fitIdx) {
    const yArr = normalizeY(y)
    const family = resolveEnum(FAMILY, this.#params.family, 0)

    if (family === 7) {
      // Accuracy (multinomial) -- argmax of probabilities
      const probs = this.predictMultinomial(X, fitIdx)
      const nClasses = this.nTasks
      let correct = 0
      for (let i = 0; i < yArr.length; i++) {
        let bestClass = 0, bestProb = probs[i * nClasses]
        for (let k = 1; k < nClasses; k++) {
          if (probs[i * nClasses + k] > bestProb) {
            bestProb = probs[i * nClasses + k]
            bestClass = k
          }
        }
        if (bestClass === yArr[i]) correct++
      }
      return correct / yArr.length
    }

    const preds = this.predict(X, fitIdx)

    if (family === 1) {
      // Accuracy (binomial)
      let correct = 0
      for (let i = 0; i < preds.length; i++) {
        if ((preds[i] >= 0.5 ? 1 : 0) === yArr[i]) correct++
      }
      return correct / preds.length
    } else {
      // R-squared
      let ssRes = 0, ssTot = 0, yMean = 0
      for (let i = 0; i < yArr.length; i++) yMean += yArr[i]
      yMean /= yArr.length
      for (let i = 0; i < yArr.length; i++) {
        ssRes += (yArr[i] - preds[i]) ** 2
        ssTot += (yArr[i] - yMean) ** 2
      }
      return ssTot === 0 ? 0 : 1 - ssRes / ssTot
    }
  }

  // --- Path inspection ---

  get nFits() { return this.#nFits }
  get nFeatures() { return this.#nFeatures }

  get idxMin() {
    if (!this.#handle || this.#freed) return -1
    return getWasm()._wl_gam_get_idx_min(this.#handle)
  }

  get idx1se() {
    if (!this.#handle || this.#freed) return -1
    return getWasm()._wl_gam_get_idx_1se(this.#handle)
  }

  getLambda(fitIdx) {
    this.#ensureFitted()
    return getWasm()._wl_gam_get_lambda(this.#handle, fitIdx)
  }

  getDeviance(fitIdx) {
    this.#ensureFitted()
    return getWasm()._wl_gam_get_deviance(this.#handle, fitIdx)
  }

  getDf(fitIdx) {
    this.#ensureFitted()
    return getWasm()._wl_gam_get_df(this.#handle, fitIdx)
  }

  getCvMean(fitIdx) {
    this.#ensureFitted()
    return getWasm()._wl_gam_get_cv_mean(this.#handle, fitIdx)
  }

  getCvSe(fitIdx) {
    this.#ensureFitted()
    return getWasm()._wl_gam_get_cv_se(this.#handle, fitIdx)
  }

  getCoefs(fitIdx) {
    this.#ensureFitted()
    const idx = fitIdx ?? (this.#nFits - 1)
    const wasm = getWasm()
    const nCoefs = this.#nFeatures + 1
    const result = new Float64Array(nCoefs)
    for (let j = 0; j < nCoefs; j++) {
      result[j] = wasm._wl_gam_get_coef(this.#handle, idx, j)
    }
    return result
  }

  // --- Model I/O ---

  save() {
    this.#ensureFitted()
    const rawBytes = this.#saveRaw()
    const family = resolveEnum(FAMILY, this.#params.family, 0)
    const typeId = (family === 1 || family === 7)
      ? 'wlearn.gam.classifier@1'
      : 'wlearn.gam.regressor@1'

    const metadata = {
      nFeatures: this.#nFeatures,
      nFits: this.#nFits
    }

    return encodeBundle(
      { typeId, params: this.getParams(), metadata },
      [{ id: 'model', data: rawBytes }]
    )
  }

  static async load(bytes) {
    const { manifest, toc, blobs } = decodeBundle(bytes)
    return GAMModel._fromBundle(manifest, toc, blobs)
  }

  static async _fromBundle(manifest, toc, blobs) {
    await loadGAM()
    const wasm = getWasm()

    const entry = toc.find(e => e.id === 'model')
    if (!entry) throw new Error('Bundle missing "model" artifact')
    const raw = blobs.subarray(entry.offset, entry.offset + entry.length)

    const bufPtr = wasm._malloc(raw.length)
    wasm.HEAPU8.set(raw, bufPtr)
    const modelPtr = wasm._wl_gam_load(bufPtr, raw.length)
    wasm._free(bufPtr)

    if (!modelPtr) {
      throw new Error(`load failed: ${getLastError()}`)
    }

    const params = manifest.params || {}
    const metadata = manifest.metadata || {}
    const nFeatures = metadata.nFeatures || wasm._wl_gam_get_n_features(modelPtr)
    const nFits = metadata.nFits || wasm._wl_gam_get_n_fits(modelPtr)

    return new GAMModel(LOAD_SENTINEL, modelPtr, {
      params, nFeatures, nFits
    })
  }

  dispose() {
    if (this.#freed) return
    this.#freed = true

    if (this.#handle) {
      const wasm = getWasm()
      wasm._wl_gam_free(this.#handle)
    }

    if (this.#ptrRef) this.#ptrRef[0] = null
    if (leakRegistry) leakRegistry.unregister(this)

    this.#handle = null
    this.#fitted = false
  }

  // --- Params ---

  getParams() {
    return { ...this.#params }
  }

  setParams(p) {
    Object.assign(this.#params, p)
    return this
  }

  get isFitted() {
    return this.#fitted && !this.#freed
  }

  get capabilities() {
    const family = resolveEnum(FAMILY, this.#params.family, 0)
    const isClassifier = family === 1 || family === 7
    return {
      classifier: isClassifier,
      regressor: !isClassifier,
      predictProba: isClassifier,
      decisionFunction: false,
      sampleWeight: false,
      csr: false,
      earlyStopping: false,
      featureImportances: false
    }
  }

  static defaultSearchSpace() {
    return {
      family: { type: 'categorical', values: ['gaussian', 'binomial', 'poisson', 'gamma'] },
      penalty: { type: 'categorical', values: ['elasticnet', 'lasso', 'ridge', 'mcp', 'scad', 'slope'] },
      alpha: { type: 'uniform', low: 0.0, high: 1.0 },
      nLambda: { type: 'categorical', values: [50, 100] },
      nFolds: { type: 'categorical', values: [0, 5] }
    }
  }

  // --- Private helpers ---

  #normalizeX(X) {
    return normalizeX(X, 'auto')
  }

  #ensureFitted(requireFit = true) {
    if (this.#freed) throw new DisposedError('GAMModel has been disposed.')
    if (requireFit && !this.#fitted) throw new NotFittedError('GAMModel is not fitted. Call fit() first.')
  }

  #resolveFitIdx(fitIdx) {
    if (fitIdx !== undefined) return fitIdx
    const wasm = getWasm()
    const idxMin = wasm._wl_gam_get_idx_min(this.#handle)
    return idxMin >= 0 ? idxMin : this.#nFits - 1
  }

  #registerLeak() {
    this.#ptrRef = [this.#handle]
    if (leakRegistry) {
      leakRegistry.register(this, {
        ptr: this.#ptrRef,
        freeFn: (h) => getWasm()._wl_gam_free(h)
      }, this)
    }
  }

  #saveRaw() {
    const wasm = getWasm()

    const outBufPtr = wasm._malloc(4)
    const outLenPtr = wasm._malloc(4)

    const ret = wasm._wl_gam_save(this.#handle, outBufPtr, outLenPtr)

    if (ret !== 0) {
      wasm._free(outBufPtr)
      wasm._free(outLenPtr)
      throw new Error(`save failed: ${getLastError()}`)
    }

    const bufPtr = wasm.getValue(outBufPtr, 'i32')
    const bufLen = wasm.getValue(outLenPtr, 'i32')

    const result = new Uint8Array(bufLen)
    result.set(wasm.HEAPU8.subarray(bufPtr, bufPtr + bufLen))

    wasm._wl_gam_free_buffer(bufPtr)
    wasm._free(outBufPtr)
    wasm._free(outLenPtr)

    return result
  }
}

// --- Register loaders with @wlearn/core ---

register('wlearn.gam.classifier@1', (m, t, b) => GAMModel._fromBundle(m, t, b))
register('wlearn.gam.regressor@1', (m, t, b) => GAMModel._fromBundle(m, t, b))
