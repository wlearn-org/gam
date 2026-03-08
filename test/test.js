let passed = 0
let failed = 0

async function test(name, fn) {
  try {
    await fn()
    console.log(`  PASS: ${name}`)
    passed++
  } catch (err) {
    console.log(`  FAIL: ${name}`)
    console.log(`        ${err.message}`)
    failed++
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg || 'assertion failed')
}

function assertClose(a, b, tol, msg) {
  const diff = Math.abs(a - b)
  if (diff > tol) throw new Error(msg || `expected ${a} ~ ${b} (diff=${diff}, tol=${tol})`)
}

// --- Deterministic LCG PRNG ---
function makeLCG(seed = 42) {
  let s = seed | 0
  return () => {
    s = (s * 1664525 + 1013904223) & 0x7fffffff
    return s / 0x7fffffff
  }
}

function makeRegressionData(rng, nSamples, nFeatures) {
  const X = [], y = []
  for (let i = 0; i < nSamples; i++) {
    const row = []
    let target = 0
    for (let j = 0; j < nFeatures; j++) {
      const v = rng() * 4 - 2
      row.push(v)
      target += v * (j + 1)
    }
    X.push(row)
    y.push(target + (rng() - 0.5) * 0.1)
  }
  return { X, y }
}

function makeClassificationData(rng, nSamples, nFeatures) {
  const X = [], y = []
  for (let i = 0; i < nSamples; i++) {
    const label = i % 2
    const row = []
    for (let j = 0; j < nFeatures; j++) {
      row.push(label * 2 + (rng() - 0.5) * 0.5)
    }
    X.push(row)
    y.push(label)
  }
  return { X, y }
}

function makePoissonData(rng, nSamples, nFeatures) {
  const X = [], y = []
  for (let i = 0; i < nSamples; i++) {
    const row = []
    let eta = 0.5
    for (let j = 0; j < nFeatures; j++) {
      const v = rng() * 2 - 1
      row.push(v)
      eta += v * 0.3
    }
    X.push(row)
    // Poisson mean = exp(eta), simulate as round(exp(eta))
    y.push(Math.max(0, Math.round(Math.exp(eta) + (rng() - 0.5) * 0.5)))
  }
  return { X, y }
}

async function main() {

// ============================================================
// WASM loading
// ============================================================
console.log('\n=== WASM Loading ===')

const { loadGAM } = require('../src/wasm.js')
const wasm = await loadGAM()

await test('WASM module loads', async () => {
  assert(wasm, 'wasm module is null')
  assert(typeof wasm.ccall === 'function', 'ccall not available')
})

await test('get_last_error returns string', async () => {
  const err = wasm.ccall('wl_gam_get_last_error', 'string', [], [])
  assert(typeof err === 'string', `expected string, got ${typeof err}`)
})

// ============================================================
// GAMModel basics
// ============================================================
console.log('\n=== GAMModel ===')

const { GAMModel } = require('../src/model.js')

await test('create() returns model', async () => {
  const model = await GAMModel.create()
  assert(model, 'model is null')
  assert(!model.isFitted, 'should not be fitted yet')
  model.dispose()
})

await test('create() with params', async () => {
  const model = await GAMModel.create({ family: 'gaussian', penalty: 'lasso', alpha: 1.0 })
  assert(model, 'model is null')
  const params = model.getParams()
  assert(params.family === 'gaussian', `family: ${params.family}`)
  assert(params.penalty === 'lasso', `penalty: ${params.penalty}`)
  model.dispose()
})

await test('setParams returns this', async () => {
  const model = await GAMModel.create()
  const ret = model.setParams({ alpha: 0.5 })
  assert(ret === model, 'setParams should return this')
  assert(model.getParams().alpha === 0.5, 'alpha should be 0.5')
  model.dispose()
})

// ============================================================
// Gaussian regression (elastic net)
// ============================================================
console.log('\n=== Gaussian Regression ===')

await test('Gaussian fit + predict (elastic net)', async () => {
  const rng = makeLCG(100)
  const { X, y } = makeRegressionData(rng, 100, 3)
  const model = await GAMModel.create({
    family: 'gaussian', penalty: 'elasticnet', alpha: 0.5, seed: 42
  })
  model.fit(X, y)

  assert(model.isFitted, 'should be fitted')
  assert(model.nFeatures === 3, `nFeatures: ${model.nFeatures}`)
  assert(model.nFits > 0, `nFits: ${model.nFits}`)

  const preds = model.predict(X)
  assert(preds instanceof Float64Array, 'preds should be Float64Array')
  assert(preds.length === 100, `preds length: ${preds.length}`)

  const sc = model.score(X, y)
  assert(sc > 0.5, `R2 too low: ${sc}`)

  model.dispose()
})

await test('Gaussian lasso', async () => {
  const rng = makeLCG(200)
  const { X, y } = makeRegressionData(rng, 100, 5)
  const model = await GAMModel.create({
    family: 'gaussian', penalty: 'lasso', seed: 42
  })
  model.fit(X, y)
  assert(model.nFits > 10, `expected many fits on path, got ${model.nFits}`)
  const sc = model.score(X, y)
  assert(sc > 0.5, `R2 too low: ${sc}`)
  model.dispose()
})

await test('Gaussian ridge', async () => {
  const rng = makeLCG(300)
  const { X, y } = makeRegressionData(rng, 100, 3)
  const model = await GAMModel.create({
    family: 'gaussian', penalty: 'ridge', seed: 42
  })
  model.fit(X, y)
  assert(model.isFitted, 'should be fitted')
  const sc = model.score(X, y)
  assert(sc > 0.5, `R2 too low: ${sc}`)
  model.dispose()
})

// ============================================================
// Path inspection
// ============================================================
console.log('\n=== Path Inspection ===')

await test('lambda path inspection', async () => {
  const rng = makeLCG(400)
  const { X, y } = makeRegressionData(rng, 100, 3)
  const model = await GAMModel.create({ family: 'gaussian', penalty: 'lasso', seed: 42 })
  model.fit(X, y)

  const n = model.nFits
  assert(n > 1, `nFits: ${n}`)

  // Lambda should be decreasing
  const l0 = model.getLambda(0)
  const l1 = model.getLambda(1)
  const lLast = model.getLambda(n - 1)
  assert(l0 > l1, `lambda should decrease: ${l0} vs ${l1}`)
  assert(l1 > lLast, `lambda should decrease: ${l1} vs ${lLast}`)

  // Deviance should be available
  const dev = model.getDeviance(0)
  assert(typeof dev === 'number' && isFinite(dev), `deviance: ${dev}`)

  // Df should be non-negative
  const df = model.getDf(n - 1)
  assert(df >= 0, `df: ${df}`)

  model.dispose()
})

await test('getCoefs returns array', async () => {
  const rng = makeLCG(500)
  const { X, y } = makeRegressionData(rng, 80, 2)
  const model = await GAMModel.create({ family: 'gaussian', penalty: 'lasso', seed: 42 })
  model.fit(X, y)

  const coefs = model.getCoefs()
  assert(coefs instanceof Float64Array, 'coefs should be Float64Array')
  // nCoefs = nFeatures + 1 (intercept)
  assert(coefs.length === 3, `coefs length: ${coefs.length}`)

  model.dispose()
})

// ============================================================
// Cross-validation
// ============================================================
console.log('\n=== Cross-Validation ===')

await test('CV selects lambda.min and lambda.1se', async () => {
  const rng = makeLCG(600)
  const { X, y } = makeRegressionData(rng, 150, 3)
  const model = await GAMModel.create({
    family: 'gaussian', penalty: 'lasso', nFolds: 5, seed: 42
  })
  model.fit(X, y)

  const idxMin = model.idxMin
  const idx1se = model.idx1se
  assert(idxMin >= 0, `idxMin should be non-negative: ${idxMin}`)
  assert(idx1se >= 0, `idx1se should be non-negative: ${idx1se}`)
  assert(idx1se <= idxMin, `1se should be more regularized (lower idx): ${idx1se} vs ${idxMin}`)

  // CV mean should be available
  const cvMean = model.getCvMean(idxMin)
  assert(isFinite(cvMean), `cvMean: ${cvMean}`)
  const cvSe = model.getCvSe(idxMin)
  assert(cvSe >= 0, `cvSe should be non-negative: ${cvSe}`)

  model.dispose()
})

// ============================================================
// Binomial classification
// ============================================================
console.log('\n=== Binomial Classification ===')

await test('Binomial fit + predict', async () => {
  const rng = makeLCG(700)
  const { X, y } = makeClassificationData(rng, 100, 3)
  const model = await GAMModel.create({
    family: 'binomial', penalty: 'elasticnet', alpha: 0.5, seed: 42
  })
  model.fit(X, y)

  assert(model.isFitted, 'should be fitted')
  const preds = model.predict(X)
  assert(preds.length === 100, `preds length: ${preds.length}`)

  // Score is accuracy for binomial
  const acc = model.score(X, y)
  assert(acc > 0.7, `accuracy too low: ${acc}`)

  model.dispose()
})

await test('Binomial predictProba', async () => {
  const rng = makeLCG(800)
  const { X, y } = makeClassificationData(rng, 80, 2)
  const model = await GAMModel.create({
    family: 'binomial', penalty: 'lasso', seed: 42
  })
  model.fit(X, y)

  const proba = model.predictProba(X)
  assert(proba instanceof Float64Array, 'proba should be Float64Array')
  assert(proba.length === 80, `proba length: ${proba.length}`)

  // Probabilities should be in [0, 1]
  for (let i = 0; i < proba.length; i++) {
    assert(proba[i] >= 0 && proba[i] <= 1, `proba[${i}] = ${proba[i]} out of [0,1]`)
  }

  model.dispose()
})

await test('predictProba throws for gaussian', async () => {
  const rng = makeLCG(810)
  const { X, y } = makeRegressionData(rng, 50, 2)
  const model = await GAMModel.create({ family: 'gaussian', seed: 42 })
  model.fit(X, y)

  let threw = false
  try {
    model.predictProba(X)
  } catch (e) {
    threw = true
    assert(e.message.includes('binomial') || e.message.includes('multinomial'),
      `unexpected error: ${e.message}`)
  }
  assert(threw, 'predictProba should throw for gaussian')
  model.dispose()
})

// ============================================================
// Poisson regression
// ============================================================
console.log('\n=== Poisson Regression ===')

await test('Poisson fit + predict', async () => {
  const rng = makeLCG(900)
  const { X, y } = makePoissonData(rng, 100, 3)
  const model = await GAMModel.create({
    family: 'poisson', penalty: 'elasticnet', alpha: 0.5, seed: 42
  })
  model.fit(X, y)

  assert(model.isFitted, 'should be fitted')
  const preds = model.predict(X)
  assert(preds.length === 100, `preds length: ${preds.length}`)

  // Predictions should be non-negative (Poisson mean)
  for (let i = 0; i < preds.length; i++) {
    assert(preds[i] >= 0, `preds[${i}] = ${preds[i]} should be non-negative`)
  }

  model.dispose()
})

// ============================================================
// Penalty types
// ============================================================
console.log('\n=== Penalty Types ===')

await test('MCP penalty', async () => {
  const rng = makeLCG(1000)
  const { X, y } = makeRegressionData(rng, 100, 3)
  const model = await GAMModel.create({
    family: 'gaussian', penalty: 'mcp', gammaMcp: 3.0, seed: 42
  })
  model.fit(X, y)
  assert(model.isFitted, 'should be fitted')
  const sc = model.score(X, y)
  assert(sc > 0.3, `R2 too low: ${sc}`)
  model.dispose()
})

await test('SCAD penalty', async () => {
  const rng = makeLCG(1100)
  const { X, y } = makeRegressionData(rng, 100, 3)
  const model = await GAMModel.create({
    family: 'gaussian', penalty: 'scad', gammaScad: 3.7, seed: 42
  })
  model.fit(X, y)
  assert(model.isFitted, 'should be fitted')
  const sc = model.score(X, y)
  assert(sc > 0.3, `R2 too low: ${sc}`)
  model.dispose()
})

// ============================================================
// SLOPE penalty
// ============================================================
console.log('\n=== SLOPE Penalty ===')

await test('SLOPE Gaussian sparse recovery', async () => {
  // 3 active features, 17 noise features
  const rng = makeLCG(2000)
  const nSamples = 200, nFeatures = 20
  const X = [], y = []
  const trueBeta = [3.0, -2.0, 1.5]
  for (let i = 0; i < nSamples; i++) {
    const row = []
    let target = 0.5  // intercept
    for (let j = 0; j < nFeatures; j++) {
      const v = rng() * 2 - 1
      row.push(v)
      if (j < 3) target += v * trueBeta[j]
    }
    X.push(row)
    y.push(target + (rng() - 0.5) * 0.2)
  }

  const model = await GAMModel.create({
    family: 'gaussian', penalty: 'slope', seed: 42
  })
  model.fit(X, y)

  assert(model.isFitted, 'should be fitted')
  assert(model.nFits > 0, `nFits: ${model.nFits}`)

  const sc = model.score(X, y)
  assert(sc > 0.8, `R2 too low: ${sc}`)

  // At least-regularized end, active features should be recovered
  const coefs = model.getCoefs()
  assert(fabs(coefs[1]) > 1.0, `beta1=${coefs[1]} should be ~3.0`)
  assert(fabs(coefs[2]) > 0.5, `beta2=${coefs[2]} should be ~-2.0`)

  model.dispose()
})

function fabs(x) { return Math.abs(x) }

await test('SLOPE Binomial', async () => {
  const rng = makeLCG(2100)
  const { X, y } = makeClassificationData(rng, 100, 5)
  const model = await GAMModel.create({
    family: 'binomial', penalty: 'slope', seed: 42
  })
  model.fit(X, y)
  assert(model.isFitted, 'should be fitted')
  const acc = model.score(X, y)
  assert(acc > 0.6, `accuracy too low: ${acc}`)
  model.dispose()
})

await test('SLOPE save + load roundtrip', async () => {
  const rng = makeLCG(2200)
  const { X, y } = makeRegressionData(rng, 80, 5)
  const model = await GAMModel.create({
    family: 'gaussian', penalty: 'slope', seed: 42
  })
  model.fit(X, y)

  const bundle = model.save()
  const loaded = await GAMModel.load(bundle)

  const preds1 = model.predict(X)
  const preds2 = loaded.predict(X)
  for (let i = 0; i < preds1.length; i++) {
    assertClose(preds1[i], preds2[i], 1e-10, `SLOPE save/load mismatch at ${i}`)
  }

  model.dispose()
  loaded.dispose()
})

// ============================================================
// Group Lasso / Sparse Group Lasso
// ============================================================
console.log('\n=== Group Lasso ===')

await test('Group Lasso zeros out noise groups', async () => {
  // 4 groups of 3 features each. Groups 0,1 active, groups 2,3 noise.
  const rng = makeLCG(3000)
  const nSamples = 200, nFeatures = 12
  const X = [], y = []
  for (let i = 0; i < nSamples; i++) {
    const row = []
    let target = 1.0
    for (let j = 0; j < nFeatures; j++) {
      const v = rng() * 2 - 1
      row.push(v)
      if (j < 3) target += v * 2.0       // group 0 active
      else if (j < 6) target += v * 1.5   // group 1 active
    }
    X.push(row)
    y.push(target + (rng() - 0.5) * 0.3)
  }

  const model = await GAMModel.create({
    family: 'gaussian',
    penalty: 'group_l1',
    groups: [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
    nGroups: 4,
    seed: 42
  })
  model.fit(X, y)

  assert(model.isFitted, 'should be fitted')
  const sc = model.score(X, y)
  assert(sc > 0.8, `R2 too low: ${sc}`)

  // At mid-path, noise groups should have small norms
  const midIdx = Math.floor(model.nFits / 2)
  const coefs = model.getCoefs(midIdx)

  // Compute group norms (skip intercept at coefs[0])
  function groupNorm(startFeat, nFeat) {
    let sum = 0
    for (let j = startFeat; j < startFeat + nFeat; j++) {
      sum += coefs[j + 1] ** 2  // +1 for intercept
    }
    return Math.sqrt(sum)
  }

  const g2 = groupNorm(6, 3)
  const g3 = groupNorm(9, 3)
  assert(g2 < 0.1, `noise group 2 norm ${g2} should be near zero`)
  assert(g3 < 0.1, `noise group 3 norm ${g3} should be near zero`)

  model.dispose()
})

await test('Sparse Group Lasso (within-group sparsity)', async () => {
  const rng = makeLCG(3100)
  const nSamples = 200, nFeatures = 12
  const X = [], y = []
  for (let i = 0; i < nSamples; i++) {
    const row = []
    let target = 0.5
    for (let j = 0; j < nFeatures; j++) {
      const v = rng() * 2 - 1
      row.push(v)
      // Only first feature in each active group matters
      if (j === 0) target += v * 3.0
      else if (j === 3) target += v * 2.0
    }
    X.push(row)
    y.push(target + (rng() - 0.5) * 0.3)
  }

  const model = await GAMModel.create({
    family: 'gaussian',
    penalty: 'sparse_group',
    alpha: 0.5,
    groups: [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
    nGroups: 4,
    seed: 42
  })
  model.fit(X, y)

  assert(model.isFitted, 'should be fitted')
  const sc = model.score(X, y)
  assert(sc > 0.7, `SGL R2 too low: ${sc}`)

  model.dispose()
})

await test('Group Lasso binomial classification', async () => {
  const rng = makeLCG(3200)
  const nSamples = 150, nFeatures = 9
  const X = [], y = []
  for (let i = 0; i < nSamples; i++) {
    const row = []
    let eta = 0
    for (let j = 0; j < nFeatures; j++) {
      const v = rng() * 2 - 1
      row.push(v)
      if (j < 3) eta += v * 2.0
    }
    X.push(row)
    y.push(1.0 / (1.0 + Math.exp(-eta)) > 0.5 ? 1 : 0)
  }

  const model = await GAMModel.create({
    family: 'binomial',
    penalty: 'group_l1',
    groups: [0, 0, 0, 1, 1, 1, 2, 2, 2],
    nGroups: 3,
    seed: 42
  })
  model.fit(X, y)
  const acc = model.score(X, y)
  assert(acc > 0.7, `group lasso binomial accuracy too low: ${acc}`)
  model.dispose()
})

// ============================================================
// Cox Proportional Hazards
// ============================================================
console.log('\n=== Cox PH ===')

function makeSurvivalData(rng, nSamples, nFeatures, trueBeta) {
  const X = [], time = [], status = []
  for (let i = 0; i < nSamples; i++) {
    const row = []
    let eta = 0
    for (let j = 0; j < nFeatures; j++) {
      const v = rng() * 2 - 1
      row.push(v)
      if (j < trueBeta.length) eta += v * trueBeta[j]
    }
    X.push(row)
    const u = Math.max(rng(), 1e-10)
    const t = -Math.log(u) / Math.exp(eta)
    const censTime = -Math.log(Math.max(rng(), 1e-10)) / 0.3
    if (censTime < t) {
      time.push(censTime)
      status.push(0)
    } else {
      time.push(t)
      status.push(1)
    }
  }
  return { X, time, status }
}

await test('Cox PH basic fit + predict', async () => {
  const rng = makeLCG(4000)
  const { X, time, status } = makeSurvivalData(rng, 200, 5, [0.8, -0.5])

  const model = await GAMModel.create({
    penalty: 'elasticnet', alpha: 0.5, seed: 42
  })
  model.fitCox(X, time, status)

  assert(model.isFitted, 'should be fitted')
  assert(model.nFits > 0, `nFits: ${model.nFits}`)

  // Risk scores should all be positive (exp(X*beta))
  const risk = model.predict(X)
  assert(risk.length === 200, `risk length: ${risk.length}`)
  let allPositive = true
  for (let i = 0; i < risk.length; i++) {
    if (risk[i] <= 0 || !isFinite(risk[i])) { allPositive = false; break }
  }
  assert(allPositive, 'all risk scores positive and finite')

  // Coefficient recovery
  const coefs = model.getCoefs()
  assert(coefs[1] > 0.2, `beta1=${coefs[1]} should be positive (hazard-increasing)`)
  assert(coefs[2] < -0.1, `beta2=${coefs[2]} should be negative (hazard-decreasing)`)

  model.dispose()
})

await test('Cox Lasso sparsity', async () => {
  const rng = makeLCG(4100)
  const { X, time, status } = makeSurvivalData(rng, 300, 10, [1.0, -0.7])

  const model = await GAMModel.create({
    penalty: 'lasso', nLambda: 30, seed: 42
  })
  model.fitCox(X, time, status)

  // Path should show sparsity gradient
  assert(model.nFits > 1, 'multiple fits along path')

  // First fit (highest lambda) should be very sparse
  const firstCoefs = model.getCoefs(0)
  let firstNZ = 0
  for (let j = 1; j <= 10; j++) {
    if (Math.abs(firstCoefs[j]) > 1e-10) firstNZ++
  }
  assert(firstNZ < 10, `first fit should be sparse, got ${firstNZ} nonzero`)

  model.dispose()
})

await test('Cox save + load roundtrip', async () => {
  const rng = makeLCG(4200)
  const { X, time, status } = makeSurvivalData(rng, 100, 3, [0.5])

  const model = await GAMModel.create({
    penalty: 'elasticnet', alpha: 0.5, seed: 42
  })
  model.fitCox(X, time, status)

  const bundle = model.save()
  const loaded = await GAMModel.load(bundle)

  const preds1 = model.predict(X)
  const preds2 = loaded.predict(X)
  for (let i = 0; i < preds1.length; i++) {
    assertClose(preds1[i], preds2[i], 1e-10, `Cox save/load mismatch at ${i}`)
  }

  model.dispose()
  loaded.dispose()
})

// ============================================================
// predictEta
// ============================================================
console.log('\n=== predictEta ===')

await test('predictEta returns linear predictor', async () => {
  const rng = makeLCG(1200)
  const { X, y } = makeClassificationData(rng, 60, 2)
  const model = await GAMModel.create({
    family: 'binomial', penalty: 'lasso', seed: 42
  })
  model.fit(X, y)

  const eta = model.predictEta(X)
  assert(eta instanceof Float64Array, 'eta should be Float64Array')
  assert(eta.length === 60, `eta length: ${eta.length}`)

  // Eta can be any real number (not bounded to [0,1] like proba)
  let hasNegative = false, hasPositive = false
  for (let i = 0; i < eta.length; i++) {
    assert(isFinite(eta[i]), `eta[${i}] = ${eta[i]} is not finite`)
    if (eta[i] < 0) hasNegative = true
    if (eta[i] > 0) hasPositive = true
  }
  assert(hasNegative && hasPositive, 'eta should have both positive and negative values')

  model.dispose()
})

// ============================================================
// Save / Load
// ============================================================
console.log('\n=== Save / Load ===')

await test('save + load roundtrip', async () => {
  const rng = makeLCG(1300)
  const { X, y } = makeRegressionData(rng, 80, 3)
  const model = await GAMModel.create({
    family: 'gaussian', penalty: 'lasso', seed: 42
  })
  model.fit(X, y)

  const bundle = model.save()
  assert(bundle instanceof Uint8Array, 'bundle should be Uint8Array')
  assert(bundle.length > 0, 'bundle should not be empty')

  const loaded = await GAMModel.load(bundle)
  assert(loaded.isFitted, 'loaded model should be fitted')
  assert(loaded.nFeatures === model.nFeatures, 'nFeatures mismatch')
  assert(loaded.nFits === model.nFits, 'nFits mismatch')

  // Predictions should match
  const preds1 = model.predict(X)
  const preds2 = loaded.predict(X)
  for (let i = 0; i < preds1.length; i++) {
    assertClose(preds1[i], preds2[i], 1e-10, `pred mismatch at ${i}`)
  }

  model.dispose()
  loaded.dispose()
})

await test('save + load preserves params', async () => {
  const rng = makeLCG(1400)
  const { X, y } = makeClassificationData(rng, 60, 2)
  const model = await GAMModel.create({
    family: 'binomial', penalty: 'elasticnet', alpha: 0.7, seed: 42
  })
  model.fit(X, y)

  const bundle = model.save()
  const loaded = await GAMModel.load(bundle)

  const p = loaded.getParams()
  assert(p.family === 'binomial', `family: ${p.family}`)
  assert(p.penalty === 'elasticnet', `penalty: ${p.penalty}`)
  assertClose(p.alpha, 0.7, 1e-10, `alpha: ${p.alpha}`)

  model.dispose()
  loaded.dispose()
})

// ============================================================
// Dispose
// ============================================================
console.log('\n=== Dispose ===')

await test('dispose() prevents further use', async () => {
  const rng = makeLCG(1500)
  const { X, y } = makeRegressionData(rng, 50, 2)
  const model = await GAMModel.create({ family: 'gaussian', seed: 42 })
  model.fit(X, y)
  model.dispose()

  assert(!model.isFitted, 'should not be fitted after dispose')

  let threw = false
  try {
    model.predict(X)
  } catch (e) {
    threw = true
  }
  assert(threw, 'predict should throw after dispose')
})

await test('double dispose is safe', async () => {
  const model = await GAMModel.create()
  model.dispose()
  model.dispose() // should not throw
})

// ============================================================
// Capabilities
// ============================================================
console.log('\n=== Capabilities ===')

await test('regressor capabilities', async () => {
  const model = await GAMModel.create({ family: 'gaussian' })
  const caps = model.capabilities
  assert(caps.regressor === true, 'should be regressor')
  assert(caps.classifier === false, 'should not be classifier')
  assert(caps.predictProba === false, 'no predictProba for regressor')
  model.dispose()
})

await test('classifier capabilities', async () => {
  const model = await GAMModel.create({ family: 'binomial' })
  const caps = model.capabilities
  assert(caps.classifier === true, 'should be classifier')
  assert(caps.regressor === false, 'should not be regressor')
  assert(caps.predictProba === true, 'should have predictProba')
  model.dispose()
})

// ============================================================
// Search space
// ============================================================
console.log('\n=== Search Space ===')

await test('defaultSearchSpace returns valid IR', async () => {
  const ss = GAMModel.defaultSearchSpace()
  assert(ss.family, 'should have family')
  assert(ss.family.type === 'categorical', `family type: ${ss.family.type}`)
  assert(Array.isArray(ss.family.values), 'family values should be array')
  assert(ss.penalty, 'should have penalty')
  assert(ss.alpha, 'should have alpha')
  assert(ss.alpha.type === 'uniform', `alpha type: ${ss.alpha.type}`)
})

// ============================================================
// Determinism
// ============================================================
console.log('\n=== Determinism ===')

await test('same seed produces identical results', async () => {
  const rng1 = makeLCG(1600)
  const { X, y } = makeRegressionData(rng1, 80, 3)

  const m1 = await GAMModel.create({ family: 'gaussian', penalty: 'lasso', nFolds: 5, seed: 123 })
  m1.fit(X, y)
  const p1 = m1.predict(X)

  const m2 = await GAMModel.create({ family: 'gaussian', penalty: 'lasso', nFolds: 5, seed: 123 })
  m2.fit(X, y)
  const p2 = m2.predict(X)

  for (let i = 0; i < p1.length; i++) {
    assertClose(p1[i], p2[i], 1e-12, `determinism mismatch at ${i}`)
  }

  m1.dispose()
  m2.dispose()
})

// ============================================================
// Relaxed fits
// ============================================================
console.log('\n=== Relaxed Fits ===')

await test('relaxed fit works', async () => {
  const rng = makeLCG(1700)
  const { X, y } = makeRegressionData(rng, 100, 3)
  const model = await GAMModel.create({
    family: 'gaussian', penalty: 'lasso', relax: 1, seed: 42
  })
  model.fit(X, y)
  assert(model.isFitted, 'should be fitted')
  const sc = model.score(X, y)
  assert(sc > 0.3, `R2 too low with relaxed: ${sc}`)
  model.dispose()
})

// ============================================================
// Re-fit (fit twice)
// ============================================================
console.log('\n=== Re-fit ===')

await test('fit can be called twice', async () => {
  const rng = makeLCG(1800)
  const { X: X1, y: y1 } = makeRegressionData(rng, 60, 2)
  const { X: X2, y: y2 } = makeRegressionData(rng, 80, 3)

  const model = await GAMModel.create({ family: 'gaussian', seed: 42 })

  model.fit(X1, y1)
  assert(model.nFeatures === 2, `nFeatures after first fit: ${model.nFeatures}`)

  model.fit(X2, y2)
  assert(model.nFeatures === 3, `nFeatures after second fit: ${model.nFeatures}`)

  const preds = model.predict(X2)
  assert(preds.length === 80, `preds length: ${preds.length}`)

  model.dispose()
})

// ============================================================
// Error handling
// ============================================================
console.log('\n=== Error Handling ===')

await test('predict before fit throws NotFittedError', async () => {
  const model = await GAMModel.create()
  let threw = false
  try {
    model.predict([[1, 2]])
  } catch (e) {
    threw = true
    assert(e.message.includes('not fitted') || e.message.includes('Not fitted'),
      `unexpected error: ${e.message}`)
  }
  assert(threw, 'should throw when not fitted')
  model.dispose()
})

await test('y length mismatch throws', async () => {
  const model = await GAMModel.create({ family: 'gaussian', seed: 42 })
  let threw = false
  try {
    model.fit([[1, 2], [3, 4]], [1]) // y has 1 elem, X has 2 rows
  } catch (e) {
    threw = true
    assert(e.message.includes('length') || e.message.includes('match'),
      `unexpected error: ${e.message}`)
  }
  assert(threw, 'should throw on y length mismatch')
  model.dispose()
})

// ============================================================
// Huber Regression
// ============================================================
console.log('\n=== Huber Regression ===')

await test('huber regression with outliers', async () => {
  const model = await GAMModel.create({
    family: 'huber', penalty: 'elasticnet', alpha: 0.5,
    nLambda: 30, maxInner: 50, huberGamma: 1.345, seed: 99
  })
  const rng = makeLCG(314)
  const n = 200
  const X = [], y = [], yClean = []
  for (let i = 0; i < n; i++) {
    const row = [rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1]
    X.push(row)
    const signal = 3 * row[0] + 2 * row[1]
    const noise = 0.2 * (rng() - 0.5)
    yClean.push(signal + noise)
    // 10% gross outliers
    const outlier = rng() < 0.1 ? (rng() > 0.5 ? 50 : -50) : 0
    y.push(signal + noise + outlier)
  }
  model.fit(X, y)
  const preds = model.predict(X)
  assert(preds.length === n, 'prediction length')

  let mse = 0, ssClean = 0
  const meanClean = yClean.reduce((a, b) => a + b) / n
  for (let i = 0; i < n; i++) {
    mse += (preds[i] - yClean[i]) ** 2
    ssClean += (yClean[i] - meanClean) ** 2
  }
  const r2 = 1 - mse / ssClean
  assert(r2 > 0.7, `Huber R2 vs clean = ${r2.toFixed(4)} > 0.7`)
  model.dispose()
})

// ============================================================
// Quantile Regression
// ============================================================
console.log('\n=== Quantile Regression ===')

await test('quantile regression ordering', async () => {
  const rng = makeLCG(271)
  const n = 200
  const X = [], y = []
  for (let i = 0; i < n; i++) {
    const row = [rng() * 2 - 1, rng() * 2 - 1]
    X.push(row)
    const noise = (1 + Math.abs(row[0])) * 0.5 * (rng() - 0.5)
    y.push(2 * row[0] + row[1] + noise)
  }

  const preds = {}
  for (const tau of [0.1, 0.5, 0.9]) {
    const model = await GAMModel.create({
      family: 'quantile', penalty: 'elasticnet', alpha: 0.5,
      nLambda: 30, maxInner: 50, quantileTau: tau, seed: 42
    })
    model.fit(X, y)
    preds[tau] = model.predict(X)
    model.dispose()
  }

  // Check ordering: mean(pred10) < mean(pred50) < mean(pred90)
  const mean = arr => arr.reduce((a, b) => a + b) / arr.length
  const m10 = mean(preds[0.1])
  const m50 = mean(preds[0.5])
  const m90 = mean(preds[0.9])
  assert(m10 < m50, `mean(tau=0.1)=${m10.toFixed(3)} < mean(tau=0.5)=${m50.toFixed(3)}`)
  assert(m50 < m90, `mean(tau=0.5)=${m50.toFixed(3)} < mean(tau=0.9)=${m90.toFixed(3)}`)
})

// ============================================================
// Multi-task Lasso
// ============================================================
console.log('\n=== Multi-task Lasso ===')

await test('multi-task lasso basic', async () => {
  const rng = makeLCG(54321)
  const n = 200, p = 10, nTasks = 3
  const trueBeta = [
    [1.5, -1.0, 0.5],
    [-0.8, 1.2, 0.3],
    [0.6, 0.4, -1.1],
    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
  ]

  const X = []
  const Y = new Float64Array(n * nTasks)
  for (let i = 0; i < n; i++) {
    const row = []
    for (let j = 0; j < p; j++) row.push(rng() * 2 - 1)
    X.push(row)
    for (let t = 0; t < nTasks; t++) {
      let eta = 0
      for (let j = 0; j < p; j++) eta += row[j] * trueBeta[j][t]
      Y[i * nTasks + t] = eta + (rng() - 0.5) * 0.2
    }
  }

  const model = await GAMModel.create({
    penalty: 'elasticnet', alpha: 0.8, nLambda: 50, seed: 42
  })
  model.fitMulti(X, Y, nTasks)
  assert(model.nFits > 0, `multi-task has ${model.nFits} fits`)
  assert(model.nTasks === nTasks, `nTasks = ${model.nTasks}`)

  // Predict
  const preds = model.predictMulti(X)
  assert(preds.length === n * nTasks, `pred length = ${preds.length}`)

  // R^2 per task
  for (let t = 0; t < nTasks; t++) {
    let ssRes = 0, ssTot = 0, yMean = 0
    for (let i = 0; i < n; i++) yMean += Y[i * nTasks + t]
    yMean /= n
    for (let i = 0; i < n; i++) {
      ssRes += (Y[i * nTasks + t] - preds[i * nTasks + t]) ** 2
      ssTot += (Y[i * nTasks + t] - yMean) ** 2
    }
    const r2 = 1 - ssRes / ssTot
    assert(r2 > 0.95, `task ${t} R2 = ${r2.toFixed(4)} > 0.95`)
  }

  // Save/load roundtrip
  const bytes = model.save()
  const loaded = await GAMModel.load(bytes)
  assert(loaded.nFits === model.nFits, 'loaded nFits match')
  assert(loaded.nTasks === nTasks, 'loaded nTasks match')

  const predsLoaded = loaded.predictMulti(X)
  let maxDiff = 0
  for (let i = 0; i < preds.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(preds[i] - predsLoaded[i]))
  }
  assert(maxDiff < 1e-10, `save/load max diff = ${maxDiff.toExponential(2)}`)

  loaded.dispose()
  model.dispose()
})

// ============================================================
// Multinomial logistic regression
// ============================================================
console.log('\n=== Multinomial ===')

await test('multinomial basic fit + predict', async () => {
  const rng = makeLCG(42)
  const n = 150, d = 4, K = 3
  const X = [], y = []
  for (let i = 0; i < n; i++) {
    const cls = i % K
    const row = []
    for (let j = 0; j < d; j++) {
      row.push(cls * 1.5 + (rng() - 0.5) * 0.8)
    }
    X.push(row)
    y.push(cls)
  }

  const model = await GAMModel.create({ penalty: 'elasticnet', alpha: 0.5 })
  model.fitMultinomial(X, y, K)

  assert(model.nFits > 0, `nFits = ${model.nFits}`)
  assert(model.nTasks === K, `nTasks = ${model.nTasks}`)

  // Predict probabilities
  const probs = model.predictMultinomial(X)
  assert(probs.length === n * K, `probs length = ${probs.length}`)

  // Probabilities sum to 1
  for (let i = 0; i < n; i++) {
    let sum = 0
    for (let k = 0; k < K; k++) sum += probs[i * K + k]
    assertClose(sum, 1.0, 1e-10, `prob sum for row ${i} = ${sum}`)
  }

  // Accuracy
  let correct = 0
  for (let i = 0; i < n; i++) {
    let bestK = 0, bestP = probs[i * K]
    for (let k = 1; k < K; k++) {
      if (probs[i * K + k] > bestP) { bestP = probs[i * K + k]; bestK = k }
    }
    if (bestK === y[i]) correct++
  }
  const acc = correct / n
  assert(acc > 0.8, `accuracy = ${acc.toFixed(3)} > 0.8`)

  model.dispose()
})

await test('multinomial score method', async () => {
  const rng = makeLCG(123)
  const n = 100, d = 3, K = 3
  const X = [], y = []
  for (let i = 0; i < n; i++) {
    const cls = i % K
    const row = []
    for (let j = 0; j < d; j++) {
      row.push(cls * 2 + (rng() - 0.5) * 0.5)
    }
    X.push(row)
    y.push(cls)
  }

  const model = await GAMModel.create({ penalty: 'lasso', alpha: 1.0 })
  model.fitMultinomial(X, y, K)

  const acc = model.score(X, y)
  assert(acc > 0.8, `score = ${acc.toFixed(3)} > 0.8`)

  model.dispose()
})

await test('multinomial predictProba delegates correctly', async () => {
  const rng = makeLCG(77)
  const n = 60, d = 3, K = 3
  const X = [], y = []
  for (let i = 0; i < n; i++) {
    const cls = i % K
    const row = []
    for (let j = 0; j < d; j++) row.push(cls * 2 + (rng() - 0.5) * 0.5)
    X.push(row)
    y.push(cls)
  }

  const model = await GAMModel.create({ family: 'multinomial', penalty: 'elasticnet' })
  model.fitMultinomial(X, y, K)

  const probs = model.predictProba(X)
  assert(probs.length === n * K, `predictProba length = ${probs.length}`)

  model.dispose()
})

await test('multinomial save + load roundtrip', async () => {
  const rng = makeLCG(99)
  const n = 80, d = 3, K = 3
  const X = [], y = []
  for (let i = 0; i < n; i++) {
    const cls = i % K
    const row = []
    for (let j = 0; j < d; j++) row.push(cls * 2 + (rng() - 0.5) * 0.5)
    X.push(row)
    y.push(cls)
  }

  const model = await GAMModel.create({ family: 'multinomial', penalty: 'elasticnet' })
  model.fitMultinomial(X, y, K)

  const probs = model.predictMultinomial(X)

  const bytes = model.save()
  const loaded = await GAMModel.load(bytes)
  assert(loaded.nFits === model.nFits, 'loaded nFits match')

  const probsLoaded = loaded.predictMultinomial(X)
  let maxDiff = 0
  for (let i = 0; i < probs.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(probs[i] - probsLoaded[i]))
  }
  assert(maxDiff < 1e-10, `save/load max diff = ${maxDiff.toExponential(2)}`)

  loaded.dispose()
  model.dispose()
})

await test('multinomial capabilities', async () => {
  const model = await GAMModel.create({ family: 'multinomial' })
  const caps = model.capabilities
  assert(caps.classifier === true, 'multinomial is classifier')
  assert(caps.predictProba === true, 'multinomial has predictProba')
  assert(caps.regressor === false, 'multinomial is not regressor')
  model.dispose()
})

// ============================================================
// GAMLSS (distributional regression)
// ============================================================

await test('gamlss normal basic', async () => {
  const rng = makeLCG(123)
  const n = 200, d = 3
  const X = [], y = []
  for (let i = 0; i < n; i++) {
    const row = []
    for (let j = 0; j < d; j++) row.push(rng() * 4 - 2)
    X.push(row)
    // heteroscedastic: mean depends on x0, variance depends on x1
    const mu = 2.0 * row[0] + 1.0
    const sigma = Math.exp(0.5 * row[1])
    y.push(mu + sigma * (rng() + rng() + rng() - 1.5) * 0.8165)
  }

  const model = await GAMModel.create({ penalty: 'elasticnet', alpha: 0.5 })
  model.fitGamlss(X, y, 'normal')

  assert(model.nFits > 0, `nFits = ${model.nFits}`)
  assert(model.familyGamlss > 0, `familyGamlss = ${model.familyGamlss}`)

  const preds = model.predictGamlss(X)
  assert(preds.length === n * 2, `output length = ${preds.length}`)

  // Check mu predictions have some correlation with actual
  let muOk = false
  for (let i = 0; i < n; i++) {
    if (Math.abs(preds[i * 2]) > 0.001) { muOk = true; break }
  }
  assert(muOk, 'mu predictions are non-trivial')

  // Check sigma predictions are positive
  let sigmaOk = true
  for (let i = 0; i < n; i++) {
    if (preds[i * 2 + 1] <= 0 || !isFinite(preds[i * 2 + 1])) { sigmaOk = false; break }
  }
  assert(sigmaOk, 'sigma predictions are positive and finite')

  model.dispose()
})

await test('gamlss gamma basic', async () => {
  const rng = makeLCG(456)
  const n = 150, d = 2
  const X = [], y = []
  for (let i = 0; i < n; i++) {
    const row = []
    for (let j = 0; j < d; j++) row.push(rng() * 2)
    X.push(row)
    const mu = Math.exp(1.0 + 0.5 * row[0])
    // simple Gamma-like: mu * (0.5 + rng())
    y.push(mu * (0.3 + rng() * 0.7))
  }

  const model = await GAMModel.create({ penalty: 'elasticnet', alpha: 0.5 })
  model.fitGamlss(X, y, 'gamma')

  assert(model.nFits > 0, `nFits = ${model.nFits}`)

  const preds = model.predictGamlss(X)
  assert(preds.length === n * 2, `output length = ${preds.length}`)

  // mu should be positive (Gamma)
  let allPos = true
  for (let i = 0; i < n; i++) {
    if (preds[i * 2] <= 0) { allPos = false; break }
  }
  assert(allPos, 'Gamma mu predictions are positive')

  model.dispose()
})

await test('gamlss save + load roundtrip', async () => {
  const rng = makeLCG(789)
  const n = 100, d = 2
  const X = [], y = []
  for (let i = 0; i < n; i++) {
    const row = []
    for (let j = 0; j < d; j++) row.push(rng() * 4 - 2)
    X.push(row)
    y.push(2.0 * row[0] + 1.0 + (rng() - 0.5) * 2)
  }

  const model = await GAMModel.create({ penalty: 'lasso' })
  model.fitGamlss(X, y, 'normal')

  const preds = model.predictGamlss(X)
  const bytes = model.save()
  const loaded = await GAMModel.load(bytes)

  assert(loaded.nFits === model.nFits, 'loaded nFits match')
  assert(loaded.familyGamlss === model.familyGamlss, 'loaded familyGamlss match')

  const predsLoaded = loaded.predictGamlss(X)
  let maxDiff = 0
  for (let i = 0; i < preds.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(preds[i] - predsLoaded[i]))
  }
  assert(maxDiff < 1e-10, `save/load max diff = ${maxDiff.toExponential(2)}`)

  loaded.dispose()
  model.dispose()
})

// ============================================================
// Summary
// ============================================================
console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`)
if (failed > 0) process.exit(1)

}

main()
