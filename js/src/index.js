const { loadGAM, getWasm } = require('./wasm.js')
const { GAMModel: GAMModelImpl } = require('./model.js')
const { createModelClass } = require('@wlearn/core')

const GAMModel = createModelClass(GAMModelImpl, GAMModelImpl, { name: 'GAMModel', load: loadGAM })

// Convenience: create, fit, return fitted model
async function train(params, X, y) {
  const model = await GAMModel.create(params)
  await model.fit(X, y)
  return model
}

// Convenience: load WLRN bundle and predict, auto-disposes model
async function predict(bundleBytes, X) {
  const model = await GAMModel.load(bundleBytes)
  const result = model.predict(X)
  model.dispose()
  return result
}

module.exports = { loadGAM, getWasm, GAMModel, train, predict }
