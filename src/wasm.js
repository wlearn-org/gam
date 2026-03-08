// WASM loader -- loads the GAM WASM module (singleton, lazy init)

let wasmModule = null
let loading = null

async function loadGAM(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    const createGAM = require('../wasm/gam.cjs')
    wasmModule = await createGAM(options)
    return wasmModule
  })()

  return loading
}

function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadGAM() first')
  return wasmModule
}

module.exports = { loadGAM, getWasm }
