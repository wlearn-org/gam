// WASM loader -- loads the GAM WASM module (singleton, lazy init)

import { createRequire } from 'module'

let wasmModule = null
let loading = null

export async function loadGAM(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    const require = createRequire(import.meta.url)
    const createGAM = require('../wasm/gam.cjs')
    wasmModule = await createGAM(options)
    return wasmModule
  })()

  return loading
}

export function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadGAM() first')
  return wasmModule
}
