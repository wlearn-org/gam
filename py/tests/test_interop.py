"""Real native/WASM artifact checks; no canonical fixtures are rewritten."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from wlearn.bundle import decode_bundle
from wlearn_gam import GAMModel


@pytest.mark.parametrize('family,relax', [('gaussian', 0), ('gaussian', 1), ('binomial', 1), ('multinomial', 0)])
def test_native_wasm_roundtrip(tmp_path, family, relax):
    X = np.linspace(-2, 2, 80).reshape(-1, 1)
    y = 1.25 + 3 * X[:, 0] if family == 'gaussian' else (np.arange(80) % 3 == 0).astype(float)
    if family == 'multinomial':
        y = np.array([-5, 3, 9])[np.arange(80) % 3]
    params = {'family': family, 'relax': relax, 'penalty': 'lasso',
              'nLambda': 3, 'lambdaMinRatio': 0.3}
    py_path, js_path, resaved = [tmp_path / name for name in ['py.wlrn', 'js.wlrn', 'resaved.wlrn']]
    model = GAMModel(params).fit(X, y)
    try:
        model.save(py_path)
        js = r'''
const fs = require('node:fs')
const req = JSON.parse(fs.readFileSync(0, 'utf8'))
const { GAMModel } = require(req.package)
;(async () => {
  const fromPython = await GAMModel.load(fs.readFileSync(req.py))
  const fresh = await GAMModel.create(req.params)
  try {
    fresh.fit(req.X, req.y)
    fs.writeFileSync(req.js, fresh.save())
    fs.writeFileSync(req.resaved, fromPython.save())
    console.log(JSON.stringify({
      ordinary: Array.from(fromPython.predict(req.X, 2)),
      relaxed: fromPython.hasRelaxed ? Array.from(fromPython.predictRelaxed(req.X, 2)) : null,
    }))
  } finally { fromPython.dispose(); fresh.dispose() }
})().catch(error => { console.error(error); process.exitCode = 1 })
'''
        request = {'package': os.environ.get('GAM_JS_PACKAGE', str(Path(__file__).resolve().parents[2] / 'js')),
                   'py': str(py_path), 'js': str(js_path), 'resaved': str(resaved),
                   'X': X.tolist(), 'y': y.tolist(), 'params': params}
        result = subprocess.run(['node', '-e', js], input=json.dumps(request),
                                capture_output=True, text=True, check=True)
        predictions = json.loads(result.stdout.strip().splitlines()[-1])
        np.testing.assert_allclose(predictions['ordinary'], model.predict(X, 2), atol=1e-5)
        if relax:
            np.testing.assert_allclose(predictions['relaxed'], model.predict_relaxed(X, 2), atol=1e-5)
        before, after = decode_bundle(py_path.read_bytes()), decode_bundle(resaved.read_bytes())
        assert before[:2] == after[:2]
        assert bytes(before[2]) == bytes(after[2])

        loaded = GAMModel.load(js_path)
        try:
            original, saved = decode_bundle(js_path.read_bytes()), decode_bundle(loaded.save())
            assert original[:2] == saved[:2]
            assert bytes(original[2]) == bytes(saved[2])
            np.testing.assert_allclose(loaded.predict(X, 2), model.predict(X, 2), atol=1e-5)
            if relax:
                np.testing.assert_allclose(loaded.predict_relaxed(X, 2), model.predict_relaxed(X, 2), atol=1e-5)
        finally:
            loaded.dispose()
    finally:
        model.dispose()
