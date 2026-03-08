#!/bin/bash
set -euo pipefail

# Build GAM C11 core as WASM via Emscripten
# Prerequisites: emsdk activated (emcc in PATH)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/wasm"

# Verify prerequisites
if ! command -v emcc &> /dev/null; then
  echo "ERROR: emcc not found. Activate emsdk first:"
  echo "  source /path/to/emsdk/emsdk_env.sh"
  exit 1
fi

echo "=== Compiling WASM ==="
mkdir -p "$OUTPUT_DIR"

EXPORTED_FUNCTIONS='[
  "_wl_gam_get_last_error",
  "_wl_gam_fit",
  "_wl_gam_fit_lambda",
  "_wl_gam_fit_pf",
  "_wl_gam_fit_groups",
  "_wl_gam_fit_cox",
  "_wl_gam_fit_multi",
  "_wl_gam_predict_multi",
  "_wl_gam_get_n_tasks",
  "_wl_gam_fit_multinomial",
  "_wl_gam_predict_multinomial",
  "_wl_gam_fit_gamlss",
  "_wl_gam_predict_gamlss",
  "_wl_gam_get_family_gamlss",
  "_wl_gam_predict",
  "_wl_gam_predict_eta",
  "_wl_gam_predict_proba",
  "_wl_gam_get_n_fits",
  "_wl_gam_get_n_features",
  "_wl_gam_get_n_coefs",
  "_wl_gam_get_family",
  "_wl_gam_get_idx_min",
  "_wl_gam_get_idx_1se",
  "_wl_gam_get_lambda",
  "_wl_gam_get_deviance",
  "_wl_gam_get_df",
  "_wl_gam_get_cv_mean",
  "_wl_gam_get_cv_se",
  "_wl_gam_get_coef",
  "_wl_gam_save",
  "_wl_gam_load",
  "_wl_gam_free",
  "_wl_gam_free_buffer",
  "_wl_gam_bspline_basis",
  "_malloc",
  "_free"
]'

EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue","HEAPF64","HEAPU8","HEAP32"]'

emcc \
  "${PROJECT_DIR}/csrc/gam.c" \
  "${PROJECT_DIR}/csrc/wl_api.c" \
  -I "${PROJECT_DIR}/csrc" \
  -o "${OUTPUT_DIR}/gam.js" \
  -std=c11 \
  -s MODULARIZE=1 \
  -s SINGLE_FILE=1 \
  -s EXPORT_NAME=createGAM \
  -s EXPORTED_FUNCTIONS="${EXPORTED_FUNCTIONS}" \
  -s EXPORTED_RUNTIME_METHODS="${EXPORTED_RUNTIME_METHODS}" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=16777216 \
  -s ENVIRONMENT='web,node' \
  -O2

echo "=== Verifying exports ==="
bash "${SCRIPT_DIR}/verify-exports.sh"

echo "=== Writing BUILD_INFO ==="
cat > "${OUTPUT_DIR}/BUILD_INFO" <<EOF
upstream: none (C11 from scratch)
build_date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
emscripten: $(emcc --version | head -1)
build_flags: -O2 -std=c11 SINGLE_FILE=1
wasm_embedded: true
EOF

echo "=== Build complete ==="
ls -lh "${OUTPUT_DIR}/gam.js"
cat "${OUTPUT_DIR}/BUILD_INFO"
