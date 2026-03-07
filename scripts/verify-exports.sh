#!/bin/bash
set -euo pipefail

# Verify that the built WASM module exports all expected symbols.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WASM_FILE="${PROJECT_DIR}/wasm/gam.cjs"

if [ ! -f "$WASM_FILE" ]; then
  echo "ERROR: ${WASM_FILE} not found. Run build-wasm.sh first."
  exit 1
fi

EXPECTED_EXPORTS=(
  wl_gam_get_last_error
  wl_gam_fit
  wl_gam_fit_lambda
  wl_gam_fit_pf
  wl_gam_fit_groups
  wl_gam_fit_cox
  wl_gam_fit_multi
  wl_gam_predict_multi
  wl_gam_get_n_tasks
  wl_gam_fit_multinomial
  wl_gam_predict_multinomial
  wl_gam_fit_gamlss
  wl_gam_predict_gamlss
  wl_gam_get_family_gamlss
  wl_gam_predict
  wl_gam_predict_eta
  wl_gam_predict_proba
  wl_gam_get_n_fits
  wl_gam_get_n_features
  wl_gam_get_n_coefs
  wl_gam_get_family
  wl_gam_get_idx_min
  wl_gam_get_idx_1se
  wl_gam_get_lambda
  wl_gam_get_deviance
  wl_gam_get_df
  wl_gam_get_cv_mean
  wl_gam_get_cv_se
  wl_gam_get_coef
  wl_gam_save
  wl_gam_load
  wl_gam_free
  wl_gam_free_buffer
  wl_gam_bspline_basis
)

missing=0
for fn in "${EXPECTED_EXPORTS[@]}"; do
  if ! grep -q "_${fn}" "$WASM_FILE"; then
    echo "MISSING: _${fn}"
    missing=$((missing + 1))
  fi
done

if [ $missing -gt 0 ]; then
  echo "ERROR: ${missing} exports missing from ${WASM_FILE}"
  exit 1
fi

echo "All ${#EXPECTED_EXPORTS[@]} exports verified."
