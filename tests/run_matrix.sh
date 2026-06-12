#!/usr/bin/env bash
# Full-stack matrix only.
#
# gloo:           BACKEND=gloo ./tests/run_matrix.sh
# nccl:           BACKEND=nccl ./tests/run_matrix.sh
# nccl no-merge:  BACKEND=nccl MERGE=0 ./tests/run_matrix.sh
# Continue on Vast with files:
#   edit local_notes/matrix_blacklist.txt, then run: BACKEND=nccl ./tests/run_matrix.sh
# Run only a subset:
#   edit local_notes/matrix_whitelist.txt, then run: BACKEND=gloo ./tests/run_matrix.sh
# Matrix default behavior is keep-going.
# gloo default: per-case subprocess keep-going.
# nccl default: grouped merged keep-going, with per-case hang watchdog.
# Reports are always emitted to local_notes/ by default.
#
# Full-stack matrix: 4-choose-3 from {dp, pp, cp, tp}, each=2, world=8.
# ep=dp*cp*tp (max expert sharding, no extra ranks).
#
#   A: dp=2 pp=2 cp=2 tp=1  ep=4  world=8
#   B: dp=2 pp=2 cp=1 tp=2  ep=4  world=8
#   C: dp=2 pp=1 cp=2 tp=2  ep=8  world=8
#   D: dp=1 pp=2 cp=2 tp=2  ep=4  world=8  (dp=1 → zero0 only)
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
BACKEND="${BACKEND:-gloo}"
MERGE="${MERGE:-}"
WHITELIST="${WHITELIST:-local_notes/matrix_whitelist.txt}"
BLACKLIST="${BLACKLIST:-local_notes/matrix_blacklist.txt}"
CASE_FILTER="${CASE_FILTER:-}"
MAX_CASES="${MAX_CASES:-}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-40}"
REPORT_FILE="${REPORT_FILE:-local_notes/matrix_report.log}"
FAILURES_FILE="${FAILURES_FILE:-local_notes/matrix_failures.txt}"
PASSES_FILE="${PASSES_FILE:-local_notes/matrix_passes.txt}"
export PYTHONPATH="${PYTHONPATH:-.}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-180}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-180}"
mkdir -p local_notes
touch "${WHITELIST}" "${BLACKLIST}" "${REPORT_FILE}" "${FAILURES_FILE}" "${PASSES_FILE}"

# ── Full-stack matrix (backend: ${BACKEND}) ─────────────────────────────────────
echo "=== full-stack matrix (backend=${BACKEND}) ==="
MATRIX_ARGS=(
  --backend "${BACKEND}"
  --world-size 8
)
if [ "${MERGE}" = "1" ] || [ "${MERGE}" = "true" ] || [ "${MERGE}" = "TRUE" ]; then
  MATRIX_ARGS+=(--merge)
elif [ "${MERGE}" = "0" ] || [ "${MERGE}" = "false" ] || [ "${MERGE}" = "FALSE" ]; then
  MATRIX_ARGS+=(--no-merge)
elif [ "${BACKEND}" = "nccl" ]; then
  MATRIX_ARGS+=(--merge)
fi
if [ -n "${WHITELIST}" ]; then
  MATRIX_ARGS+=(--whitelist "${WHITELIST}")
fi
if [ -n "${BLACKLIST}" ]; then
  MATRIX_ARGS+=(--blacklist "${BLACKLIST}")
fi
if [ -n "${CASE_FILTER}" ]; then
  MATRIX_ARGS+=(--case-filter "${CASE_FILTER}")
fi
if [ -n "${MAX_CASES}" ]; then
  MATRIX_ARGS+=(--max-cases "${MAX_CASES}")
fi
if [ -n "${CASE_TIMEOUT_SEC}" ]; then
  MATRIX_ARGS+=(--case-timeout-sec "${CASE_TIMEOUT_SEC}")
fi
if [ -n "${REPORT_FILE}" ]; then
  MATRIX_ARGS+=(--report-file "${REPORT_FILE}")
fi
if [ -n "${FAILURES_FILE}" ]; then
  MATRIX_ARGS+=(--failures-file "${FAILURES_FILE}")
fi
if [ -n "${PASSES_FILE}" ]; then
  MATRIX_ARGS+=(--passes-file "${PASSES_FILE}")
fi
"${PYTHON_BIN}" tests/tiny_transformer_full_stack_matrix_runner.py "${MATRIX_ARGS[@]}"
echo "=== matrix PASS ==="
