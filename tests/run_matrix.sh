#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-.}"

echo "=== compileall ==="
"${PYTHON_BIN}" -m compileall data models parallel runtime state tests train utils tools

echo "=== smoke/runtime core ==="
"${PYTHON_BIN}" tests/smoke_runtime_core.py

echo "=== smoke/trainer loop ==="
"${PYTHON_BIN}" tests/smoke_trainer_loop.py

echo "=== smoke/pretrain cli ==="
"${PYTHON_BIN}" tests/smoke_pretrain_cli.py

echo "=== checkpoint manifest validation ==="
"${PYTHON_BIN}" tests/checkpoint_manifest_validation.py

echo "=== runtime optimizer checkpoint resume ==="
"${PYTHON_BIN}" tests/tiny_model_runtime_optimizer_checkpoint_resume.py

echo "=== simple dataloader checkpoint resume ==="
"${PYTHON_BIN}" tests/simple_dataloader_checkpoint_resume.py

echo "=== pretraining dataloader checkpoint resume ==="
"${PYTHON_BIN}" tests/pretraining_dataloader_resume.py

echo "=== zero3 checkpoint resume ==="
"${PYTHON_BIN}" tests/tiny_model_zero3_checkpoint_resume.py

echo "=== zero3 checkpoint roundtrip ==="
"${PYTHON_BIN}" tests/tiny_model_zero3_checkpoint_roundtrip.py

echo "=== tp checkpoint manifest ==="
"${PYTHON_BIN}" tests/tiny_transformer_tp_checkpoint_manifest.py

echo "=== tiny transformer integration matrix ==="
for c in tp_sp tp_sp_ddp_sync tp_sp_ddp_async tp_sp_ddp_bucket tp_sp_zero1 tp_sp_zero2 tp_sp_zero3; do
  echo "--- case: ${c} ---"
  "${PYTHON_BIN}" tests/tiny_transformer_runtime_core_integration.py --case "${c}"
done

echo "=== tp+sp+zero3+bf16+clip checkpoint resume ==="
"${PYTHON_BIN}" tests/tiny_transformer_tp_sp_zero3_bf16_clip_checkpoint_resume.py

echo "=== tp+sp+zero3+bf16+clip+accum2 mid-step checkpoint resume ==="
"${PYTHON_BIN}" tests/tiny_transformer_tp_sp_zero3_bf16_clip_accum2_midstep_resume.py

echo "=== pretraining loader + tp+sp+zero3+bf16+clip+accum2 checkpoint resume ==="
"${PYTHON_BIN}" tests/pretraining_loader_tp_sp_zero3_bf16_clip_accum2_resume.py

echo "=== matrix PASS ==="
