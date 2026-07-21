#!/usr/bin/env bash
# Lightweight non-matrix regressions.
#
# This script intentionally excludes the full-stack distributed transformer
# matrix. Use tests/run_matrix.sh for TP/PP/CP/EP/ZeRO full-stack coverage.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-.}"
if [ -z "${GLOO_SOCKET_IFNAME:-}" ]; then
  if [ "$(uname -s)" = "Darwin" ]; then
    export GLOO_SOCKET_IFNAME="lo0"
  else
    export GLOO_SOCKET_IFNAME="lo"
  fi
fi

echo "=== compileall ==="
"${PYTHON_BIN}" -m compileall -q data models parallel runtime state tests train utils tools

echo "=== persistent chained work refire ==="
"${PYTHON_BIN}" tests/chained_work_refire.py

echo "=== smoke/runtime core ==="
"${PYTHON_BIN}" tests/smoke_runtime_core.py

echo "=== smoke/trainer loop ==="
"${PYTHON_BIN}" tests/smoke_trainer_loop.py

echo "=== smoke/train cli ==="
"${PYTHON_BIN}" tests/smoke_train_cli.py

echo "=== checkpoint manifest validation ==="
"${PYTHON_BIN}" tests/checkpoint_manifest_validation.py

echo "=== runtime optimizer checkpoint resume ==="
"${PYTHON_BIN}" tests/tiny_model_runtime_optimizer_checkpoint_resume.py
"${PYTHON_BIN}" tests/tiny_model_runtime_optimizer_checkpoint_resume.py --dtype bf16

echo "=== dataloader checkpoint resume ==="
"${PYTHON_BIN}" tests/simple_dataloader_checkpoint_resume.py
"${PYTHON_BIN}" tests/pretraining_dataloader_resume.py
"${PYTHON_BIN}" tests/sft_dataloader_resume.py
"${PYTHON_BIN}" tests/sft_runtime_fields.py
"${PYTHON_BIN}" tests/sequence_ids_attention_mask.py

echo "=== zero3 checkpoint resume/roundtrip ==="
"${PYTHON_BIN}" tests/tiny_model_zero3_checkpoint_resume.py
"${PYTHON_BIN}" tests/tiny_model_zero3_checkpoint_roundtrip.py

echo "=== tiny model ddp equivalence ==="
"${PYTHON_BIN}" tests/tiny_model_ddp_runtime_core_equivalence.py --ddp-mode naive
"${PYTHON_BIN}" tests/tiny_model_ddp_runtime_core_equivalence.py --ddp-mode async
"${PYTHON_BIN}" tests/tiny_model_ddp_runtime_core_equivalence.py --ddp-mode bucket

echo "=== tiny model zero equivalence ==="
"${PYTHON_BIN}" tests/tiny_model_zero_runtime_core_equivalence.py --zero-stage 1 --master-port 29520
"${PYTHON_BIN}" tests/tiny_model_zero_runtime_core_equivalence.py --zero-stage 2 --master-port 29521
"${PYTHON_BIN}" tests/tiny_model_zero_runtime_core_equivalence.py --zero-stage 3 --master-port 29522
"${PYTHON_BIN}" tests/tiny_model_zero_runtime_core_equivalence.py --zero-stage 3 --grad-accum-steps 2 --master-port 29523

echo "=== tiny MoE auxiliary loss ==="
"${PYTHON_BIN}" tests/tiny_moe_aux_loss.py

echo "=== grad clip global norm equivalence ==="
"${PYTHON_BIN}" tests/tiny_model_zero_grad_clip_equivalence.py --zero-stage 1 --master-port 29545
"${PYTHON_BIN}" tests/tiny_model_zero_grad_clip_equivalence.py --zero-stage 2 --master-port 29548
"${PYTHON_BIN}" tests/tiny_model_zero_grad_clip_equivalence.py --zero-stage 3 --master-port 29546
"${PYTHON_BIN}" tests/tiny_transformer_tp_grad_clip_equivalence.py

echo "=== tp checkpoint manifest ==="
"${PYTHON_BIN}" tests/tiny_transformer_tp_checkpoint_manifest.py

echo "=== pretraining loader + tp+sp+zero3+bf16+clip+accum2 checkpoint resume ==="
"${PYTHON_BIN}" tests/pretraining_loader_tp_sp_zero3_bf16_clip_accum2_resume.py

echo "=== smoke regressions PASS ==="
