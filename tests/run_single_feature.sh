#!/usr/bin/env bash
# Sequential single-feature / targeted regression tests.
# Always uses local subprocess launches; no fancy matrix orchestration here.
#
# Usage:
#   bash tests/run_single_feature.sh

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-.}"

echo "=== compileall ==="
"${PYTHON_BIN}" -m compileall data models parallel runtime state tests train utils tools

echo "=== persistent chained work refire ==="
"${PYTHON_BIN}" tests/chained_work_refire.py

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

echo "=== tiny model ddp equivalence ==="
"${PYTHON_BIN}" tests/tiny_model_ddp_runtime_core_equivalence.py --ddp-mode naive
"${PYTHON_BIN}" tests/tiny_model_ddp_runtime_core_equivalence.py --ddp-mode async
"${PYTHON_BIN}" tests/tiny_model_ddp_runtime_core_equivalence.py --ddp-mode bucket

echo "=== tiny model zero1/2/3 equivalence ==="
"${PYTHON_BIN}" tests/tiny_model_zero1_runtime_core_equivalence.py
"${PYTHON_BIN}" tests/tiny_model_zero2_runtime_core_equivalence.py
"${PYTHON_BIN}" tests/tiny_model_zero3_runtime_core_equivalence.py
"${PYTHON_BIN}" tests/tiny_model_zero3_runtime_core_equivalence.py --grad-accum-steps 2

echo "=== grad clip global norm equivalence ==="
"${PYTHON_BIN}" tests/tiny_model_zero1_grad_clip_equivalence.py
"${PYTHON_BIN}" tests/tiny_model_zero2_grad_clip_equivalence.py
"${PYTHON_BIN}" tests/tiny_model_zero3_grad_clip_equivalence.py
"${PYTHON_BIN}" tests/tiny_transformer_tp_grad_clip_equivalence.py

echo "=== tiny transformer tp equivalence ==="
"${PYTHON_BIN}" tests/tiny_transformer_tp_runtime_core_equivalence.py
"${PYTHON_BIN}" tests/tiny_transformer_tp_runtime_core_equivalence.py --use-sp true
"${PYTHON_BIN}" tests/tiny_transformer_tp_runtime_core_equivalence.py --world-size 4 --tp-size 4
"${PYTHON_BIN}" tests/tiny_transformer_tp_runtime_core_equivalence.py --world-size 4 --tp-size 4 --use-sp true

echo "=== tp checkpoint manifest ==="
"${PYTHON_BIN}" tests/tiny_transformer_tp_checkpoint_manifest.py

echo "=== tiny transformer pp equivalence ==="
for c in pp pp_ddp_sync pp_zero1 pp_zero2 pp_zero3 pp_tp pp_tp_sp cp_pp; do
  echo "--- pp case: ${c} ---"
  if [ "${c}" = "pp" ]; then
    "${PYTHON_BIN}" tests/tiny_transformer_pp_runtime_core_equivalence.py --case "${c}" --world-size 2 --dp-size 1 --pp-size 2
  elif [ "${c}" = "cp_pp" ]; then
    "${PYTHON_BIN}" tests/tiny_transformer_pp_runtime_core_equivalence.py --case "${c}" --world-size 4 --dp-size 1 --pp-size 2 --cp-size 2
  elif [ "${c}" = "pp_tp" ] || [ "${c}" = "pp_tp_sp" ]; then
    "${PYTHON_BIN}" tests/tiny_transformer_pp_runtime_core_equivalence.py --case "${c}" --world-size 4 --dp-size 1 --pp-size 2 --tp-size 2
  else
    "${PYTHON_BIN}" tests/tiny_transformer_pp_runtime_core_equivalence.py --case "${c}" --world-size 4 --dp-size 2 --pp-size 2
  fi
done

echo "=== tiny transformer integration matrix ==="
for c in tp_sp tp_sp_ddp_sync tp_sp_ddp_async tp_sp_ddp_bucket tp_sp_zero1 tp_sp_zero2 tp_sp_zero3; do
  echo "--- case: ${c} ---"
  "${PYTHON_BIN}" tests/tiny_transformer_runtime_core_integration.py --case "${c}"
done

echo "=== tiny transformer cp equivalence ==="
"${PYTHON_BIN}" tests/tiny_transformer_cp_runtime_core_equivalence.py --case cp --world-size 2 --dp-size 1 --cp-size 2
"${PYTHON_BIN}" tests/tiny_transformer_cp_runtime_core_equivalence.py --case cp --world-size 2 --dp-size 1 --cp-size 2 --cp-attn-core ring
"${PYTHON_BIN}" tests/tiny_transformer_cp_runtime_core_equivalence.py --case cp_ddp_sync --world-size 4 --dp-size 2 --cp-size 2
"${PYTHON_BIN}" tests/tiny_transformer_cp_runtime_core_equivalence.py --case cp_ddp_async --world-size 4 --dp-size 2 --cp-size 2
"${PYTHON_BIN}" tests/tiny_transformer_cp_runtime_core_equivalence.py --case cp_ddp_bucket --world-size 4 --dp-size 2 --cp-size 2
"${PYTHON_BIN}" tests/tiny_transformer_cp_runtime_core_equivalence.py --case cp_zero1 --world-size 4 --dp-size 2 --cp-size 2
"${PYTHON_BIN}" tests/tiny_transformer_cp_runtime_core_equivalence.py --case cp_zero2 --world-size 4 --dp-size 2 --cp-size 2
"${PYTHON_BIN}" tests/tiny_transformer_cp_runtime_core_equivalence.py --case cp_zero3 --world-size 4 --dp-size 2 --cp-size 2
"${PYTHON_BIN}" tests/tiny_transformer_cp_runtime_core_equivalence.py --case cp_zero3 --world-size 4 --dp-size 2 --cp-size 2 --cp-attn-core ring
"${PYTHON_BIN}" tests/tiny_transformer_cp_runtime_core_equivalence.py --case cp --world-size 4 --dp-size 1 --cp-size 2 --tp-size 2
"${PYTHON_BIN}" tests/tiny_transformer_cp_runtime_core_equivalence.py --case cp --world-size 4 --dp-size 1 --cp-size 2 --tp-size 2 --use-sp

echo "=== tiny transformer ep equivalence ==="
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep --world-size 2 --dp-size 2 --ep-size 2 --tp-size 1
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep_ddp_sync --world-size 2 --dp-size 2 --ep-size 2 --tp-size 1
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep_ddp_async --world-size 2 --dp-size 2 --ep-size 2 --tp-size 1
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep_ddp_bucket --world-size 2 --dp-size 2 --ep-size 2 --tp-size 1
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep_zero1 --world-size 2 --dp-size 2 --ep-size 2 --tp-size 1
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep_zero2 --world-size 2 --dp-size 2 --ep-size 2 --tp-size 1
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep_zero3 --world-size 2 --dp-size 2 --ep-size 2 --tp-size 1
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep_tp --world-size 4 --dp-size 2 --ep-size 2 --tp-size 2
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep_tp_sp --world-size 4 --dp-size 2 --ep-size 2 --tp-size 2
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep --world-size 4 --dp-size 4 --ep-size 2 --tp-size 1
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep_ddp_sync --world-size 4 --dp-size 4 --ep-size 2 --tp-size 1
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep_ddp_async --world-size 4 --dp-size 4 --ep-size 2 --tp-size 1
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep_ddp_bucket --world-size 4 --dp-size 4 --ep-size 2 --tp-size 1
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep_tp --world-size 8 --dp-size 4 --ep-size 2 --tp-size 2
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep_tp_sp --world-size 8 --dp-size 4 --ep-size 2 --tp-size 2
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep_zero1 --world-size 4 --dp-size 4 --ep-size 2 --tp-size 1
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep_zero2 --world-size 4 --dp-size 4 --ep-size 2 --tp-size 1
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep_zero3 --world-size 4 --dp-size 4 --ep-size 2 --tp-size 1
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep_tp_sp_zero1 --world-size 8 --dp-size 4 --ep-size 2 --tp-size 2
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep_tp_sp_zero2 --world-size 8 --dp-size 4 --ep-size 2 --tp-size 2
"${PYTHON_BIN}" tests/tiny_transformer_ep_runtime_core_equivalence.py --case ep_tp_sp_zero3 --world-size 8 --dp-size 4 --ep-size 2 --tp-size 2
"${PYTHON_BIN}" tests/tiny_transformer_ep_tp_sp_erep_subgroup_equivalence.py --world-size 4 --tp-size 4 --ep-size 2
"${PYTHON_BIN}" tests/tiny_transformer_ep_grad_accum2_equivalence.py --world-size 4 --ep-size 2

echo "=== tiny transformer ep+cp+zero equivalence ==="
"${PYTHON_BIN}" tests/tiny_transformer_ep_cp_zero_equivalence.py --case ep_cp_zero1 --world-size 4 --dp-size 2 --cp-size 2 --ep-size 2
"${PYTHON_BIN}" tests/tiny_transformer_ep_cp_zero_equivalence.py --case ep_cp_zero2 --world-size 4 --dp-size 2 --cp-size 2 --ep-size 2
"${PYTHON_BIN}" tests/tiny_transformer_ep_cp_zero_equivalence.py --case ep_cp_zero3 --world-size 4 --dp-size 2 --cp-size 2 --ep-size 2
"${PYTHON_BIN}" tests/tiny_transformer_ep_cp_zero_equivalence.py --case ep_cp_zero1 --world-size 4 --dp-size 2 --cp-size 2 --ep-size 2 --no-reuse-cp-for-ep
"${PYTHON_BIN}" tests/tiny_transformer_ep_cp_zero_equivalence.py --case ep_cp_zero2 --world-size 4 --dp-size 2 --cp-size 2 --ep-size 2 --no-reuse-cp-for-ep
"${PYTHON_BIN}" tests/tiny_transformer_ep_cp_zero_equivalence.py --case ep_cp_zero3 --world-size 4 --dp-size 2 --cp-size 2 --ep-size 2 --no-reuse-cp-for-ep

echo "=== tiny transformer ep+pp+zero equivalence ==="
for c in ep_pp_zero0 ep_pp_zero1 ep_pp_zero2 ep_pp_zero3; do
  echo "--- ep_pp case: ${c} ---"
  "${PYTHON_BIN}" tests/tiny_transformer_ep_pp_zero_equivalence.py --case "${c}"
done

echo "=== ep full-stack no-reuse targeted coverage ==="
"${PYTHON_BIN}" tests/tiny_transformer_ep_full_stack_equivalence.py \
  --world-size 8 --dp-size 2 --pp-size 2 --cp-size 2 --tp-size 1 --ep-size 2 \
  --zero-stage 1 --cp-attn-core ring --no-reuse-cp-for-ep
"${PYTHON_BIN}" tests/tiny_transformer_ep_full_stack_equivalence.py \
  --world-size 8 --dp-size 2 --pp-size 2 --cp-size 1 --tp-size 2 --ep-size 2 \
  --zero-stage 1 --no-reuse-tp-for-ep
"${PYTHON_BIN}" tests/tiny_transformer_ep_full_stack_equivalence.py \
  --world-size 8 --dp-size 2 --pp-size 1 --cp-size 2 --tp-size 2 --ep-size 2 \
  --zero-stage 1 --cp-attn-core ring --no-reuse-tp-for-ep --no-reuse-cp-for-ep

echo "=== tiny transformer ep zero3 checkpoint/resume ==="
"${PYTHON_BIN}" tests/tiny_transformer_ep_zero3_checkpoint_resume.py --world-size 2 --dp-size 2 --ep-size 2
"${PYTHON_BIN}" tests/tiny_transformer_ep_zero3_accum2_midstep_resume.py --world-size 2 --dp-size 2 --ep-size 2
"${PYTHON_BIN}" tests/tiny_transformer_ep_zero3_checkpoint_resume.py --world-size 4 --dp-size 4 --ep-size 2
"${PYTHON_BIN}" tests/tiny_transformer_ep_zero3_accum2_midstep_resume.py --world-size 4 --dp-size 4 --ep-size 2 --global-batch-size 8

echo "=== tp+sp+zero3+bf16+clip checkpoint resume ==="
"${PYTHON_BIN}" tests/tiny_transformer_tp_sp_zero3_bf16_clip_checkpoint_resume.py

echo "=== tp+sp+zero3+bf16+clip+accum2 mid-step checkpoint resume ==="
"${PYTHON_BIN}" tests/tiny_transformer_tp_sp_zero3_bf16_clip_accum2_midstep_resume.py

echo "=== pretraining loader + tp+sp+zero3+bf16+clip+accum2 checkpoint resume ==="
"${PYTHON_BIN}" tests/pretraining_loader_tp_sp_zero3_bf16_clip_accum2_resume.py

echo "=== single-feature PASS ==="
