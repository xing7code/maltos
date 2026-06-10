#!/usr/bin/env bash
# GPU test matrix — requires nccl backend (run on GPU nodes, e.g. vast.ai).
# All tests here use configurations that would deadlock or be too slow on CPU/gloo.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-.}"

echo "=== tiny transformer ep full-stack zero3 (full: dp=2 pp=2 cp=2 tp=2 ep=2) ==="
"${PYTHON_BIN}" tests/tiny_transformer_ep_full_stack_equivalence.py --world-size 16 --dp-size 2 --pp-size 2 --cp-size 2 --tp-size 2 --ep-size 2
"${PYTHON_BIN}" tests/tiny_transformer_ep_full_stack_zero3_checkpoint_resume.py --world-size 16 --dp-size 2 --pp-size 2 --cp-size 2 --tp-size 2 --ep-size 2
"${PYTHON_BIN}" tests/tiny_transformer_ep_full_stack_zero3_accum2_midstep_resume.py --world-size 16 --dp-size 2 --pp-size 2 --cp-size 2 --tp-size 2 --ep-size 2

echo "=== tiny transformer full-stack combos afab (full: dp=2 pp=2 cp=2 tp=2) ==="
"${PYTHON_BIN}" tests/tiny_transformer_full_stack_equivalence.py --world-size 16 --dp-size 2 --pp-size 2 --cp-size 2 --tp-size 2 --zero-stage 1
"${PYTHON_BIN}" tests/tiny_transformer_full_stack_equivalence.py --world-size 16 --dp-size 2 --pp-size 2 --cp-size 2 --tp-size 2 --zero-stage 2
"${PYTHON_BIN}" tests/tiny_transformer_full_stack_equivalence.py --world-size 16 --dp-size 2 --pp-size 2 --cp-size 2 --tp-size 2 --zero-stage 3

echo "=== tiny transformer full-stack zero3 checkpoint/resume (full: dp=2 pp=2 cp=2 tp=2) ==="
"${PYTHON_BIN}" tests/tiny_transformer_full_stack_zero3_checkpoint_resume.py --world-size 16 --dp-size 2 --pp-size 2 --cp-size 2 --tp-size 2
"${PYTHON_BIN}" tests/tiny_transformer_full_stack_zero3_accum2_midstep_resume.py --world-size 16 --dp-size 2 --pp-size 2 --cp-size 2 --tp-size 2

echo "=== tiny transformer full-stack 1f1b (full: dp=2 pp=2 cp=2 tp=2) ==="
"${PYTHON_BIN}" tests/tiny_transformer_full_stack_equivalence.py --pp-schedule 1f1b --world-size 16 --dp-size 2 --pp-size 2 --cp-size 2 --tp-size 2 --zero-stage 3
"${PYTHON_BIN}" tests/tiny_transformer_full_stack_zero3_checkpoint_resume.py --pp-schedule 1f1b --world-size 16 --dp-size 2 --pp-size 2 --cp-size 2 --tp-size 2
"${PYTHON_BIN}" tests/tiny_transformer_full_stack_zero3_accum2_midstep_resume.py --pp-schedule 1f1b --world-size 16 --dp-size 2 --pp-size 2 --cp-size 2 --tp-size 2

echo "=== tiny transformer ep full-stack zero3 (ep=4: dp=4 pp=2 cp=2 tp=2 ep=4) ==="
"${PYTHON_BIN}" tests/tiny_transformer_ep_full_stack_equivalence.py --world-size 32 --dp-size 4 --pp-size 2 --cp-size 2 --tp-size 2 --ep-size 4
"${PYTHON_BIN}" tests/tiny_transformer_ep_full_stack_zero3_checkpoint_resume.py --world-size 32 --dp-size 4 --pp-size 2 --cp-size 2 --tp-size 2 --ep-size 4
"${PYTHON_BIN}" tests/tiny_transformer_ep_full_stack_zero3_accum2_midstep_resume.py --world-size 32 --dp-size 4 --pp-size 2 --cp-size 2 --tp-size 2 --ep-size 4

echo "=== tiny transformer ep full-stack 1f1b zero3 ==="
"${PYTHON_BIN}" tests/tiny_transformer_ep_full_stack_equivalence.py --pp-schedule 1f1b --world-size 16 --dp-size 2 --pp-size 2 --cp-size 2 --tp-size 2 --ep-size 2
"${PYTHON_BIN}" tests/tiny_transformer_ep_full_stack_zero3_accum2_midstep_resume.py --pp-schedule 1f1b --world-size 16 --dp-size 2 --pp-size 2 --cp-size 2 --tp-size 2 --ep-size 2

echo "=== tiny transformer ep full-stack 1f1b zero3 (ep=4) ==="
"${PYTHON_BIN}" tests/tiny_transformer_ep_full_stack_equivalence.py --pp-schedule 1f1b --world-size 32 --dp-size 4 --pp-size 2 --cp-size 2 --tp-size 2 --ep-size 4
"${PYTHON_BIN}" tests/tiny_transformer_ep_full_stack_zero3_accum2_midstep_resume.py --pp-schedule 1f1b --world-size 32 --dp-size 4 --pp-size 2 --cp-size 2 --tp-size 2 --ep-size 4

echo "=== matrix_gpu PASS ==="
