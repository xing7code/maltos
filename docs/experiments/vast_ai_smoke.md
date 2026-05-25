# Vast.ai Pretraining Smoke

Goal: run a small real-data LLaMA pretraining smoke with observable metrics and resumable checkpoints.

## 1. Setup

```bash
git clone <repo-url>
cd llm-train-systems
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Optional W&B:

```bash
wandb login
```

## 2. Prepare 10M Tokens

```bash
PYTHONPATH=. python tools/prepare_token_shards.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --config sample-10BT \
  --split train \
  --column text \
  --tokenizer-name-or-path Qwen/Qwen2.5-7B \
  --output-dir datasets/fineweb_10m \
  --max-tokens 10000000 \
  --tokens-per-shard 5000000 \
  --log-every-tokens 1000000 \
  --streaming
```

## 3. Single-GPU Smoke

```bash
PYTHONPATH=. python tools/pretrain.py \
  --config configs/llama_10m.yaml \
  --data datasets/fineweb_10m \
  --dp-size 1 \
  --tp-size 1 \
  --no-use-sp \
  --zero-stage 0 \
  --max-steps 200 \
  --log-every 10 \
  --checkpoint-dir checkpoints/llama_10m_single \
  --metrics-jsonl logs/llama_10m_single.jsonl \
  --wandb-run-name llama-10m-single
```

## 4. Resume Check

```bash
PYTHONPATH=. python tools/pretrain.py \
  --config configs/llama_10m.yaml \
  --data datasets/fineweb_10m \
  --dp-size 1 \
  --tp-size 1 \
  --no-use-sp \
  --zero-stage 0 \
  --max-steps 300 \
  --resume-from checkpoints/llama_10m_single/step_00000100 \
  --log-every 10 \
  --metrics-jsonl logs/llama_10m_single_resume.jsonl \
  --wandb-run-name llama-10m-single-resume
```

## 5. 4-GPU Main Smoke

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 tools/pretrain.py \
  --config configs/llama_10m.yaml \
  --data datasets/fineweb_10m \
  --max-steps 500 \
  --log-every 10 \
  --checkpoint-dir checkpoints/llama_10m_dp2_tp2_zero3 \
  --metrics-jsonl logs/llama_10m_dp2_tp2_zero3.jsonl \
  --wandb-run-name llama-10m-dp2-tp2-sp-zero3
```

## Metrics To Watch

- `loss`
- `lr`
- `train/tokens`
- `train/tokens_per_sec`
- `perf/step_sec`
- `grad_clip/grad_norm`
- `memory/max_allocated_gb`
- `precision/overflow`

## Scale-Up

After 10M tokens works, prepare 50M:

```bash
PYTHONPATH=. python tools/prepare_token_shards.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --config sample-10BT \
  --split train \
  --column text \
  --tokenizer-name-or-path Qwen/Qwen2.5-7B \
  --output-dir datasets/fineweb_50m \
  --max-tokens 50000000 \
  --tokens-per-shard 10000000 \
  --log-every-tokens 1000000 \
  --streaming
```
