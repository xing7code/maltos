"""Pull the AI2 OLMo-2 13B SFT loss series for the deck's comparison figure.

Their run logs a *sum*-reduced loss (one example per device), so the raw series
is dominated by response-length variance, not by model quality.  The figure
therefore compares each curve against its own first value, after smoothing.

  export WANDB_API_KEY=...
  PYTHONPATH=. .venv/bin/python tools/fetch_ai2_loss.py --list
  PYTHONPATH=. .venv/bin/python tools/fetch_ai2_loss.py --key train_loss
"""

from __future__ import annotations

import argparse
import csv

RUN = "ai2-llm/open_instruct_public/e5e33952"
TOKENS_PER_STEP = 1.055e9 / 26932  # their own totals: ~39.2 K tokens/step


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=RUN)
    p.add_argument("--key", default=None, help="metric name; use --list first")
    p.add_argument("--list", action="store_true", help="print available metric names")
    p.add_argument("--max-tokens", type=float, default=105e6)
    p.add_argument("--window", type=int, default=51, help="moving-average window")
    p.add_argument("--out", default="workspace/olmo2_13b_sft/ai2_loss.csv")
    args = p.parse_args()

    import wandb

    run = wandb.Api(timeout=60).run(args.run)
    if args.list or not args.key:
        keys = sorted(k for k in run.summary.keys() if not k.startswith("_"))
        print("summary keys:", keys)
        row = next(iter(run.scan_history()), {})
        print("history keys:", sorted(k for k in row if not k.startswith("_")))
        return

    steps, vals, toks = [], [], []
    for i, row in enumerate(run.scan_history(keys=[args.key, "total_tokens"])):
        v = row.get(args.key)
        if v is None:
            continue
        t = row.get("total_tokens")
        t = float(t) if t is not None else i * TOKENS_PER_STEP
        steps.append(i)
        vals.append(float(v))
        toks.append(t)
        if t > args.max_tokens:
            break

    half = args.window // 2
    smooth = [
        sum(vals[max(0, i - half) : min(len(vals), i + half + 1)])
        / len(vals[max(0, i - half) : min(len(vals), i + half + 1)])
        for i in range(len(vals))
    ]
    base = smooth[0] if smooth else 1.0

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "tokens", "raw", "smoothed", "normalized"])
        for s, t, v, sm in zip(steps, toks, vals, smooth):
            w.writerow([s, round(t), v, round(sm, 4), round(sm / base, 4)])

    print(f"{len(vals)} points -> {args.out}")
    print(f"raw first/last   : {vals[0]:.2f} / {vals[-1]:.2f}")
    print(f"smoothed first/last: {smooth[0]:.2f} / {smooth[-1]:.2f}")
    print(f"normalized last  : {smooth[-1] / base:.3f}")
    print(f"tokens first/last: {toks[0]:,.0f} / {toks[-1]:,.0f}")


if __name__ == "__main__":
    main()
