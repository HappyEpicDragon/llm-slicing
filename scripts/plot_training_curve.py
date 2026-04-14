#!/usr/bin/env python3
"""Plot DT training loss curves from training_history.json.

Usage (default model only — for paper appendix):
    pixi run python scripts/plot_training_curve.py \
        --history data/channel_generality/dt/model/latest/training_history.json

With reward panel (noisy, not recommended for paper):
    pixi run python scripts/plot_training_curve.py \
        --history data/channel_generality/dt/model/latest/training_history.json \
        --show-reward

Batch mode (all sensitivity variants, for debugging):
    pixi run python scripts/plot_training_curve.py \
        --history data/channel_generality/dt/model/latest/training_history.json \
                  data/channel_generality/dt_ctx5/model/latest/training_history.json \
                  data/channel_generality/dt_ctx10/model/latest/training_history.json

Custom output:
    pixi run python scripts/plot_training_curve.py \
        --history data/channel_generality/dt/model/latest/training_history.json \
        --output outputs/figures/channel_generality/training_curve.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def plot_single(hist_path: str, output: str | None, show_reward: bool):
    with open(hist_path) as f:
        hist = json.load(f)

    epochs = list(range(1, len(hist["train_loss"]) + 1))
    has_val = any(v != 0.0 for v in hist["val_loss"])
    has_eval = show_reward and len(hist.get("eval_reward", [])) > 0

    n_panels = 1 + int(has_eval)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 3.8))
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(epochs, hist["train_loss"], label="Train Loss", linewidth=1.5)
    if has_val:
        ax.plot(epochs, hist["val_loss"], label="Val Loss", linewidth=1.5, linestyle="--")

    best_epoch = int(min(range(len(hist["val_loss"])), key=lambda i: hist["val_loss"][i])) + 1 if has_val else None
    if best_epoch is not None:
        ax.axvline(best_epoch, color="tab:red", linestyle=":", alpha=0.6, label=f"Best Val (epoch {best_epoch})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if has_eval:
        ax2 = axes[1]
        ax2.plot(hist["eval_epoch"], hist["eval_reward"],
                 marker="o", markersize=4, linewidth=1.5, color="tab:green")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Avg Episode Reward")
        ax2.set_title("Closed-Loop Evaluation")
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        out = Path(output)
    else:
        out = Path(hist_path).parent / "training_curve.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(out), bbox_inches="tight")
    png_path = str(out.with_suffix(".png"))
    fig.savefig(png_path, bbox_inches="tight")
    print(f"Saved: {out}  and  {png_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot DT training convergence curves")
    parser.add_argument("--history", required=True, nargs="+",
                        help="Path(s) to training_history.json")
    parser.add_argument("--output", default=None,
                        help="Output path (only used when a single --history is given)")
    parser.add_argument("--show-reward", action="store_true",
                        help="Include closed-loop reward panel (noisy; off by default)")
    args = parser.parse_args()

    for hp in args.history:
        if not Path(hp).exists():
            print(f"[SKIP] not found: {hp}")
            continue
        out = args.output if len(args.history) == 1 else None
        plot_single(hp, out, args.show_reward)


if __name__ == "__main__":
    main()
