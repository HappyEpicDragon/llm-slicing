#!/usr/bin/env python3
"""Phase 3 Fig.C: IDT vs PPO-Lagrangian-Discrete degradation histogram on s9."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _stats(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "crash": int(np.sum(x > 0.3)),
        "n": int(x.size),
    }


def _textbox_text(name: str, st: dict[str, float]) -> str:
    return (
        f"{name}\n"
        f"mean={st['mean']:.3f}\n"
        f"std={st['std']:.3f}\n"
        f"median={st['median']:.3f}\n"
        f"p95={st['p95']:.3f}\n"
        f"crash(>0.3)={int(st['crash'])}/{int(st['n'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dt_npz",
        type=str,
        default="data/channel_generality/dt_v2_tiny/eval_ood_100ep/b2_e2_5seed100/metric_raw/scenario_9/ep_seed0.npz",
    )
    parser.add_argument(
        "--lag_json",
        type=str,
        default="data/channel_generality/ppo_lagrangian_v2/metric_json/scenario_9/seed_0/nhp_violations.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/figures/channel_generality",
    )
    args = parser.parse_args()

    rr = repo_root()
    dt_path = rr / args.dt_npz
    lag_path = rr / args.lag_json
    out_dir = (rr / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dt_npz = np.load(dt_path)
    if "nhp_viols" not in dt_npz.files:
        raise KeyError(f"'nhp_viols' not in {dt_path}")
    dt_ep = np.asarray(dt_npz["nhp_viols"], dtype=float)

    with lag_path.open(encoding="utf-8") as f:
        lag = json.load(f)
    if "nhp_violations" not in lag:
        raise KeyError(f"'nhp_violations' not in {lag_path}")
    lag_ep = np.asarray(lag["nhp_violations"], dtype=float)

    bins = np.linspace(0.0, 0.6, 30)
    dt_st = _stats(dt_ep)
    lag_st = _stats(lag_ep)

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)

    # Left: IDT
    ax = axes[0]
    ax.hist(dt_ep, bins=bins, color="#4C72B0", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axvspan(0.3, 0.6, color="red", alpha=0.08)
    ax.axvline(0.3, color="red", linestyle="--", linewidth=1.0)
    ax.set_title("IDT (Scenario E)")
    ax.set_xlabel("nhp_violations")
    ax.set_ylabel("episode count")
    ax.grid(alpha=0.2)
    ax.text(
        0.98,
        0.95,
        _textbox_text("IDT", dt_st),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    # Right: PPO-Lagrangian-Discrete
    ax = axes[1]
    ax.hist(lag_ep, bins=bins, color="#C44E52", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axvspan(0.3, 0.6, color="red", alpha=0.08)
    ax.axvline(0.3, color="red", linestyle="--", linewidth=1.0)
    ax.set_title("PPO-Lagrangian-Discrete (Scenario E)")
    ax.set_xlabel("nhp_violations")
    ax.grid(alpha=0.2)
    ax.text(
        0.98,
        0.95,
        _textbox_text("PPO-Lagrangian-Discrete", lag_st),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    fig.tight_layout()
    out_pdf = out_dir / "lagrangian_degradation_s9.pdf"
    out_png = out_dir / "lagrangian_degradation_s9.png"
    fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
    fig.savefig(out_png, bbox_inches="tight", format="png", dpi=320)
    plt.close(fig)

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    print(
        f"[stats] IDT mean={dt_st['mean']:.3f}, std={dt_st['std']:.3f}, p95={dt_st['p95']:.3f}, crash={dt_st['crash']}/{dt_st['n']}"
    )
    print(
        f"[stats] PPO-Lagrangian-Discrete mean={lag_st['mean']:.3f}, std={lag_st['std']:.3f}, p95={lag_st['p95']:.3f}, crash={lag_st['crash']}/{lag_st['n']}"
    )


if __name__ == "__main__":
    main()

