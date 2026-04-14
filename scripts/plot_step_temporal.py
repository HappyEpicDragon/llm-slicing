#!/usr/bin/env python3
"""Phase 3 Fig.B: scenario-C temporal curves (M1-M4 only)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCENARIO = 7
SCENARIO_TITLE = "Scenario C"
STEPS = 1000

METHODS = ["M1", "M2", "M3", "M4"]
METHOD_LABEL = {
    "M1": "IDT",
    "M2": "DT-Mix",
    "M3": "PPO-Discrete",
    "M4": "PPO-baseline",
}
METHOD_RAW_ROOT = {
    "M1": "data/channel_generality/dt_v2_tiny/eval_ood_100ep/b2_e2_5seed100/metric_raw",
    "M2": "data/channel_generality/dt_baseline_v2/eval_ood/metric_raw",
    "M3": "data/channel_generality/ppo_teacher_5seed/eval_ood/metric_raw",
    "M4": "data/channel_generality/ppo_multi/metric_raw",
}

COLORS = {
    "M1": "#4C72B0",
    "M2": "#DD8452",
    "M3": "#55A868",
    "M4": "#C44E52",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _to_ep_step(arr: np.ndarray, n_steps: int = STEPS) -> np.ndarray:
    """Normalize raw array into shape [episodes, steps]."""
    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim != 1:
        raise ValueError(f"Unexpected ndim={a.ndim} for temporal series.")
    if a.size % n_steps != 0:
        raise ValueError(f"1D array size {a.size} cannot reshape to (?, {n_steps}).")
    return a.reshape(-1, n_steps)


def load_step_curve(npz_path: Path, metric: str, episode_index: int = 0) -> np.ndarray:
    d = np.load(npz_path)
    if metric not in d.files:
        raise KeyError(f"{metric} not found in {npz_path}")
    mat = _to_ep_step(d[metric], n_steps=STEPS)
    if episode_index < 0 or episode_index >= mat.shape[0]:
        raise IndexError(f"episode_index={episode_index} out of range [0,{mat.shape[0]-1}] for {npz_path}")
    return mat[episode_index].astype(float)


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    window = min(window, y.size)
    if window <= 1:
        return y
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def draw(out_prefix: str, episode_index: int = 0, step_window: int = 31, viol_window: int = 31) -> None:
    rr = repo_root()
    out_dir = rr / "outputs/figures/channel_generality"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )

    # Use a column-friendly canvas width to avoid aggressive shrinking in two-column papers.
    fig, (ax_dist, ax_viol) = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=False)

    x_step = np.arange(STEPS)

    for m in METHODS:
        p = rr / METHOD_RAW_ROOT[m] / f"scenario_{SCENARIO}" / "ep_seed0.npz"
        if not p.is_file():
            print(f"[warn] missing {p}")
            continue

        try:
            y_step = load_step_curve(p, "step_nhp_dist", episode_index=episode_index)
            y_step = moving_average(y_step, step_window)
            ax_dist.plot(x_step, y_step, label=METHOD_LABEL[m], color=COLORS[m], linewidth=2.0)
        except Exception as e:
            print(f"[warn] failed loading step_nhp_dist from {p}: {e}")

        try:
            y_viol = load_step_curve(p, "step_nhp_dist", episode_index=episode_index)
            # Step-level violation proxy: drift < 0 means SLA violation at this step.
            y_viol = (y_viol < 0.0).astype(float)
            y_viol = moving_average(y_viol, viol_window)
            ax_viol.plot(x_step, y_viol, label=METHOD_LABEL[m], color=COLORS[m], linewidth=2.0)
        except Exception as e:
            print(f"[warn] failed building step-level violations from {p}: {e}")

    ax_dist.set_title(f"{SCENARIO_TITLE}: NHP Distance")
    ax_dist.set_xlabel("Step")
    ax_dist.set_ylabel("NHP Distance")
    ax_dist.grid(alpha=0.25)
    ax_dist.set_xlim(0, STEPS - 1)
    ax_dist.margins(x=0)

    ax_viol.set_title(f"{SCENARIO_TITLE}: NHP Violations")
    ax_viol.set_xlabel("Step")
    ax_viol.set_ylabel("NHP Violation Rate")
    ax_viol.grid(alpha=0.25)
    ax_viol.set_xlim(0, STEPS - 1)
    ax_viol.set_ylim(-0.02, 1.02)
    ax_viol.margins(x=0)

    handles, labels = ax_dist.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_pdf = out_dir / f"{out_prefix}.pdf"
    out_png = out_dir / f"{out_prefix}.png"
    fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
    fig.savefig(out_png, bbox_inches="tight", format="png", dpi=320)
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode_index",
        type=int,
        default=0,
        help="Which episode index to plot for step-level NHP distance.",
    )
    parser.add_argument(
        "--step_window",
        type=int,
        default=31,
        help="Moving-average window for step-level NHP distance.",
    )
    parser.add_argument(
        "--viol_window",
        type=int,
        default=31,
        help="Moving-average window for step-level NHP violations.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="step_temporal_nhp_dist",
        help="Output file prefix under outputs/figures/channel_generality.",
    )
    args = parser.parse_args()

    draw(
        out_prefix=args.out_prefix,
        episode_index=args.episode_index,
        step_window=args.step_window,
        viol_window=args.viol_window,
    )


if __name__ == "__main__":
    main()

