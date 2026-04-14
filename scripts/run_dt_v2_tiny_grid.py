"""
Run DT-v2 tiny 3x3 grid:
  backbone: (64,2,2), (32,2,2), (24,2,2)
  encoder hidden: 64, 32, 16

This script launches train + test sequentially per grid, and assigns grids to GPUs.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


BACKBONE_GRID: Dict[str, Tuple[int, int, int]] = {
    "b1": (64, 2, 2),
    "b2": (32, 2, 2),
    "b3": (24, 2, 2),
}
ENCODER_GRID: Dict[str, int] = {
    "e1": 64,
    "e2": 32,
    "e3": 16,
}


def run_cmd(cmd: List[str]) -> None:
    print("\n[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_overrides(grid_id: str, embed_dim: int, n_layer: int, n_head: int, enc_hidden: int, device: str):
    train_overrides = [
        f"train_dt_v2_tiny.grid_id={grid_id}",
        f"train_dt_v2_tiny.model.embed_dim={embed_dim}",
        f"train_dt_v2_tiny.model.n_layer={n_layer}",
        f"train_dt_v2_tiny.model.n_head={n_head}",
        "train_dt_v2_tiny.model.encoder_type=slice_attn",
        f"train_dt_v2_tiny.model.encoder_hidden_dim={enc_hidden}",
        "train_dt_v2_tiny.model.encoder_num_heads=2",
        f"train_dt_v2_tiny.device={device}",
    ]

    model_path = f"data/channel_generality/dt_v2_tiny/models/{grid_id}/final_dt_v2.pth"
    test_overrides = [
        f"test_dt_v2_tiny.grid_id={grid_id}",
        f"test_dt_v2_tiny.model_path={model_path}",
        f"test_dt_v2_tiny.model.embed_dim={embed_dim}",
        f"test_dt_v2_tiny.model.n_layer={n_layer}",
        f"test_dt_v2_tiny.model.n_head={n_head}",
        "test_dt_v2_tiny.encoder_type=slice_attn",
        f"test_dt_v2_tiny.model.encoder_hidden_dim={enc_hidden}",
        "test_dt_v2_tiny.model.encoder_num_heads=2",
        f"test_dt_v2_tiny.device={device}",
    ]
    return train_overrides, test_overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixi", default="/root/.pixi/bin/pixi")
    parser.add_argument("--workspace", default="/root/decision_transformer_slicing")
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--only", nargs="*", default=None, help="optional subset of grid ids, e.g. b2_e2")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    grids = []
    for b_key, e_key in itertools.product(("b1", "b2", "b3"), ("e1", "e2", "e3")):
        grid_id = f"{b_key}_{e_key}"
        d, l, h = BACKBONE_GRID[b_key]
        eh = ENCODER_GRID[e_key]
        grids.append((grid_id, d, l, h, eh))
    if args.only:
        wanted = set(args.only)
        grids = [g for g in grids if g[0] in wanted]

    assignments = {gpu: [] for gpu in args.gpus}
    for idx, g in enumerate(grids):
        gpu = args.gpus[idx % len(args.gpus)]
        assignments[gpu].append(g)

    assign_path = workspace / "data/channel_generality/dt_v2_tiny/outputs"
    assign_path.mkdir(parents=True, exist_ok=True)
    with (assign_path / "grid_assignments.json").open("w") as f:
        json.dump({k: [x[0] for x in v] for k, v in assignments.items()}, f, indent=2)

    for gpu, jobs in assignments.items():
        for grid_id, d, l, h, eh in jobs:
            device = f"cuda:{gpu}"
            train_ovr, test_ovr = build_overrides(grid_id, d, l, h, eh, device)

            run_cmd(
                [args.pixi, "run", "sim", "train_dt_v2_tiny", *train_ovr]
            )
            run_cmd(
                [args.pixi, "run", "sim", "test_dt_v2_tiny", *test_ovr]
            )


if __name__ == "__main__":
    main()
