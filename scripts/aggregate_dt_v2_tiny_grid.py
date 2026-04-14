"""
Aggregate DT-v2 tiny grid outputs into one summary JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from omegaconf import OmegaConf

from src.basic_apis.dt_v2.model_v2 import build_dt_v2


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
ACTION_DIMS = [11] + [5] * 5 + [3] * 5


def count_params(grid_id: str) -> dict:
    b_key, e_key = grid_id.split("_")
    d, l, h = BACKBONE_GRID[b_key]
    eh = ENCODER_GRID[e_key]
    cfg = OmegaConf.create({
        "context_len": 20,
        "embed_dim": d,
        "n_layer": l,
        "n_head": h,
        "activation": "relu",
        "dropout": 0.1,
        "act_dim": sum(ACTION_DIMS),
    })
    model = build_dt_v2(
        cfg, ACTION_DIMS, inter_dim=8, intra_dim=7, embed_dim=d,
        encoder_type="slice_attn", encoder_hidden_dim=eh, encoder_num_heads=2,
    )
    total = sum(p.numel() for p in model.parameters())
    enc = sum(p.numel() for p in model.state_encoder.parameters())
    return {"total": total, "encoder": enc, "backbone": total - enc}


def read_metric(metric_root: Path, grid_id: str) -> dict:
    p = metric_root / grid_id / "global_summary.json"
    if not p.exists():
        return {"missing": True}
    data = json.loads(p.read_text())
    per = data.get("per_scenario", {})
    hp_viol = [per[k]["hp_viol_mean"] for k in sorted(per.keys())]
    nhp_viol = [per[k]["nhp_viol_mean"] for k in sorted(per.keys())]
    hp_dist = [per[k]["hp_dist_mean"] for k in sorted(per.keys())]
    nhp_dist = [per[k]["nhp_dist_mean"] for k in sorted(per.keys())]
    return {
        "missing": False,
        "total_combined": float(data.get("total", 0.0)),
        "mean_hp_viol": float(sum(hp_viol) / len(hp_viol)),
        "mean_nhp_viol": float(sum(nhp_viol) / len(nhp_viol)),
        "mean_hp_dist": float(sum(hp_dist) / len(hp_dist)),
        "mean_nhp_dist": float(sum(nhp_dist) / len(nhp_dist)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root", default="data/channel_generality/dt_v2_tiny/models")
    parser.add_argument("--metric_root", default="data/channel_generality/dt_v2_tiny/metric_json")
    parser.add_argument("--out", default="data/channel_generality/dt_v2_tiny/outputs/grid_summary.json")
    args = parser.parse_args()

    out = {}
    for b_key in ("b1", "b2", "b3"):
        for e_key in ("e1", "e2", "e3"):
            gid = f"{b_key}_{e_key}"
            params = count_params(gid)
            metric = read_metric(Path(args.metric_root), gid)
            model_ok = (Path(args.model_root) / gid / "final_dt_v2.pth").exists()
            out[gid] = {
                "backbone": {"embed_dim": BACKBONE_GRID[b_key][0], "n_layer": BACKBONE_GRID[b_key][1], "n_head": BACKBONE_GRID[b_key][2]},
                "slice_attn_hidden_dim": ENCODER_GRID[e_key],
                "params": params,
                "model_exists": model_ok,
                "metrics": metric,
            }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[aggregate] wrote {out_path}")


if __name__ == "__main__":
    main()
