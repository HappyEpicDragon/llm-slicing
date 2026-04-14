"""
Check whether per-scenario PPO expert policies are constant (output same action
regardless of state).  Also prints a brief action diversity summary.

Usage:
    cd /root/decision_transformer_slicing
    pixi run python -u scripts/check_constant_policy_v2.py --n_steps 200
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.basic_apis.dt_v2.env_v2 import HierarchicalSlicingEnvV2
from src.basic_apis.network_slicing_business.path_manager import PathManager

# action_dims for v2: codebook(11) + ordering(5x5) + intra(5x3) = 51 total
ACTION_DIMS = [11] + [5] * 5 + [3] * 5
ACTION_NAMES = (
    ["codebook"]
    + [f"order_s{i}" for i in range(5)]
    + [f"intra_s{i}" for i in range(5)]
)


def run_steps(env, model, n_steps):
    """Collect n_steps worth of actions from one or more episodes."""
    obs = env.reset()
    actions_list = []
    step = 0
    done = False
    while step < n_steps:
        action, _ = model.predict(obs, deterministic=True)
        actions_list.append(action[0].copy())  # shape (51,) or flat int array
        obs, _, dones, _ = env.step(action)
        step += 1
        if dones[0]:
            obs = env.reset()
            done = True  # episode boundary (doesn't stop collection)
    return np.array(actions_list)  # (n_steps, n_action_parts)


def analyse_actions(actions_arr, action_dims, action_names):
    """
    actions_arr: (T, n_parts) where each value is an integer index.
    Returns dict with per-part unique count and overall is_constant flag.
    """
    results = {}
    is_constant_overall = True
    ptr = 0
    for dim, name in zip(action_dims, action_names):
        col = actions_arr[:, ptr] if actions_arr.ndim == 2 else actions_arr
        ptr += 1  # actions are already split per part after predict
        unique_vals = np.unique(col)
        n_unique = len(unique_vals)
        counter = Counter(col.tolist())
        most_common_val, most_common_cnt = counter.most_common(1)[0]
        freq = most_common_cnt / len(col)
        is_constant = n_unique == 1
        if not is_constant:
            is_constant_overall = False
        results[name] = {
            "n_unique": n_unique,
            "most_common": int(most_common_val),
            "most_common_freq": float(freq),
            "is_constant": is_constant,
        }
    return results, is_constant_overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="data/channel_generality/dt_v2_per_scenario",
        help="Directory containing ppo_s{sid}/best_model/best_model.zip",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        nargs="+",
        default=None,
        help="Default: 0 1 2 3 4 5 6 8",
    )
    parser.add_argument("--n_steps", type=int, default=300,
                        help="Steps per scenario to collect for diversity check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.scenarios is None:
        args.scenarios = [0, 1, 2, 3, 4, 5, 6, 8]

    env_cfg = OmegaConf.load(os.path.join(ROOT, "conf/environment/env_ha.yaml"))
    pm = PathManager(ROOT)

    print(
        f"Constant-policy check: n_steps={args.n_steps}, seed={args.seed}\n"
        f"{'Scenario':<10} {'Constant?':<12} {'Codebook unique':<18} "
        f"{'OrderS0 unique':<17} {'IntraS0 unique':<16}",
        flush=True,
    )
    print("-" * 80, flush=True)

    summary_rows = []
    for scen in args.scenarios:
        model_path = os.path.join(
            ROOT, args.root, f"ppo_s{scen}", "best_model", "best_model.zip"
        )
        if not os.path.isfile(model_path):
            print(f"S{scen}: SKIP (missing {model_path})", flush=True)
            continue

        model = PPO.load(model_path, device=args.device)

        cfg = env_cfg.env_settings.copy()
        cfg.mode = "testing"
        cfg.scenario_mode = "inside"
        cfg.inside.testing.active_scenario_list = [scen]
        cfg.inside.testing.init_scenario_episode = 0
        cfg.inside.testing.max_scenario_episodes = 20

        def _make(c=cfg, s=args.seed):
            def _init():
                return HierarchicalSlicingEnvV2(c, np.random.default_rng(s), pm)
            return _init

        env = DummyVecEnv([_make()])
        try:
            actions = run_steps(env, model, args.n_steps)
        finally:
            env.close()

        # actions shape: (n_steps, 51) - but SB3 predict returns flat action
        # Need to split by action_dims
        if actions.ndim == 1:
            # scalar actions - shouldn't happen for MultiDiscrete
            print(f"S{scen}: unexpected 1-D action array", flush=True)
            continue

        # split columns by action_dims (cumulative)
        splits = np.split(actions, np.cumsum(ACTION_DIMS[:-1]), axis=1)
        # Each split is (n_steps, dim) - take argmax if needed, or just use raw
        # For MultiDiscrete envs, SB3 returns one integer per sub-action directly
        # So actions[:, i] is the i-th sub-action
        per_part = {}
        for i, (dim, name) in enumerate(zip(ACTION_DIMS, ACTION_NAMES)):
            col = actions[:, i]
            unique_vals = np.unique(col)
            counter = Counter(col.tolist())
            most_common_val, most_common_cnt = counter.most_common(1)[0]
            freq = most_common_cnt / len(col)
            per_part[name] = {
                "n_unique": len(unique_vals),
                "most_common": int(most_common_val),
                "most_common_freq": float(freq),
            }

        is_constant = all(v["n_unique"] == 1 for v in per_part.values())
        codebook_u = per_part["codebook"]["n_unique"]
        order_s0_u = per_part["order_s0"]["n_unique"]
        intra_s0_u = per_part["intra_s0"]["n_unique"]

        flag = "YES ⚠" if is_constant else "no"
        print(
            f"S{scen:<9} {flag:<12} {codebook_u:<18} {order_s0_u:<17} {intra_s0_u:<16}",
            flush=True,
        )

        # Detailed printout if constant
        if is_constant:
            print(f"  → ALL actions constant! Values: {[per_part[n]['most_common'] for n in ACTION_NAMES]}", flush=True)

        # Print any sub-action that IS constant
        const_parts = [n for n, v in per_part.items() if v["n_unique"] == 1]
        if const_parts and not is_constant:
            for cp in const_parts:
                print(
                    f"  → {cp} is constant at {per_part[cp]['most_common']} "
                    f"(all {args.n_steps} steps)",
                    flush=True,
                )

        summary_rows.append((scen, is_constant, per_part))

    print("-" * 80, flush=True)
    n_const = sum(1 for _, is_c, _ in summary_rows if is_c)
    print(
        f"\nSummary: {n_const}/{len(summary_rows)} scenario experts have ALL-constant policies",
        flush=True,
    )
    if n_const > 0:
        const_scens = [scen for scen, is_c, _ in summary_rows if is_c]
        print(f"  Constant policy scenarios: {const_scens}", flush=True)


if __name__ == "__main__":
    main()
