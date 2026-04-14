"""
Evaluate the joint PPO teacher (dt_v2_split_b_norm/ppo_teacher/best_model)
on each training scenario (S0-6, S8) individually.
Produces hp_viol / nhp_viol / comb metrics for comparison with per-scenario experts.

Usage:
    cd /root/decision_transformer_slicing
    pixi run python -u scripts/eval_joint_ppo_per_scenario.py --n_episodes 20
"""
import argparse
import os
import sys

import numpy as np
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.basic_apis.dt_v2.env_v2 import HierarchicalSlicingEnvV2
from src.basic_apis.network_slicing_business.path_manager import PathManager

NUM_SLICES = 5


def _compute_step_metrics(info, num_slices=NUM_SLICES):
    hp_dist = nhp_dist = 0.0
    hp_viols = nhp_viols = hp_active = nhp_active = 0
    for s_idx in range(num_slices):
        is_hp = info.get(f"meta/slice_{s_idx}_priority", 0) > 0
        drifts = [
            float(info.get(f"drift/slice_{s_idx}_{met}", 0.0))
            for met in ("thr", "rel", "lat")
            if info.get(f"meta/slice_{s_idx}_{met}_req", 0) > 0
        ]
        if not drifts:
            continue
        if is_hp:
            hp_active += 1
        else:
            nhp_active += 1
        mean_drift = float(np.mean(drifts))
        if mean_drift < 0:
            if is_hp:
                hp_viols += 1
                hp_dist += mean_drift
            else:
                nhp_viols += 1
                nhp_dist += mean_drift
    return hp_dist, nhp_dist, hp_viols, nhp_viols, hp_active, nhp_active


def run_ppo_episode(env, model):
    obs = env.reset()
    hp_viol_sum = nhp_viol_sum = 0
    hp_active_t = nhp_active_t = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        next_obs, _rewards, dones, infos = env.step(action)
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        _, _, hv, nv, ha, na = _compute_step_metrics(info)
        hp_viol_sum += hv
        nhp_viol_sum += nv
        hp_active_t += ha
        nhp_active_t += na
        obs = next_obs
        if dones[0]:
            done = True
    hp_v = hp_viol_sum / hp_active_t if hp_active_t > 0 else 0.0
    nhp_v = nhp_viol_sum / nhp_active_t if nhp_active_t > 0 else 0.0
    return hp_v, nhp_v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="data/channel_generality/dt_v2_split_b_norm/ppo_teacher/best_model/best_model.zip",
        help="Path to the joint PPO teacher model",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        nargs="+",
        default=None,
        help="Default: 0 1 2 3 4 5 6 8 (Split B train scenarios)",
    )
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.scenarios is None:
        args.scenarios = [0, 1, 2, 3, 4, 5, 6, 8]

    model_path = os.path.join(ROOT, args.model_path)
    if not os.path.isfile(model_path):
        print(f"ERROR: model not found at {model_path}", flush=True)
        sys.exit(1)

    model = PPO.load(model_path, device=args.device)
    print(f"Joint PPO loaded from {model_path}", flush=True)
    print(
        f"Eval: n_episodes={args.n_episodes}, seeds={args.seeds}, device={args.device}",
        flush=True,
    )

    env_cfg = OmegaConf.load(os.path.join(ROOT, "conf/environment/env_ha.yaml"))
    pm = PathManager(ROOT)

    rows = []
    for scen in args.scenarios:
        scen_hp, scen_nhp = [], []

        for seed in args.seeds:
            cfg = env_cfg.env_settings.copy()
            cfg.mode = "testing"
            cfg.scenario_mode = "inside"
            cfg.inside.testing.active_scenario_list = [scen]
            cfg.inside.testing.init_scenario_episode = 0
            cfg.inside.testing.max_scenario_episodes = args.n_episodes

            def _make(c=cfg, s=seed):
                def _init():
                    return HierarchicalSlicingEnvV2(c, np.random.default_rng(s), pm)
                return _init

            env = DummyVecEnv([_make()])
            try:
                for _ep in range(args.n_episodes):
                    hp_v, nhp_v = run_ppo_episode(env, model)
                    scen_hp.append(hp_v)
                    scen_nhp.append(nhp_v)
            finally:
                env.close()

        comb = float(np.mean(scen_hp) + np.mean(scen_nhp))
        rows.append((scen, float(np.mean(scen_hp)), float(np.mean(scen_nhp)), comb))
        print(
            f"S{scen}: hp={np.mean(scen_hp):.4f} nhp={np.mean(scen_nhp):.4f} comb={comb:.4f}",
            flush=True,
        )

    if rows:
        mean_comb = float(np.mean([r[3] for r in rows]))
        print(f"\nMean comb over {len(rows)} scenarios: {mean_comb:.4f}", flush=True)


if __name__ == "__main__":
    main()
