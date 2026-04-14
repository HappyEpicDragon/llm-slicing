"""
Evaluate old ppo_ha_weighted per-scenario expert models on their own scenarios.
Uses HierarchicalSlicingEnv (not EnvV2).
Produces the same hp_viol / nhp_viol / comb metrics for comparison.

Usage:
    cd /root/decision_transformer_slicing
    pixi run python -u scripts/eval_ppo_ha_weighted_per_scenario.py --n_episodes 20
"""
import os
import sys
import argparse
import numpy as np
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.basic_apis.ppo.ppo_ha_weighted.hierarchical_slicing_env import HierarchicalSlicingEnv
from src.basic_apis.ppo.ppo_ha_weighted.agent_hierarchical import HierarchicalSmartPolicy
from src.basic_apis.network_slicing_business.path_manager import PathManager

NUM_SLICES = 5


def _compute_step_metrics(info):
    hp_viols = nhp_viols = hp_active = nhp_active = 0
    for s_idx in range(NUM_SLICES):
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
        if float(np.mean(drifts)) < 0:
            if is_hp:
                hp_viols += 1
            else:
                nhp_viols += 1
    return hp_viols, nhp_viols, hp_active, nhp_active


def run_episode(env, model):
    obs = env.reset()
    hp_viol_sum = nhp_viol_sum = 0
    hp_active_t = nhp_active_t = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _rewards, dones, infos = env.step(action)
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        hv, nv, ha, na = _compute_step_metrics(info)
        hp_viol_sum += hv
        nhp_viol_sum += nv
        hp_active_t += ha
        nhp_active_t += na
        if dones[0]:
            done = True
    hp_v = hp_viol_sum / hp_active_t if hp_active_t > 0 else 0.0
    nhp_v = nhp_viol_sum / nhp_active_t if nhp_active_t > 0 else 0.0
    return hp_v, nhp_v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="data/channel_generality/ppo_ha_weighted/models",
    )
    parser.add_argument(
        "--scenarios", type=int, nargs="+", default=None,
        help="Default: 0 1 2 3 4"
    )
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.scenarios is None:
        args.scenarios = [0, 1, 2, 3, 4]

    env_cfg = OmegaConf.load(os.path.join(ROOT, "conf/environment/env_ha.yaml"))
    pm = PathManager(ROOT)

    print(
        f"ppo_ha_weighted expert eval: n_episodes={args.n_episodes}, "
        f"seeds={args.seeds}, device={args.device}",
        flush=True,
    )
    rows = []
    for scen in args.scenarios:
        model_path = os.path.join(
            ROOT, args.root, f"scenario_{scen}", "latest", "best_model", "best_model.zip"
        )
        if not os.path.isfile(model_path):
            print(f"S{scen}: SKIP (missing {model_path})", flush=True)
            continue

        model = PPO.load(
            model_path, device=args.device,
            custom_objects={"learning_rate": 0.0, "clip_range": 0.1},
        )

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
                    return HierarchicalSlicingEnv(c, np.random.default_rng(s), pm)
                return _init

            env = DummyVecEnv([_make()])
            try:
                for _ep in range(args.n_episodes):
                    hp_v, nhp_v = run_episode(env, model)
                    scen_hp.append(hp_v)
                    scen_nhp.append(nhp_v)
            finally:
                env.close()

        comb = float(np.mean(scen_hp) + np.mean(scen_nhp))
        rows.append((scen, np.mean(scen_hp), np.mean(scen_nhp), comb))
        print(
            f"S{scen}: hp={np.mean(scen_hp):.4f} nhp={np.mean(scen_nhp):.4f} "
            f"comb={comb:.4f}  ({model_path})",
            flush=True,
        )

    if rows:
        mean_comb = float(np.mean([r[3] for r in rows]))
        print(f"\nMean comb over {len(rows)} old experts: {mean_comb:.4f}", flush=True)


if __name__ == "__main__":
    main()
