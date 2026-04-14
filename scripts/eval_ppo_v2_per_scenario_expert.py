"""
Evaluate each per-scenario PPO expert's best_model.zip on its own scenario.
Metrics match src.basic_apis.dt_v2.test_v2 (hp_viol_mean, nhp_viol_mean, combined).
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
        "--root",
        type=str,
        default="data/channel_generality/dt_v2_per_scenario",
        help="Directory containing ppo_s{sid}[suffix]/best_model/best_model.zip",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        nargs="+",
        default=None,
        help="Default: 0 1 2 3 4 5 6 8 (Split B train scenarios)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Folder suffix, e.g. '_slice_attn' → ppo_s{sid}_slice_attn/",
    )
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.scenarios is None:
        args.scenarios = [0, 1, 2, 3, 4, 5, 6, 8]

    env_cfg = OmegaConf.load(os.path.join(ROOT, "conf/environment/env_ha.yaml"))
    pm = PathManager(ROOT)

    print(
        f"Per-scenario expert eval: n_episodes={args.n_episodes}, seeds={args.seeds}, device={args.device}",
        flush=True,
    )
    rows = []
    for scen in args.scenarios:
        model_path = os.path.join(
            ROOT, args.root, f"ppo_s{scen}{args.suffix}", "best_model", "best_model.zip"
        )
        if not os.path.isfile(model_path):
            print(f"S{scen}: SKIP (missing {model_path})", flush=True)
            continue

        model = PPO.load(model_path, device=args.device)
        scen_hp, scen_nhp = [], []

        for seed in args.seeds:
            cfg = env_cfg.env_settings.copy()
            cfg.mode = "testing"
            cfg.scenario_mode = "inside"
            cfg.inside.testing.active_scenario_list = [scen]
            # Use YAML defaults: init=60, max=80 → held-out ep 60-79
            # Do NOT override init_scenario_episode; that was data leakage.

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
        rows.append((scen, np.mean(scen_hp), np.mean(scen_nhp), comb))
        print(
            f"S{scen}: hp={np.mean(scen_hp):.4f} nhp={np.mean(scen_nhp):.4f} comb={comb:.4f}  ({model_path})",
            flush=True,
        )

    if rows:
        mean_comb = float(np.mean([r[3] for r in rows]))
        print(f"\nMean comb over {len(rows)} experts: {mean_comb:.4f}", flush=True)


def test_ppo_v2(cfg, path_manager):
    """Hydra entry point: called by channel_generality.py -> test_ppo_v2 mode.

    Supports init_episode/max_episode overrides for OOD testing and
    saves step-level npz data alongside summary JSON.
    """
    import json
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    tc = cfg.test_ppo_v2
    env_cfg = OmegaConf.load("conf/environment/env_ha.yaml")
    pm = path_manager

    model_path = str(tc.model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"[test_ppo_v2] model not found: {model_path}")

    model = PPO.load(model_path, device=str(tc.get("device", "cuda")))

    init_ep = getattr(tc, "init_episode", None)
    max_ep = getattr(tc, "max_episode", None)

    rows = []
    for scen in list(tc.test_scenarios):
        scen_hp, scen_nhp, scen_hp_d, scen_nhp_d = [], [], [], []

        for seed in list(tc.test_seeds):
            env_settings = env_cfg.env_settings.copy()
            env_settings.mode = "testing"
            env_settings.scenario_mode = "inside"
            env_settings.inside.testing.active_scenario_list = [scen]

            if init_ep is not None:
                env_settings.inside.testing.init_scenario_episode = int(init_ep)
            if max_ep is not None:
                env_settings.inside.testing.max_scenario_episodes = int(max_ep)

            def _make(s=env_settings, sd=seed):
                def _init():
                    return HierarchicalSlicingEnvV2(s, np.random.default_rng(sd), pm)
                return _init

            env = DummyVecEnv([_make()])
            seed_hp_viols, seed_nhp_viols = [], []
            seed_step_hp, seed_step_nhp = [], []
            seed_rewards = []
            try:
                for _ in range(int(tc.n_episodes)):
                    hd, nd, hv, nv, ha, na = 0.0, 0.0, 0, 0, 0, 0
                    obs = env.reset()
                    done = False
                    ep_reward = 0.0
                    step_hp, step_nhp = [], []
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        next_obs, rews, dones, infos = env.step(action)
                        info = infos[0] if isinstance(infos, (list, tuple)) else infos
                        ep_reward += float(rews[0])
                        _hd, _nd, _hv, _nv, _ha, _na = _compute_step_metrics(info)
                        hd += _hd; nd += _nd; hv += _hv; nv += _nv; ha += _ha; na += _na
                        step_hp.append(_hd / _ha if _ha > 0 else 0.0)
                        step_nhp.append(_nd / _na if _na > 0 else 0.0)
                        obs = next_obs
                        if dones[0]:
                            done = True
                    scen_hp.append(hv / ha if ha > 0 else 0.0)
                    scen_nhp.append(nv / na if na > 0 else 0.0)
                    scen_hp_d.append(hd / ha if ha > 0 else 0.0)
                    scen_nhp_d.append(nd / na if na > 0 else 0.0)
                    seed_hp_viols.append(hv / ha if ha > 0 else 0.0)
                    seed_nhp_viols.append(nv / na if na > 0 else 0.0)
                    seed_step_hp.append(step_hp)
                    seed_step_nhp.append(step_nhp)
                    seed_rewards.append(ep_reward)
            finally:
                env.close()

            raw_dir = os.path.join(str(tc.save_root), str(tc.save_metric_key),
                                   "metric_raw", f"scenario_{scen}")
            os.makedirs(raw_dir, exist_ok=True)
            max_steps = max((len(s) for s in seed_step_hp), default=0)
            def _pad(lst, length):
                arr = np.array(lst, dtype=np.float64)
                if len(arr) < length:
                    arr = np.concatenate([arr, np.zeros(length - len(arr))])
                return arr
            np.savez(
                os.path.join(raw_dir, f"ep_seed{seed}.npz"),
                ep_rewards=np.array(seed_rewards, dtype=np.float64),
                hp_viols=np.array(seed_hp_viols, dtype=np.float64),
                nhp_viols=np.array(seed_nhp_viols, dtype=np.float64),
                step_hp_dist=np.array([_pad(s, max_steps) for s in seed_step_hp], dtype=np.float64),
                step_nhp_dist=np.array([_pad(s, max_steps) for s in seed_step_nhp], dtype=np.float64),
            )

        summary = {
            "hp_viol_mean": float(np.mean(scen_hp)),
            "nhp_viol_mean": float(np.mean(scen_nhp)),
            "hp_dist_mean": float(np.mean(scen_hp_d)),
            "nhp_dist_mean": float(np.mean(scen_nhp_d)),
            "combined": float(np.mean(scen_hp) + np.mean(scen_nhp)),
        }
        rows.append((scen, summary))
        print(f"S{scen}: hp={summary['hp_viol_mean']:.4f} nhp={summary['nhp_viol_mean']:.4f} "
              f"comb={summary['combined']:.4f}", flush=True)
        save_path = os.path.join(str(tc.save_root), str(tc.save_metric_key),
                                 f"scenario_{scen}", "summary.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2)

    mean_comb = float(np.mean([r[1]["combined"] for r in rows])) if rows else 0.0
    print(f"\n[test_ppo_v2] mean combined over {len(rows)} scenarios: {mean_comb:.4f}")


if __name__ == "__main__":
    main()
