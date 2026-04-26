"""Evaluate PPO teacher policies on the DT environment."""

import json
import os
import shutil

import numpy as np
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.basic_apis.dt.env import HierarchicalSlicingEnvV2
from src.basic_apis.ppo.ppo_ha_weighted.agent_hierarchical import (
    HierarchicalMLPPolicy,
    HierarchicalPooledAttentionPolicy,
    HierarchicalSliceAttnPolicy,
    HierarchicalSmartPolicy,
)


def _save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _compute_step_metrics(info, num_slices=5):
    hp_dist = nhp_dist = 0.0
    hp_viols = nhp_viols = hp_active = nhp_active = 0
    for s_idx in range(num_slices):
        is_hp = info.get(f"meta/slice_{s_idx}_priority", 0) > 0
        drifts = [
            float(info.get(f"drift/slice_{s_idx}_{metric}", 0.0))
            for metric in ("thr", "rel", "lat")
            if info.get(f"meta/slice_{s_idx}_{metric}_req", 0) > 0
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
                hp_dist += mean_drift
                hp_viols += 1
            else:
                nhp_dist += mean_drift
                nhp_viols += 1

    return hp_dist, nhp_dist, hp_viols, nhp_viols, hp_active, nhp_active


def _make_env(env_settings, paths_cfg, workdir, seed):
    def _init():
        return HierarchicalSlicingEnvV2(
            env_settings,
            np.random.default_rng(seed),
            paths_cfg=paths_cfg,
            workdir=workdir,
        )

    return _init


def _load_teacher(model_path, env, device):
    custom_objects = {
        "HierarchicalSmartPolicy": HierarchicalSmartPolicy,
        "HierarchicalMLPPolicy": HierarchicalMLPPolicy,
        "HierarchicalPooledAttentionPolicy": HierarchicalPooledAttentionPolicy,
        "HierarchicalSliceAttnPolicy": HierarchicalSliceAttnPolicy,
        "learning_rate": 0.0,
        "clip_range": 0.1,
    }
    return PPO.load(model_path, env=env, device=device, custom_objects=custom_objects)


def test_dt_teacher(cfg, paths_cfg=None, workdir=None):
    tc = cfg.test_dt_teacher
    paths_cfg = paths_cfg if paths_cfg is not None else cfg.get("paths", None)
    workdir = workdir if workdir is not None else str(cfg.get("workdir", "."))

    model_path = str(tc.model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DT teacher PPO model not found: {model_path}")

    save_root = str(tc.save_root)
    clean_before_save = bool(tc.get("clean_before_save", False))
    if clean_before_save and os.path.exists(save_root):
        shutil.rmtree(save_root)

    device = str(tc.get("device", "cpu"))
    env_cfg_raw = OmegaConf.load("conf/environment/env_ha.yaml")
    scenario_mode = str(tc.env_updates.scenario_mode)
    mode = str(tc.env_updates.mode)

    all_results = {}
    for scenario_id in list(tc.test_scenarios):
        scen_hp, scen_nhp, scen_hp_d, scen_nhp_d = [], [], [], []
        for seed in list(tc.test_seeds):
            env_settings = env_cfg_raw.env_settings.copy()
            env_settings.mode = mode
            env_settings.scenario_mode = scenario_mode
            env_settings[scenario_mode][mode].active_scenario_list = [scenario_id]
            if tc.get("init_episode") is not None:
                env_settings[scenario_mode][mode].init_scenario_episode = int(tc.init_episode)
            if tc.get("max_episode") is not None:
                env_settings[scenario_mode][mode].max_scenario_episodes = int(tc.max_episode)

            env = DummyVecEnv([_make_env(env_settings, paths_cfg, workdir, int(seed))])
            model = _load_teacher(model_path, env, device)

            ep_rewards = []
            seed_hp_viols, seed_nhp_viols = [], []
            step_hp_dists, step_nhp_dists = [], []
            for _ in range(int(tc.n_episodes)):
                obs = env.reset()
                done = False
                ep_reward = 0.0
                hp_v = nhp_v = hp_a = nhp_a = 0
                hp_d = nhp_d = 0.0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, rewards, dones, infos = env.step(action)
                    info = infos[0]
                    ep_reward += float(rewards[0])
                    sd_hp, sd_nhp, sv_hp, sv_nhp, sa_hp, sa_nhp = _compute_step_metrics(info)
                    hp_d += sd_hp
                    nhp_d += sd_nhp
                    hp_v += sv_hp
                    nhp_v += sv_nhp
                    hp_a += sa_hp
                    nhp_a += sa_nhp
                    step_hp_dists.append(sd_hp / sa_hp if sa_hp > 0 else 0.0)
                    step_nhp_dists.append(sd_nhp / sa_nhp if sa_nhp > 0 else 0.0)
                    done = bool(dones[0])

                ep_rewards.append(ep_reward)
                seed_hp_viols.append(hp_v / hp_a if hp_a > 0 else 0.0)
                seed_nhp_viols.append(nhp_v / nhp_a if nhp_a > 0 else 0.0)
                scen_hp.extend(seed_hp_viols[-1:])
                scen_nhp.extend(seed_nhp_viols[-1:])
                scen_hp_d.append(hp_d / hp_a if hp_a > 0 else 0.0)
                scen_nhp_d.append(nhp_d / nhp_a if nhp_a > 0 else 0.0)

            raw_dir = os.path.join(save_root, "metric_raw", f"scenario_{scenario_id}")
            os.makedirs(raw_dir, exist_ok=True)
            np.savez(
                os.path.join(raw_dir, f"ep_seed{seed}.npz"),
                ep_rewards=np.array(ep_rewards, dtype=np.float64),
                hp_viols=np.array(seed_hp_viols, dtype=np.float64),
                nhp_viols=np.array(seed_nhp_viols, dtype=np.float64),
                step_hp_dist=np.array(step_hp_dists, dtype=np.float64),
                step_nhp_dist=np.array(step_nhp_dists, dtype=np.float64),
            )
            env.close()

        summary = {
            "hp_viol_mean": float(np.mean(scen_hp)),
            "nhp_viol_mean": float(np.mean(scen_nhp)),
            "hp_dist_mean": float(np.mean(scen_hp_d)),
            "nhp_dist_mean": float(np.mean(scen_nhp_d)),
            "combined": float(np.mean(scen_hp) + np.mean(scen_nhp)),
        }
        all_results[int(scenario_id)] = summary
        _save_json(summary, os.path.join(save_root, f"scenario_{scenario_id}", "summary.json"))

    total = sum(r["combined"] for r in all_results.values())
    _save_json(
        {"per_scenario": {str(k): v for k, v in all_results.items()}, "total": total},
        os.path.join(save_root, "global_summary.json"),
    )
    print(f"[test_dt_teacher] TOTAL combined: {total:.4f}")
