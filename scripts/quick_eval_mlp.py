"""
Quick evaluation of MLP PPO model:
1. Feature extractor collapse check
2. hp/nhp/comb metrics (10 episodes)

Usage:
    cd /root/decision_transformer_slicing
    pixi run python -u scripts/quick_eval_mlp.py
"""
import os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.basic_apis.dt_v2.env_v2 import HierarchicalSlicingEnvV2
from src.basic_apis.network_slicing_business.path_manager import PathManager


def make_env(scenario_id, seed=0):
    env_cfg = OmegaConf.load(os.path.join(ROOT, "conf/environment/env_ha.yaml"))
    pm = PathManager(ROOT)
    cfg = env_cfg.env_settings.copy()
    cfg.mode = "testing"
    cfg.scenario_mode = "inside"
    cfg.inside.testing.active_scenario_list = [scenario_id]
    cfg.inside.testing.init_scenario_episode = 0
    cfg.inside.testing.max_scenario_episodes = 20
    return HierarchicalSlicingEnvV2(cfg, np.random.default_rng(seed), pm)


def eval_model(model_path, scenarios, n_episodes=10, seed=0, device="cuda"):
    model = PPO.load(model_path, device=device)
    fe = model.policy.features_extractor
    fe.eval()

    for scen in scenarios:
        env = make_env(scen, seed=seed)
        all_hp, all_nhp, all_comb = [], [], []
        all_features = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            ep_violations = {s: {"hp": [], "nhp": []} for s in range(5)}
            done = False

            while not done:
                obs_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device) for k, v in obs.items()}
                with torch.no_grad():
                    feat = fe(obs_t).cpu().numpy().squeeze(0)
                all_features.append(feat)

                obs_wrapped = {k: np.expand_dims(v, 0) for k, v in obs.items()}
                action, _ = model.predict(obs_wrapped, deterministic=True)
                action = action.squeeze(0)

                obs, r, done, _, info = env.step(action)

                for s in range(5):
                    prio = info.get(f"meta/slice_{s}_priority", 0)
                    active = info.get(f"meta/slice_{s}_active", 0)
                    if not active:
                        continue
                    for m in ["thr", "rel", "lat"]:
                        viol = info.get(f"violation/slice_{s}_{m}", 0.0)
                        if prio > 0:
                            ep_violations[s]["hp"].append(viol)
                        else:
                            ep_violations[s]["nhp"].append(viol)

            hp_viols = []
            nhp_viols = []
            for s in range(5):
                hp_viols.extend(ep_violations[s]["hp"])
                nhp_viols.extend(ep_violations[s]["nhp"])

            hp_rate = np.mean(hp_viols) if hp_viols else 0.0
            nhp_rate = np.mean(nhp_viols) if nhp_viols else 0.0
            comb = hp_rate + nhp_rate
            all_hp.append(hp_rate)
            all_nhp.append(nhp_rate)
            all_comb.append(comb)

        env.close()

        features_arr = np.array(all_features)
        norms = np.linalg.norm(features_arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normed = features_arr / norms
        n_sample = min(500, len(features_arr))
        idx = np.random.choice(len(features_arr), n_sample, replace=False)
        sub = normed[idx]
        cos_sim = sub @ sub.T
        triu = np.triu_indices(n_sample, k=1)
        mean_cos = np.mean(cos_sim[triu])
        frac_99 = np.mean(cos_sim[triu] > 0.99)

        print(f"  S{scen}: hp={np.mean(all_hp):.4f} nhp={np.mean(all_nhp):.4f} "
              f"comb={np.mean(all_comb):.4f} | "
              f"feat cos_sim={mean_cos:.6f} frac>0.99={frac_99:.4f}")


def main():
    print("=" * 70)
    print("MLP PPO (best_model)")
    print("=" * 70)
    mlp_path = os.path.join(ROOT, "data/channel_generality/dt_v2_mlp/ppo_s0/best_model/best_model.zip")
    if os.path.isfile(mlp_path):
        eval_model(mlp_path, scenarios=[0], n_episodes=10, seed=0)
    else:
        print(f"  NOT FOUND: {mlp_path}")

    mlp_ckpt = os.path.join(ROOT, "data/channel_generality/dt_v2_mlp/ppo_s0/checkpoints/ppo_v2_210000_steps.zip")
    if os.path.isfile(mlp_ckpt):
        print("\nMLP PPO (210K checkpoint)")
        eval_model(mlp_ckpt, scenarios=[0], n_episodes=10, seed=0)

    print("\n" + "=" * 70)
    print("Attention PPO (violation-only, baseline)")
    print("=" * 70)
    attn_path = os.path.join(ROOT, "data/channel_generality/dt_v2_per_scenario/ppo_s0/best_model/best_model.zip")
    if os.path.isfile(attn_path):
        eval_model(attn_path, scenarios=[0], n_episodes=10, seed=0)

    print("\n" + "=" * 70)
    print("Attention PPO (flat_bonus, baseline)")
    print("=" * 70)
    fb_path = os.path.join(ROOT, "data/channel_generality/dt_v2_flat_bonus/ppo_s0/best_model/best_model.zip")
    if os.path.isfile(fb_path):
        eval_model(fb_path, scenarios=[0], n_episodes=10, seed=0)


if __name__ == "__main__":
    main()
