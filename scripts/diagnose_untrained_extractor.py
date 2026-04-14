"""
Control experiment: Does a RANDOMLY INITIALIZED extractor also collapse?

If yes → architecture is inherently unable to differentiate obs
If no  → PPO training causes the collapse (representation collapse during learning)

Usage:
    cd /root/decision_transformer_slicing
    pixi run python -u scripts/diagnose_untrained_extractor.py
"""
import os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from omegaconf import OmegaConf
from gymnasium import spaces
from src.basic_apis.dt_v2.env_v2 import HierarchicalSlicingEnvV2
from src.basic_apis.ppo.ppo_ha_weighted.agent_hierarchical import HierarchicalAttentionExtractor
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


def main():
    env = make_env(0, seed=0)
    obs_space = env.observation_space

    # Create UNTRAINED extractor
    torch.manual_seed(42)
    extractor = HierarchicalAttentionExtractor(obs_space, features_dim=256)
    extractor.eval()

    # Also create a simple MLP baseline
    inter_dim = 5 * 8  # 40
    intra_dim = 25 * 7  # 175
    global_dim = 2
    total_dim = inter_dim + intra_dim + global_dim  # 217

    mlp_extractor = torch.nn.Sequential(
        torch.nn.Linear(total_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU()
    )
    mlp_extractor.eval()

    obs, _ = env.reset()
    attn_features = []
    mlp_features = []

    for t in range(300):
        action = env.action_space.sample()
        obs, r, done, _, info = env.step(action)
        if done:
            obs, _ = env.reset()

        obs_tensor = {}
        for k, v in obs.items():
            obs_tensor[k] = torch.tensor(v, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            feat_attn = extractor(obs_tensor).cpu().numpy().squeeze(0)
            attn_features.append(feat_attn)

            flat_obs = torch.cat([
                obs_tensor["inter_feat"].reshape(1, -1),
                obs_tensor["intra_feat"].reshape(1, -1),
                obs_tensor["global_feat"]
            ], dim=1)
            feat_mlp = mlp_extractor(flat_obs).cpu().numpy().squeeze(0)
            mlp_features.append(feat_mlp)

    env.close()

    for name, features in [("Untrained Attention", np.array(attn_features)),
                           ("Untrained MLP", np.array(mlp_features))]:
        T, D = features.shape
        feat_std = np.std(features, axis=0)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normed = features / norms
        cos_sim_matrix = normed @ normed.T
        triu_idx = np.triu_indices(T, k=1)
        cos_sims = cos_sim_matrix[triu_idx]

        centered = features - np.mean(features, axis=0)
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        explained_var = (S ** 2) / np.sum(S ** 2)
        cum_var = np.cumsum(explained_var)
        n_95 = np.searchsorted(cum_var, 0.95) + 1

        n_active = np.sum(feat_std > 0.01)
        n_high_cv = np.sum(feat_std / (np.abs(np.mean(features, axis=0)) + 1e-8) > 0.3)

        print(f"\n{'=' * 60}")
        print(f"{name}")
        print(f"{'=' * 60}")
        print(f"  Active dims (std > 0.01):  {n_active}/{D}")
        print(f"  High-CV dims:              {n_high_cv}/{D}")
        print(f"  Mean feature std:          {np.mean(feat_std):.6f}")
        print(f"  Cosine similarity:         mean={np.mean(cos_sims):.6f}, "
              f"std={np.std(cos_sims):.6f}")
        print(f"  Fraction cos > 0.99:       {np.mean(cos_sims > 0.99):.4f}")
        print(f"  Fraction cos > 0.95:       {np.mean(cos_sims > 0.95):.4f}")
        print(f"  PCA: 95% var in {n_95} components")
        print(f"  Top-1 PCA variance:        {explained_var[0]:.4f}")

        if np.mean(cos_sims > 0.99) > 0.9:
            print(f"  → COLLAPSE even without training!")
        elif np.mean(cos_sims > 0.95) > 0.8:
            print(f"  → NEAR-COLLAPSE even without training")
        else:
            print(f"  → OK: random init has reasonable feature diversity")

    print(f"\n{'=' * 60}")
    print("INTERPRETATION:")
    print("  If Untrained Attention also collapses → architecture problem")
    print("  If only trained model collapses → PPO training causes collapse")
    print("  Compare Attention vs MLP to see if attention is the issue")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
