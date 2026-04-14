"""
Experiment C: Feature extractor output analysis.

Checks whether HierarchicalAttentionExtractor produces distinguishable
representations for different timesteps within an episode.

Metrics:
  1. Pairwise cosine similarity distribution
  2. Feature variance per dimension
  3. PCA explained variance (can features be separated?)

Usage:
    cd /root/decision_transformer_slicing
    pixi run python -u scripts/diagnose_feature_extractor.py
"""
import argparse, os, sys
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


def collect_features(model, env, n_steps=300):
    """Run episode, extract feature vectors at each step."""
    obs, _ = env.reset()
    features_list = []
    obs_list = []
    actions_list = []
    rewards_list = []

    fe = model.policy.features_extractor
    fe.eval()

    for t in range(n_steps):
        obs_tensor = {}
        for k, v in obs.items():
            obs_tensor[k] = torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(model.device)

        with torch.no_grad():
            feat = fe(obs_tensor)
        features_list.append(feat.cpu().numpy().squeeze(0))

        obs_flat = np.concatenate([obs["inter_feat"].flatten(),
                                   obs["intra_feat"].flatten(),
                                   obs["global_feat"].flatten()])
        obs_list.append(obs_flat)

        # Step with deterministic policy
        obs_wrapped = {k: np.expand_dims(v, 0) for k, v in obs.items()}
        action, _ = model.predict(obs_wrapped, deterministic=True)
        action = action.squeeze(0)
        actions_list.append(action.copy())

        obs, r, done, _, info = env.step(action)
        rewards_list.append(r)
        if done:
            obs, _ = env.reset()

    return (np.array(features_list), np.array(obs_list),
            np.array(actions_list), np.array(rewards_list))


def analyze_features(features, obs_arr, actions, rewards, model_name):
    T, D = features.shape
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT C: Feature Extractor Analysis — {model_name}")
    print(f"  Feature shape: ({T}, {D})")
    print(f"{'=' * 70}")

    # 1. Per-dimension statistics
    feat_mean = np.mean(features, axis=0)
    feat_std = np.std(features, axis=0)
    feat_cv = feat_std / (np.abs(feat_mean) + 1e-8)

    n_active = np.sum(feat_std > 0.01)
    n_high_cv = np.sum(feat_cv > 0.3)
    print(f"\n--- Feature dimension statistics ---")
    print(f"  Active dims (std > 0.01): {n_active}/{D}")
    print(f"  High-CV dims (CV > 0.3):  {n_high_cv}/{D}")
    print(f"  Mean feature std:         {np.mean(feat_std):.6f}")
    print(f"  Max feature std:          {np.max(feat_std):.6f}")
    print(f"  Min feature std:          {np.min(feat_std):.6f}")

    # 2. Pairwise cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = features / norms
    cos_sim_matrix = normed @ normed.T

    triu_idx = np.triu_indices(T, k=1)
    cos_sims = cos_sim_matrix[triu_idx]

    print(f"\n--- Pairwise cosine similarity (all {len(cos_sims)} pairs) ---")
    print(f"  Mean: {np.mean(cos_sims):.6f}")
    print(f"  Std:  {np.std(cos_sims):.6f}")
    print(f"  Min:  {np.min(cos_sims):.6f}")
    print(f"  Max:  {np.max(cos_sims):.6f}")
    print(f"  Fraction > 0.99: {np.mean(cos_sims > 0.99):.4f}")
    print(f"  Fraction > 0.95: {np.mean(cos_sims > 0.95):.4f}")
    print(f"  Fraction < 0.90: {np.mean(cos_sims < 0.90):.4f}")

    # 3. L2 distance statistics
    diffs = []
    for i in range(min(T, 500)):
        for j in range(i + 1, min(T, 500)):
            diffs.append(np.linalg.norm(features[i] - features[j]))
    diffs = np.array(diffs)
    print(f"\n--- Pairwise L2 distance ---")
    print(f"  Mean: {np.mean(diffs):.6f}")
    print(f"  Std:  {np.std(diffs):.6f}")
    print(f"  Min:  {np.min(diffs):.6f}")
    print(f"  Max:  {np.max(diffs):.6f}")

    # 4. PCA analysis
    centered = features - feat_mean
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        explained_var = (S ** 2) / np.sum(S ** 2)
        cum_var = np.cumsum(explained_var)

        print(f"\n--- PCA explained variance ---")
        for k in [1, 2, 3, 5, 10, 20, 50]:
            if k <= len(cum_var):
                print(f"  Top {k:>2d} components: {cum_var[k - 1]:.4f}")

        n_95 = np.searchsorted(cum_var, 0.95) + 1
        print(f"  Components for 95% variance: {n_95}")
    except Exception as e:
        print(f"  PCA failed: {e}")

    # 5. Temporal auto-correlation (do consecutive features look different?)
    consecutive_cos = []
    consecutive_l2 = []
    for i in range(T - 1):
        cs = np.dot(normed[i], normed[i + 1])
        consecutive_cos.append(cs)
        consecutive_l2.append(np.linalg.norm(features[i] - features[i + 1]))

    print(f"\n--- Consecutive timestep similarity ---")
    print(f"  Cosine sim: mean={np.mean(consecutive_cos):.6f}, "
          f"std={np.std(consecutive_cos):.6f}")
    print(f"  L2 dist:    mean={np.mean(consecutive_l2):.6f}, "
          f"std={np.std(consecutive_l2):.6f}")

    # 6. Compare obs variation vs feature variation
    obs_std = np.mean(np.std(obs_arr, axis=0))
    feat_std_mean = np.mean(feat_std)
    print(f"\n--- Obs vs Feature variation ---")
    print(f"  Mean obs dim std:     {obs_std:.6f}")
    print(f"  Mean feature dim std: {feat_std_mean:.6f}")
    print(f"  Ratio (feat/obs):     {feat_std_mean / (obs_std + 1e-8):.4f}")

    # 7. Verdict
    print(f"\n=== VERDICT C ({model_name}) ===")
    if np.mean(cos_sims > 0.99) > 0.9:
        print("  COLLAPSE: >90% pairs have cos_sim > 0.99 → extractor maps all obs to ~same point")
        print("  → Feature extractor is the bottleneck!")
    elif np.mean(cos_sims > 0.95) > 0.8:
        print("  NEAR-COLLAPSE: >80% pairs have cos_sim > 0.95 → very low distinguishability")
        print("  → Feature extractor significantly compresses information")
    else:
        print(f"  OK: Features show reasonable variation (mean cos_sim={np.mean(cos_sims):.4f})")
        if n_high_cv < 10:
            print(f"  BUT only {n_high_cv} dims have high CV → information may be concentrated in few dims")
        print("  → Problem likely in PPO optimization, not feature extraction")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_steps", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    models = [
        ("dt_v2_per_scenario (violation-only)",
         f"data/channel_generality/dt_v2_per_scenario/ppo_s{args.scenario}/best_model/best_model.zip"),
        ("dt_v2_flat_bonus",
         f"data/channel_generality/dt_v2_flat_bonus/ppo_s{args.scenario}/best_model/best_model.zip"),
    ]

    for model_name, model_path in models:
        full_path = os.path.join(ROOT, model_path)
        if not os.path.isfile(full_path):
            print(f"SKIP {model_name}: {full_path} not found")
            continue

        print(f"\nLoading {model_name} from {model_path}")
        model = PPO.load(full_path, device=args.device)

        env = make_env(args.scenario, seed=args.seed)
        features, obs_arr, actions, rewards = collect_features(model, env, n_steps=args.n_steps)
        env.close()

        analyze_features(features, obs_arr, actions, rewards, model_name)

        # Also check action diversity
        print(f"\n--- Action diversity (deterministic policy) ---")
        for i in range(actions.shape[1]):
            unique = np.unique(actions[:, i])
            print(f"  Action[{i}]: {len(unique)} unique values, most_common={np.bincount(actions[:, i]).argmax()}")


if __name__ == "__main__":
    main()
