"""
CQL 测试脚本：在 HierarchicalSlicingEnv 中逐场景、逐 seed 评估，
保存 summary.json + step-level npz（对齐 test_v2.py / ppo_multi 格式）。
"""
import os
import json
import argparse
import numpy as np
import torch

from src.basic_apis.cql_baseline.cql_env_wrapper import CQLObsWrapper
from src.basic_apis.ppo.ppo_ha_weighted.hierarchical_slicing_env import HierarchicalSlicingEnv


def _compute_step_metrics(info, num_slices=5):
    hp_dist = nhp_dist = 0.0
    hp_viols = nhp_viols = hp_active = nhp_active = 0
    for s_idx in range(num_slices):
        is_hp = info.get(f"meta/slice_{s_idx}_priority", 0) > 0
        drifts = [float(info.get(f"drift/slice_{s_idx}_{met}", 0.0))
                  for met in ("thr", "rel", "lat")
                  if info.get(f"meta/slice_{s_idx}_{met}_req", 0) > 0]
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


def _save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _flat145_obs(obs_dict) -> np.ndarray:
    """Convert env dict obs to 145-dim flat vector for CQL-baseline (ppo-baseline data)."""
    inter = np.array(obs_dict['inter_feat'], dtype=np.float32).flatten()   # 20
    intra = np.array(obs_dict['intra_feat'], dtype=np.float32).flatten()   # 125
    return np.concatenate([inter, intra])  # 145


def test_cql_scenario(
    model_path: str,
    env_config: dict,
    scenario_id: int,
    seed: int,
    n_episodes: int = 100,
    variant: str = "expert",
    save_root: str = "data/channel_generality/cql_v2",
    init_episode: int = None,
    max_episode: int = None,
    dataset_path_for_fallback: str = None,
    obs_dim: int = 45,
):
    """Run CQL evaluation on a single scenario + seed, save JSON + npz."""
    import d3rlpy

    np.random.seed(seed)
    torch.manual_seed(seed)

    wrapper = CQLObsWrapper()
    use_full_obs = (obs_dim == 145)

    try:
        cql = d3rlpy.load_learnable(model_path)
    except Exception:
        fb_path = dataset_path_for_fallback
        if fb_path is None:
            fb_path = os.path.join(save_root, "dataset.h5")
        if not os.path.exists(fb_path):
            raise RuntimeError(f"Cannot load CQL model and fallback dataset not found: {fb_path}")

        with open(fb_path, "rb") as f:
            dataset = d3rlpy.dataset.ReplayBuffer.load(f, d3rlpy.dataset.InfiniteBuffer())

        cql = d3rlpy.algos.CQLConfig(
            actor_learning_rate=5e-5,
            critic_learning_rate=1e-3,
            initial_alpha=1.0,
            conservative_weight=1.0,
        ).create(device="cpu")
        cql.build_with_dataset(dataset)
        cql.load_model(model_path)
        print(f"[WARN] Fallback CQL loader used")
    print(f"CQL model loaded from {model_path}")

    env_config_copy = dict(env_config)
    env_settings = dict(env_config_copy.get("env_settings", env_config_copy))
    scenario_mode = env_settings.get('scenario_mode', 'inside')
    mode = 'testing'
    env_settings['mode'] = mode
    env_settings[scenario_mode][mode]['active_scenario_list'] = [scenario_id]

    if init_episode is not None:
        env_settings[scenario_mode][mode]['init_scenario_episode'] = init_episode
    if max_episode is not None:
        env_settings[scenario_mode][mode]['max_scenario_episodes'] = max_episode

    from omegaconf import OmegaConf
    from src.basic_apis.network_slicing_business.path_context import PathContext

    cfg_node = OmegaConf.create(env_settings)
    path_context = PathContext("./outputs/cql_test")
    env = HierarchicalSlicingEnv(cfg_node, np.random.default_rng(seed), path_context)

    ep_rewards, ep_hp_viols, ep_nhp_viols = [], [], []
    ep_hp_dists, ep_nhp_dists = [], []
    all_step_hp_dists, all_step_nhp_dists = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        ep_hp_dist_sum = ep_nhp_dist_sum = 0.0
        ep_hp_viol_sum = ep_nhp_viol_sum = 0
        ep_hp_active = ep_nhp_active = 0
        step_hp_dists, step_nhp_dists, step_rewards = [], [], []

        while not done:
            flat_obs = _flat145_obs(obs) if use_full_obs else wrapper.transform_obs(obs)
            action_cont = cql.predict(flat_obs[np.newaxis])[0]
            action_disc = wrapper.transform_action(action_cont)

            obs, reward, terminated, truncated, info = env.step(action_disc)
            ep_reward += reward
            done = terminated or truncated

            hp_d, nhp_d, hp_v, nhp_v, hp_a, nhp_a = _compute_step_metrics(info)
            ep_hp_dist_sum += hp_d
            ep_nhp_dist_sum += nhp_d
            ep_hp_viol_sum += hp_v
            ep_nhp_viol_sum += nhp_v
            ep_hp_active += hp_a
            ep_nhp_active += nhp_a
            step_hp_dists.append(hp_d / hp_a if hp_a > 0 else 0.0)
            step_nhp_dists.append(nhp_d / nhp_a if nhp_a > 0 else 0.0)
            step_rewards.append(float(reward))

        hp_dist_ep = ep_hp_dist_sum / ep_hp_active if ep_hp_active > 0 else 0.0
        nhp_dist_ep = ep_nhp_dist_sum / ep_nhp_active if ep_nhp_active > 0 else 0.0
        hp_viol_ep = ep_hp_viol_sum / ep_hp_active if ep_hp_active > 0 else 0.0
        nhp_viol_ep = ep_nhp_viol_sum / ep_nhp_active if ep_nhp_active > 0 else 0.0

        ep_rewards.append(ep_reward)
        ep_hp_viols.append(float(hp_viol_ep))
        ep_nhp_viols.append(float(nhp_viol_ep))
        ep_hp_dists.append(float(hp_dist_ep))
        ep_nhp_dists.append(float(nhp_dist_ep))
        all_step_hp_dists.append(step_hp_dists)
        all_step_nhp_dists.append(step_nhp_dists)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Ep {ep}: reward={ep_reward:.2f}, hp_viol={hp_viol_ep:.4f}, "
                  f"nhp_viol={nhp_viol_ep:.4f}, hp_dist={hp_dist_ep:.4f}")

    env.close()

    mean_hp_dist = float(np.mean(ep_hp_dists)) if ep_hp_dists else 0.0
    mean_nhp_dist = float(np.mean(ep_nhp_dists)) if ep_nhp_dists else 0.0

    summary = {
        "variant": variant,
        "scenario_id": scenario_id,
        "seed": seed,
        "n_episodes": n_episodes,
        "reward_mean": float(np.mean(ep_rewards)),
        "reward_std": float(np.std(ep_rewards)),
        "hp_viol_mean": float(np.mean(ep_hp_viols)),
        "nhp_viol_mean": float(np.mean(ep_nhp_viols)),
        "hp_dist_mean": mean_hp_dist,
        "nhp_dist_mean": mean_nhp_dist,
        "combined": float(np.mean(ep_hp_viols)) + float(np.mean(ep_nhp_viols)),
    }

    base_dir = os.path.join(save_root, f"cql_{variant}")
    json_dir = os.path.join(base_dir, "metric_json", f"scenario_{scenario_id}", f"seed_{seed}")
    _save_json(summary, os.path.join(json_dir, "summary.json"))
    _save_json({"hp_violations": ep_hp_viols, "mean": float(np.mean(ep_hp_viols)), "episodes": len(ep_hp_viols)},
               os.path.join(json_dir, "hp_violations.json"))
    _save_json({"nhp_violations": ep_nhp_viols, "mean": float(np.mean(ep_nhp_viols)), "episodes": len(ep_nhp_viols)},
               os.path.join(json_dir, "nhp_violations.json"))
    _save_json({"rewards": ep_rewards, "mean": float(np.mean(ep_rewards)), "episodes": len(ep_rewards)},
               os.path.join(json_dir, "episode_rewards.json"))
    _save_json({"hp_distance": ep_hp_dists, "mean": mean_hp_dist, "episodes": len(ep_hp_dists)},
               os.path.join(json_dir, "hp_distance.json"))
    _save_json({"nhp_distance": ep_nhp_dists, "mean": mean_nhp_dist, "episodes": len(ep_nhp_dists)},
               os.path.join(json_dir, "nhp_distance.json"))

    raw_dir = os.path.join(base_dir, "metric_raw", f"scenario_{scenario_id}")
    os.makedirs(raw_dir, exist_ok=True)
    max_steps = max(len(s) for s in all_step_hp_dists) if all_step_hp_dists else 0
    def _pad(lst, length):
        arr = np.array(lst, dtype=np.float64)
        if len(arr) < length:
            arr = np.concatenate([arr, np.zeros(length - len(arr))])
        return arr

    np.savez(
        os.path.join(raw_dir, f"ep_seed{seed}.npz"),
        ep_rewards=np.array(ep_rewards, dtype=np.float64),
        hp_viols=np.array(ep_hp_viols, dtype=np.float64),
        nhp_viols=np.array(ep_nhp_viols, dtype=np.float64),
        step_hp_dist=np.array([_pad(s, max_steps) for s in all_step_hp_dists], dtype=np.float64),
        step_nhp_dist=np.array([_pad(s, max_steps) for s in all_step_nhp_dists], dtype=np.float64),
    )

    print(f"Results saved to {json_dir} + {raw_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Test CQL baseline")
    parser.add_argument('--model', required=True, help='CQL model.pt path')
    parser.add_argument('--scenario', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--variant', default='expert', choices=['expert', 'raw'])
    parser.add_argument('--save_root', default='data/channel_generality/cql_v2')
    parser.add_argument('--init_episode', type=int, default=None)
    parser.add_argument('--max_episode', type=int, default=None)
    parser.add_argument('--obs_dim', type=int, default=45,
                        help='45 for cql_v2 (wrapper-compressed), 145 for cql_baseline (ppo-baseline data)')
    parser.add_argument('--env_config', default='conf/environment/env_ha.yaml')
    args = parser.parse_args()

    from omegaconf import OmegaConf
    env_cfg = OmegaConf.load(args.env_config)
    env_settings = OmegaConf.to_container(env_cfg.env_settings, resolve=True)
    env_config = {"env_settings": env_settings}

    test_cql_scenario(
        model_path=args.model,
        env_config=env_config,
        scenario_id=args.scenario,
        seed=args.seed,
        n_episodes=args.n_episodes,
        variant=args.variant,
        save_root=args.save_root,
        init_episode=args.init_episode,
        max_episode=args.max_episode,
        obs_dim=args.obs_dim,
    )


if __name__ == '__main__':
    main()
