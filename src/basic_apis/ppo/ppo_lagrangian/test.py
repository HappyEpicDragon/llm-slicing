"""
Lagrangian PPO 测试入口（B2 Baseline）。
5 场景 × 5 seeds，保存标准 JSON。

用法：
    python main.py name=channel_generality mode=test_ppo_lagrangian \
        test_ppo_lagrangian.scenario=5 test_ppo_lagrangian.seed=0
"""
import os
import json
import random
import shutil
import numpy as np
import torch
from omegaconf import DictConfig

from src.basic_apis.dt_v2.env_v2 import HierarchicalSlicingEnvV2
from src.basic_apis.ppo.ppo_lagrangian.lagrangian_ppo import LagrangianPPO
from src.basic_apis.ppo.ppo_lagrangian.obs_dim_utils import lagrangian_flat_obs_dim_v2


def _compute_step_metrics(info, num_slices=5):
    """
    与 metric_value.calc_intent_distance / calc_slice_violations 对齐的单步指标计算：
    - Distance: 每个活跃 slice 的 mean(drifts)（当 mean < 0 时计入），HP/NHP 分别求和
    - Violation: 每个活跃 slice 整体判断（mean(drifts) < 0 → 1次违约），HP/NHP 分别计数
    返回: (hp_dist, nhp_dist, hp_viols, nhp_viols, hp_active, nhp_active)
    """
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


def _load_lagrangian_for_inference(model_path: str, obs_dim: int, device: str = 'cpu') -> LagrangianPPO:
    """
    仅用于推理的 Lagrangian PPO 加载函数。
    绕过 LagrangianPPO.__init__ 中需要 env 的 PPO 构造，直接用 PPO.load 恢复策略网络。
    """
    from stable_baselines3 import PPO
    from src.basic_apis.ppo.ppo_lagrangian.cost_network import CostNetwork
    from src.basic_apis.ppo.ppo_lagrangian.lagrangian_ppo import LagrangianRewardWrapper, LagrangianCallback

    instance = LagrangianPPO.__new__(LagrangianPPO)

    # 加载 Lagrangian 状态
    lag_ckpt_path = model_path + "_lagrangian.pt"
    if os.path.exists(lag_ckpt_path):
        ckpt = torch.load(lag_ckpt_path, map_location=device)
        instance.lagrangian_multiplier = ckpt.get("lagrangian_multiplier", 10.0)
        instance.cost_critic = CostNetwork(obs_dim, hidden=128).to(device)
        instance.cost_critic.load_state_dict(ckpt["cost_critic_state_dict"])
    else:
        instance.lagrangian_multiplier = 10.0
        instance.cost_critic = CostNetwork(obs_dim, hidden=128).to(device)

    instance.cost_limit = 1e-4
    instance.lagrangian_lr = 1e-7
    instance.device = device
    instance.reward_wrapper = LagrangianRewardWrapper(instance)
    instance._lagrangian_cb = None

    # 用 PPO.load 恢复策略网络（env=None 在推理时合法）
    ppo_path = model_path + "_ppo"
    instance.ppo = PPO.load(ppo_path, env=None, device=device)

    return instance


def test_ppo_lagrangian(cfg: DictConfig, path_context):
    """Lagrangian PPO 测试主函数"""
    test_cfg = cfg.test_ppo_lagrangian

    test_scenarios = list(test_cfg.get('test_scenarios', [5, 6, 7, 8, 9]))
    test_seeds = list(test_cfg.get('test_seeds', [0, 1, 2, 3, 4]))
    model_seed_cfg = test_cfg.get('model_seed', None)
    model_seed = int(model_seed_cfg) if model_seed_cfg is not None else None
    n_episodes = int(test_cfg.get('n_episodes', 5))
    model_root = str(test_cfg.get('model_root', 'data/channel_generality/ppo_lagrangian_v2/models'))
    save_root = str(test_cfg.get('save_root', 'data/channel_generality/ppo_lagrangian_v2'))
    prefer_latest = bool(test_cfg.get('prefer_latest', True))
    strict_model_check = bool(test_cfg.get('strict_model_check', True))
    clean_before_save = bool(test_cfg.get('clean_before_save', False))
    from omegaconf import OmegaConf

    base_env_settings = OmegaConf.to_container(cfg.environment.env_settings, resolve=True)
    obs_dim = lagrangian_flat_obs_dim_v2(base_env_settings)

    all_results = {}
    if clean_before_save:
        metric_dir = os.path.join(save_root, "metric_json")
        if os.path.exists(metric_dir):
            shutil.rmtree(metric_dir)
        print(f"[Asset] cleaned old Lagrangian metrics under: {save_root}")

    for scenario_id in test_scenarios:
        for seed in test_seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            # 构建环境配置
            env_settings = OmegaConf.to_container(cfg.environment.env_settings, resolve=True)
            env_settings['mode'] = 'testing'
            scenario_mode = env_settings.get('scenario_mode', 'inside')
            env_settings[scenario_mode]['testing']['active_scenario_list'] = [scenario_id]

            from omegaconf import OmegaConf as OC
            env = HierarchicalSlicingEnvV2(OC.create(env_settings), np.random.default_rng(seed), path_context)

            # 加载模型：
            # - model_seed 设定时：使用单个训练模型，复用到所有 test_seeds
            # - 未设定时：保持旧行为，按每个 test seed 匹配同名模型
            model_seed_for_load = model_seed if model_seed is not None else seed
            candidate_paths = []
            if prefer_latest:
                candidate_paths.append(os.path.join(model_root, f"scen_all_seed{model_seed_for_load}", "latest", "model"))
                candidate_paths.append(os.path.join(model_root, f"scen{scenario_id}_seed{model_seed_for_load}", "latest", "model"))
            candidate_paths.append(os.path.join(model_root, f"scen_all_seed{model_seed_for_load}"))
            candidate_paths.append(os.path.join(model_root, f"scen{scenario_id}_seed{model_seed_for_load}"))
            model_path = None
            for cand in candidate_paths:
                if os.path.exists(cand + "_ppo.zip"):
                    model_path = cand
                    break
            if model_path is None:
                msg = f"Lagrangian model not found for scenario={scenario_id}, seed={seed}, model_seed={model_seed_for_load}"
                env.close()
                if strict_model_check:
                    raise FileNotFoundError(msg)
                print(f"  WARNING: {msg}, skipping.")
                continue

            lagrangian = _load_lagrangian_for_inference(
                model_path, obs_dim=obs_dim, device=str(test_cfg.get("device", "cpu"))
            )

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
                step_hp_dists, step_nhp_dists = [], []

                while not done:
                    action, _ = lagrangian.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
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

                hp_dist_ep  = ep_hp_dist_sum  / ep_hp_active  if ep_hp_active  > 0 else 0.0
                nhp_dist_ep = ep_nhp_dist_sum / ep_nhp_active if ep_nhp_active > 0 else 0.0
                hp_viol_ep  = ep_hp_viol_sum  / ep_hp_active  if ep_hp_active  > 0 else 0.0
                nhp_viol_ep = ep_nhp_viol_sum / ep_nhp_active if ep_nhp_active > 0 else 0.0
                ep_rewards.append(ep_reward)
                ep_hp_viols.append(float(hp_viol_ep))
                ep_nhp_viols.append(float(nhp_viol_ep))
                ep_hp_dists.append(float(hp_dist_ep))
                ep_nhp_dists.append(float(nhp_dist_ep))
                all_step_hp_dists.append(step_hp_dists)
                all_step_nhp_dists.append(step_nhp_dists)

            env.close()

            mean_hp_dist = float(np.mean(ep_hp_dists)) if ep_hp_dists else 0.0
            mean_nhp_dist = float(np.mean(ep_nhp_dists)) if ep_nhp_dists else 0.0

            summary = {
                "scenario_id": scenario_id,
                "seed": seed,
                "n_episodes": n_episodes,
                "reward_mean": float(np.mean(ep_rewards)),
                "reward_std": float(np.std(ep_rewards)),
                "hp_viol_mean": float(np.mean(ep_hp_viols)),
                "nhp_viol_mean": float(np.mean(ep_nhp_viols)),
                "hp_dist_mean": mean_hp_dist,
                "nhp_dist_mean": mean_nhp_dist,
                "per_episode_rewards": ep_rewards,
                "per_episode_hp_viols": ep_hp_viols,
                "per_episode_nhp_viols": ep_nhp_viols,
            }

            json_dir = os.path.join(save_root, "metric_json", f"scenario_{scenario_id}", f"seed_{seed}")
            _save_json(summary, os.path.join(json_dir, "summary.json"))
            _save_json({"hp_violations": ep_hp_viols, "mean": float(np.mean(ep_hp_viols))},
                       os.path.join(json_dir, "hp_violations.json"))
            _save_json({"nhp_violations": ep_nhp_viols, "mean": float(np.mean(ep_nhp_viols))},
                       os.path.join(json_dir, "nhp_violations.json"))
            _save_json({"rewards": ep_rewards, "mean": float(np.mean(ep_rewards))},
                       os.path.join(json_dir, "episode_rewards.json"))
            _save_json({"hp_distance": ep_hp_dists, "mean": mean_hp_dist},
                       os.path.join(json_dir, "hp_distance.json"))
            _save_json({"nhp_distance": ep_nhp_dists, "mean": mean_nhp_dist},
                       os.path.join(json_dir, "nhp_distance.json"))

            raw_dir = os.path.join(save_root, "metric_raw", f"scenario_{scenario_id}")
            os.makedirs(raw_dir, exist_ok=True)
            max_steps = max((len(s) for s in all_step_hp_dists), default=0)

            def _pad_steps(step_list, target_len):
                arr = np.array(step_list, dtype=np.float64)
                if len(arr) < target_len:
                    arr = np.concatenate([arr, np.zeros(target_len - len(arr), dtype=np.float64)])
                return arr

            np.savez(
                os.path.join(raw_dir, f"ep_seed{seed}.npz"),
                ep_rewards=np.array(ep_rewards, dtype=np.float64),
                hp_viols=np.array(ep_hp_viols, dtype=np.float64),
                nhp_viols=np.array(ep_nhp_viols, dtype=np.float64),
                step_hp_dist=np.array(
                    [_pad_steps(s, max_steps) for s in all_step_hp_dists], dtype=np.float64
                ),
                step_nhp_dist=np.array(
                    [_pad_steps(s, max_steps) for s in all_step_nhp_dists], dtype=np.float64
                ),
            )

            key = f"s{scenario_id}_seed{seed}"
            all_results[key] = summary
            print(f"Scenario {scenario_id}, Seed {seed}: "
                  f"R={summary['reward_mean']:.2f}, HP_Viol={summary['hp_viol_mean']:.2f}, "
                  f"HP_Dist={mean_hp_dist:.4f}")

    # 全局汇总
    if all_results:
        all_r = [v["reward_mean"] for v in all_results.values()]
        all_hp = [v["hp_viol_mean"] for v in all_results.values()]
        print(f"\nOverall: Avg Reward={np.mean(all_r):.2f}, Avg HP Viol={np.mean(all_hp):.2f}")

    return all_results
