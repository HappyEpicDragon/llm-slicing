"""
PPO-Lagrangian Baseline 测试入口。

使用与 ppo-baseline / dt-baseline **相同的仿真环境**（CommunicationEnv + IBSched），
在 s5–s9 的 ep 0–99（100 个 episode）上测试，5 个随机种子。

指标保存至 data/channel_generality/ppo_lagrangian_baseline/metric_json/

用法（Hydra）：
    pixi run sim channel_generality/test_ppo_lagrangian_baseline
"""
import os
import json
import random
import shutil
import numpy as np
import torch
from omegaconf import DictConfig

from src.basic_apis.ppo.ppo_lagrangian.lagrangian_ppo import LagrangianPPO
from src.basic_apis.ppo.ppo_lagrangian.obs_dim_utils import LAGRANGIAN_BASELINE_OBS_DIM
from src.basic_apis.ppo.ppo_lagrangian.train_baseline import (
    LagrangianBaselineEnv, _build_comm_env_cfg, NUM_SLICES
)

OBS_DIM = LAGRANGIAN_BASELINE_OBS_DIM  # 145


# --------------------------------------------------------------------------
# Step-level metric helper（与 test.py / ppo_ha_weighted/test.py 保持一致）
# --------------------------------------------------------------------------

def _compute_step_metrics(info, num_slices=NUM_SLICES):
    """
    单步 HP/NHP distance & violation（与 metric_value 计算口径对齐）。
    Returns: (hp_dist, nhp_dist, hp_viols, nhp_viols, hp_active, nhp_active)
    """
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


def _load_lagrangian_for_inference(model_path: str, device: str = 'cpu') -> LagrangianPPO:
    """仅用于推理的加载函数（与 test.py 相似，使用 baseline OBS_DIM）。"""
    from stable_baselines3 import PPO
    from src.basic_apis.ppo.ppo_lagrangian.cost_network import CostNetwork
    from src.basic_apis.ppo.ppo_lagrangian.lagrangian_ppo import (
        LagrangianRewardWrapper, LagrangianCallback
    )

    instance = LagrangianPPO.__new__(LagrangianPPO)
    lag_ckpt_path = model_path + "_lagrangian.pt"
    if os.path.exists(lag_ckpt_path):
        ckpt = torch.load(lag_ckpt_path, map_location=device)
        instance.lagrangian_multiplier = ckpt.get("lagrangian_multiplier", 10.0)
        instance.cost_critic = CostNetwork(OBS_DIM, hidden=128).to(device)
        instance.cost_critic.load_state_dict(ckpt["cost_critic_state_dict"])
    else:
        instance.lagrangian_multiplier = 10.0
        instance.cost_critic = CostNetwork(OBS_DIM, hidden=128).to(device)

    instance.cost_limit = 1e-4
    instance.lagrangian_lr = 1e-7
    instance.device = device
    instance.reward_wrapper = LagrangianRewardWrapper(instance)
    instance._lagrangian_cb = None

    ppo_path = model_path + "_ppo"
    instance.ppo = PPO.load(ppo_path, env=None, device=device)
    return instance


def test_ppo_lagrangian_baseline(cfg: DictConfig, paths_cfg=None, workdir=None):
    """PPO-Lagrangian Baseline 测试主函数。"""
    test_cfg = cfg.test_ppo_lagrangian_baseline
    paths_cfg = paths_cfg if paths_cfg is not None else cfg.get("paths", None)
    workdir = workdir if workdir is not None else str(cfg.get("workdir", os.getcwd()))

    test_scenarios = list(test_cfg.get('test_scenarios', [5, 6, 7, 8, 9]))
    test_seeds = list(test_cfg.get('test_seeds', [0, 1, 2, 3, 4]))
    model_seed_cfg = test_cfg.get('model_seed', None)
    model_seed = int(model_seed_cfg) if model_seed_cfg is not None else None
    n_episodes = int(test_cfg.get('n_episodes', 100))
    model_root = str(test_cfg.get(
        'model_root', 'data/channel_generality/ppo_lagrangian_baseline/models'
    ))
    save_root = str(test_cfg.get(
        'save_root', 'data/channel_generality/ppo_lagrangian_baseline'
    ))
    prefer_latest = bool(test_cfg.get('prefer_latest', True))
    strict_model_check = bool(test_cfg.get('strict_model_check', True))
    clean_before_save = bool(test_cfg.get('clean_before_save', False))
    device = str(test_cfg.get('device', 'cpu'))

    # 测试时 episode 范围：ep 0–99（OOD 场景全部 100 个 ep 均可用于测试）
    init_episode = int(test_cfg.get('init_episode', 0))
    max_episode = int(test_cfg.get('max_episode', 100))

    from omegaconf import OmegaConf
    base_env_cfg = OmegaConf.to_container(cfg.environment, resolve=True)

    if clean_before_save:
        metric_dir = os.path.join(save_root, "metric_json")
        if os.path.exists(metric_dir):
            shutil.rmtree(metric_dir)
        print(f"[Asset] cleaned old metrics under: {save_root}")

    all_results = {}

    for scenario_id in test_scenarios:
        for seed in test_seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            # 构建 CommunicationEnv 配置（测试模式，覆盖 episode 范围）
            comm_env_cfg = _build_comm_env_cfg(
                base_env_cfg, paths_cfg, workdir, [scenario_id], seed
            )
            # 切换为 testing 模式，并使用 ep 0–99
            comm_env_cfg['mode'] = 'testing'
            sm = comm_env_cfg.get('scenario_mode', 'inside')
            comm_env_cfg[sm]['testing']['active_scenario_list'] = [scenario_id]
            comm_env_cfg[sm]['testing']['initial_episode'] = init_episode
            comm_env_cfg[sm]['testing']['max_episode'] = max_episode
            comm_env_cfg[sm]['testing'].setdefault('save_hist', False)
            # 刷新 state_config
            comm_env_cfg['state_config']['initial_episode'] = init_episode
            comm_env_cfg['state_config']['max_episode'] = max_episode
            comm_env_cfg['state_config']['active_scenario_list'] = [scenario_id]
            comm_env_cfg['state_config']['scenario_skip_episodes'] = \
                comm_env_cfg[sm].get('scenario_skip_episodes', 100)
            comm_env_cfg['seed_test'] = seed

            # 查找模型文件
            model_seed_for_load = model_seed if model_seed is not None else seed
            candidate_paths = []
            if prefer_latest:
                candidate_paths.append(os.path.join(
                    model_root, f"scen_all_seed{model_seed_for_load}", "latest", "model"
                ))
                candidate_paths.append(os.path.join(
                    model_root,
                    f"scen{'_'.join(map(str, [0,1,2,3,4]))}_seed{model_seed_for_load}",
                    "latest", "model"
                ))
            candidate_paths.append(os.path.join(
                model_root, f"scen_all_seed{model_seed_for_load}"
            ))
            candidate_paths.append(os.path.join(
                model_root,
                f"scen{'_'.join(map(str, [0,1,2,3,4]))}_seed{model_seed_for_load}"
            ))

            model_path = None
            for cand in candidate_paths:
                if os.path.exists(cand + "_ppo.zip"):
                    model_path = cand
                    break
            if model_path is None:
                msg = (f"Lagrangian-Baseline model not found for "
                       f"scenario={scenario_id}, seed={seed}, "
                       f"model_seed={model_seed_for_load}")
                if strict_model_check:
                    raise FileNotFoundError(msg)
                print(f"  WARNING: {msg}, skipping.")
                continue

            lagrangian = _load_lagrangian_for_inference(model_path, device=device)

            # 创建测试环境（不需要 Lagrangian 增广，直接 predict 即可）
            env = LagrangianBaselineEnv(comm_env_cfg)

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
                    ep_reward += float(reward)
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

            json_dir = os.path.join(
                save_root, "metric_json", f"scenario_{scenario_id}", f"seed_{seed}"
            )
            _save_json(summary, os.path.join(json_dir, "summary.json"))
            _save_json(
                {"hp_violations": ep_hp_viols, "mean": float(np.mean(ep_hp_viols)),
                 "episodes": len(ep_hp_viols)},
                os.path.join(json_dir, "hp_violations.json")
            )
            _save_json(
                {"nhp_violations": ep_nhp_viols, "mean": float(np.mean(ep_nhp_viols)),
                 "episodes": len(ep_nhp_viols)},
                os.path.join(json_dir, "nhp_violations.json")
            )
            _save_json(
                {"hp_distance": ep_hp_dists, "mean": mean_hp_dist,
                 "episodes": len(ep_hp_dists)},
                os.path.join(json_dir, "hp_distance.json")
            )
            _save_json(
                {"nhp_distance": ep_nhp_dists, "mean": mean_nhp_dist,
                 "episodes": len(ep_nhp_dists)},
                os.path.join(json_dir, "nhp_distance.json")
            )
            _save_json(
                {"rewards": ep_rewards, "mean": float(np.mean(ep_rewards))},
                os.path.join(json_dir, "episode_rewards.json")
            )

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
                  f"R={summary['reward_mean']:.2f}, "
                  f"HP_Viol={summary['hp_viol_mean']:.3f}, "
                  f"HP_Dist={mean_hp_dist:.4f}, "
                  f"NHP_Viol={summary['nhp_viol_mean']:.3f}, "
                  f"NHP_Dist={mean_nhp_dist:.4f}")

    if all_results:
        all_hp = [v["hp_viol_mean"] for v in all_results.values()]
        all_nhp = [v["nhp_viol_mean"] for v in all_results.values()]
        print(f"\nOverall: Avg HP_Viol={np.mean(all_hp):.3f}, "
              f"Avg NHP_Viol={np.mean(all_nhp):.3f}")

    return all_results
