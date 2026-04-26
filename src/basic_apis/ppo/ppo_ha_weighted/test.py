import os
import json
import shutil
import numpy as np
import torch
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from omegaconf import OmegaConf

# === Local Imports ===
from src.basic_apis.ppo.ppo_ha_weighted.hierarchical_slicing_env import HierarchicalSlicingEnv
from src.basic_apis.ppo.ppo_ha_weighted.agent_hierarchical import HierarchicalSmartPolicy


def _save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _compute_step_metrics(info, num_slices=5):
    """
    与 metric_value.calc_intent_distance / calc_slice_violations 对齐的单步指标计算：
    - Distance: 每个活跃 slice 的 mean(drifts)（当 mean < 0 时计入），HP/NHP 分别求和
    - Violation: 每个活跃 slice 整体判断（mean(drifts) < 0 → 1次违约），HP/NHP 分别计数
    返回: (hp_dist, nhp_dist, hp_viols, nhp_viols, hp_active, nhp_active)
      hp_dist/nhp_dist  : 当步 HP/NHP slice 的 mean-drift 之和（≤ 0）
      hp_viols/nhp_viols: 当步违约 HP/NHP slice 数
      hp_active/nhp_active: 当步活跃 HP/NHP slice 数（归一化分母）
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


def _new_detail_stats():
    return {
        "margin_sum": 0.0,
        "margin_count": 0,
        "severity_sum": 0.0,
        "severity_count": 0,
        "qos": {
            metric: {"margin_sum": 0.0, "count": 0, "viol": 0, "severity_sum": 0.0}
            for metric in ("thr", "lat", "rel")
        },
    }


def _compute_detail_step_metrics(info, num_slices=5):
    """Metric-level intent margins for MVP reporting."""
    stats = _new_detail_stats()
    for s_idx in range(num_slices):
        if info.get(f"meta/slice_{s_idx}_active", 0) == 0:
            continue
        for metric in ("thr", "lat", "rel"):
            if info.get(f"meta/slice_{s_idx}_{metric}_req", 0) == 0:
                continue
            margin = float(info.get(f"drift/slice_{s_idx}_{metric}", 0.0))
            q_stats = stats["qos"][metric]
            q_stats["margin_sum"] += margin
            q_stats["count"] += 1
            stats["margin_sum"] += margin
            stats["margin_count"] += 1
            if margin < 0:
                severity = abs(margin)
                q_stats["viol"] += 1
                q_stats["severity_sum"] += severity
                stats["severity_sum"] += severity
                stats["severity_count"] += 1
    return stats


def _merge_detail_stats(total, step):
    total["margin_sum"] += step["margin_sum"]
    total["margin_count"] += step["margin_count"]
    total["severity_sum"] += step["severity_sum"]
    total["severity_count"] += step["severity_count"]
    for metric in ("thr", "lat", "rel"):
        dst = total["qos"][metric]
        src = step["qos"][metric]
        dst["margin_sum"] += src["margin_sum"]
        dst["count"] += src["count"]
        dst["viol"] += src["viol"]
        dst["severity_sum"] += src["severity_sum"]


def _finalize_detail_stats(stats):
    out = {
        "mean_intent_margin": (
            stats["margin_sum"] / stats["margin_count"]
            if stats["margin_count"] > 0 else 0.0
        ),
        "violation_severity": (
            stats["severity_sum"] / stats["severity_count"]
            if stats["severity_count"] > 0 else 0.0
        ),
    }
    for metric in ("thr", "lat", "rel"):
        q_stats = stats["qos"][metric]
        count = q_stats["count"]
        out[f"{metric}_viol"] = q_stats["viol"] / count if count > 0 else 0.0
        out[f"{metric}_margin"] = q_stats["margin_sum"] / count if count > 0 else 0.0
        out[f"{metric}_severity"] = (
            q_stats["severity_sum"] / q_stats["viol"]
            if q_stats["viol"] > 0 else 0.0
        )
    return out


MVP_EXTRA_METRICS = (
    "total_viol",
    "mean_intent_margin",
    "violation_severity",
    "thr_viol",
    "lat_viol",
    "rel_viol",
    "thr_margin",
    "lat_margin",
    "rel_margin",
    "thr_severity",
    "lat_severity",
    "rel_severity",
)


def make_env(cfg, paths_cfg=None, workdir=None, rank=0, seed=0):
    """H+A 环境工厂函数"""

    def _init():
        env_config = cfg.env_settings if hasattr(cfg, 'env_settings') else cfg
        if 'env_settings' in cfg:
            env_config = cfg.env_settings
        env = HierarchicalSlicingEnv(
            env_config,
            np.random.default_rng(seed + rank),
            paths_cfg=paths_cfg,
            workdir=workdir,
        )
        return env

    return _init


def _load_model(model_path, env):
    """加载 SB3 PPO 模型，兼容 HierarchicalSmartPolicy"""
    try:
        return PPO.load(model_path, env=env, device='cpu',
                        custom_objects={"learning_rate": 0.0, "clip_range": 0.1})
    except Exception:
        return PPO.load(model_path, env=env, device='cpu',
                        custom_objects={"HierarchicalSmartPolicy": HierarchicalSmartPolicy})


def test_ppo_ha(cfg, paths_cfg=None, workdir=None):
    """H+A (PPO Discrete Teacher) 测试入口，支持 5 seeds × N 场景，保存标准 JSON。"""
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    environment_cfg = cfg.environment
    paths_cfg = paths_cfg if paths_cfg is not None else cfg.get("paths", None)
    workdir = workdir if workdir is not None else str(cfg.get("workdir", os.getcwd()))
    env_updates = cfg.env_updates

    # 更新环境基础配置
    environment_cfg.env_settings.mode = env_updates.mode
    environment_cfg.env_settings.scenario_mode = env_updates.scenario_mode
    environment_cfg.env_settings.model_name = env_updates.model_name
    environment_cfg.env_settings.inside.training.update(env_updates.inside.training)
    environment_cfg.env_settings.inside.evaluating.update(env_updates.inside.evaluating)
    environment_cfg.env_settings.inside.testing.update(env_updates.inside.testing)

    model_path = cfg.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PPO-HA-weighted model not found: {model_path}")

    # 多种子配置
    test_seeds = list(cfg.get('test_seeds', [0, 1, 2, 3, 4]))
    save_results = bool(cfg.get('save_results', True))
    save_root = str(cfg.get('save_root', 'data/channel_generality/ppo_ha_weighted'))
    clean_before_save = bool(cfg.get('clean_before_save', False))
    if save_results and clean_before_save:
        for subdir in ("metric_json", "metric_raw"):
            old_dir = os.path.join(save_root, subdir)
            if os.path.exists(old_dir):
                shutil.rmtree(old_dir)
        print(f"[Asset] cleaned old metrics under: {save_root}")

    # 确定场景列表
    scenario_mode = environment_cfg.env_settings.scenario_mode
    mode = environment_cfg.env_settings.mode
    scenario_list = list(environment_cfg.env_settings[scenario_mode][mode].active_scenario_list)
    test_cfg_env = environment_cfg.env_settings[scenario_mode][mode]
    n_episodes = test_cfg_env.max_scenario_episodes - test_cfg_env.init_scenario_episode

    print(f"Loading Model from: {model_path}")
    print(f"Test Seeds: {test_seeds}, Scenarios: {scenario_list}, Episodes/seed: {n_episodes}")

    all_scenario_results = {}

    for scenario_id in scenario_list:
        print(f"\n{'='*60}\nEvaluating Scenario {scenario_id}\n{'='*60}")

        # 更新场景 ID
        environment_cfg.env_settings[scenario_mode][mode].active_scenario_list = [scenario_id]

        seed_rewards, seed_hp_viols, seed_nhp_viols = [], [], []
        seed_hp_dists, seed_nhp_dists = [], []
        seed_extra = {metric: [] for metric in MVP_EXTRA_METRICS}

        for seed in test_seeds:
            print(f"  Seed {seed} / Scenario {scenario_id}")
            env = DummyVecEnv([make_env(
                environment_cfg,
                paths_cfg=paths_cfg,
                workdir=workdir,
                rank=0,
                seed=seed,
            )])

            model = _load_model(model_path, env)
            obs = env.reset()

            ep_rewards, ep_hp_viols, ep_nhp_viols = [], [], []
            ep_hp_dists, ep_nhp_dists = [], []
            ep_extra = {metric: [] for metric in MVP_EXTRA_METRICS}
            seed_step_hp_dists, seed_step_nhp_dists = [], []
            seed_step_mean_margins, seed_step_severities = [], []
            seed_step_qos = {
                metric: {"viol": [], "margin": [], "severity": []}
                for metric in ("thr", "lat", "rel")
            }
            pbar = tqdm(total=n_episodes, unit="ep", desc=f"S{scenario_id} seed{seed}")
            current_ep_reward = 0.0
            # 当前 episode 逐步累计量
            ep_hp_dist_sum = ep_nhp_dist_sum = 0.0
            ep_hp_viol_sum = ep_nhp_viol_sum = 0
            ep_hp_active = ep_nhp_active = 0
            ep_detail_stats = _new_detail_stats()
            ep_step_hp_dists, ep_step_nhp_dists = [], []

            try:
                while len(ep_rewards) < n_episodes:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, rewards, dones, infos = env.step(action)
                    info = infos[0]

                    current_ep_reward += rewards[0]

                    # 逐步累计 distance 和 violation（与 metric_value.py 对齐）
                    hp_d, nhp_d, hp_v, nhp_v, hp_a, nhp_a = _compute_step_metrics(info)
                    detail_step = _compute_detail_step_metrics(info)
                    detail_step_final = _finalize_detail_stats(detail_step)
                    ep_hp_dist_sum += hp_d
                    ep_nhp_dist_sum += nhp_d
                    ep_hp_viol_sum += hp_v
                    ep_nhp_viol_sum += nhp_v
                    ep_hp_active += hp_a
                    ep_nhp_active += nhp_a
                    _merge_detail_stats(ep_detail_stats, detail_step)
                    # 归一化后的 step-level 值（用于 NPZ，可视化 Fig.3/4）
                    ep_step_hp_dists.append(hp_d / hp_a if hp_a > 0 else 0.0)
                    ep_step_nhp_dists.append(nhp_d / nhp_a if nhp_a > 0 else 0.0)
                    seed_step_mean_margins.append(detail_step_final["mean_intent_margin"])
                    seed_step_severities.append(detail_step_final["violation_severity"])
                    for metric in ("thr", "lat", "rel"):
                        seed_step_qos[metric]["viol"].append(detail_step_final[f"{metric}_viol"])
                        seed_step_qos[metric]["margin"].append(detail_step_final[f"{metric}_margin"])
                        seed_step_qos[metric]["severity"].append(detail_step_final[f"{metric}_severity"])

                    if dones[0]:
                        # 归一化：sum / (active_slices × n_steps)，与 metric_value.py 一致
                        hp_dist_ep  = ep_hp_dist_sum  / ep_hp_active  if ep_hp_active  > 0 else 0.0
                        nhp_dist_ep = ep_nhp_dist_sum / ep_nhp_active if ep_nhp_active > 0 else 0.0
                        hp_viol_ep  = ep_hp_viol_sum  / ep_hp_active  if ep_hp_active  > 0 else 0.0
                        nhp_viol_ep = ep_nhp_viol_sum / ep_nhp_active if ep_nhp_active > 0 else 0.0
                        total_active = ep_hp_active + ep_nhp_active
                        total_viol_ep = (
                            (ep_hp_viol_sum + ep_nhp_viol_sum) / total_active
                            if total_active > 0 else 0.0
                        )
                        detail_ep = _finalize_detail_stats(ep_detail_stats)

                        ep_rewards.append(float(current_ep_reward))
                        ep_hp_viols.append(float(hp_viol_ep))
                        ep_nhp_viols.append(float(nhp_viol_ep))
                        ep_hp_dists.append(float(hp_dist_ep))
                        ep_nhp_dists.append(float(nhp_dist_ep))
                        ep_extra["total_viol"].append(float(total_viol_ep))
                        for metric in MVP_EXTRA_METRICS:
                            if metric != "total_viol":
                                ep_extra[metric].append(float(detail_ep[metric]))
                        seed_step_hp_dists.extend(ep_step_hp_dists)
                        seed_step_nhp_dists.extend(ep_step_nhp_dists)
                        pbar.update(1)

                        # 重置 episode 累计量
                        current_ep_reward = 0.0
                        ep_hp_dist_sum = ep_nhp_dist_sum = 0.0
                        ep_hp_viol_sum = ep_nhp_viol_sum = 0
                        ep_hp_active = ep_nhp_active = 0
                        ep_detail_stats = _new_detail_stats()
                        ep_step_hp_dists, ep_step_nhp_dists = [], []
            except KeyboardInterrupt:
                print("  Interrupted.")
            finally:
                env.close()
                pbar.close()

            mean_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            mean_hp = float(np.mean(ep_hp_viols)) if ep_hp_viols else 0.0
            mean_nhp = float(np.mean(ep_nhp_viols)) if ep_nhp_viols else 0.0
            mean_hp_dist = float(np.mean(ep_hp_dists)) if ep_hp_dists else 0.0
            mean_nhp_dist = float(np.mean(ep_nhp_dists)) if ep_nhp_dists else 0.0
            mean_extra = {
                metric: float(np.mean(values)) if values else 0.0
                for metric, values in ep_extra.items()
            }
            print(f"  Seed {seed}: Reward={mean_r:.2f}, HP_Viol={mean_hp:.2f}, "
                  f"HP_Dist={mean_hp_dist:.4f}, NHP_Dist={mean_nhp_dist:.4f}, "
                  f"Total_Viol={mean_extra['total_viol']:.2f}")

            seed_rewards.append(mean_r)
            seed_hp_viols.append(mean_hp)
            seed_nhp_viols.append(mean_nhp)
            seed_hp_dists.append(mean_hp_dist)
            seed_nhp_dists.append(mean_nhp_dist)
            for metric in MVP_EXTRA_METRICS:
                seed_extra[metric].append(mean_extra[metric])

            if save_results:
                json_dir = os.path.join(save_root, "metric_json",
                                        f"scenario_{scenario_id}", f"seed_{seed}")
                _save_json({"hp_violations": ep_hp_viols, "mean": mean_hp},
                           os.path.join(json_dir, "hp_violations.json"))
                _save_json({"nhp_violations": ep_nhp_viols, "mean": mean_nhp},
                           os.path.join(json_dir, "nhp_violations.json"))
                _save_json({"rewards": ep_rewards, "mean": mean_r},
                           os.path.join(json_dir, "episode_rewards.json"))
                _save_json({"hp_distance": ep_hp_dists, "mean": mean_hp_dist},
                           os.path.join(json_dir, "hp_distance.json"))
                _save_json({"nhp_distance": ep_nhp_dists, "mean": mean_nhp_dist},
                           os.path.join(json_dir, "nhp_distance.json"))
                for metric in MVP_EXTRA_METRICS:
                    _save_json({metric: ep_extra[metric], "mean": mean_extra[metric]},
                               os.path.join(json_dir, f"{metric}.json"))
                summary = {
                    "scenario_id": scenario_id, "seed": seed,
                    "reward_mean": mean_r, "hp_viol_mean": mean_hp, "nhp_viol_mean": mean_nhp,
                    "hp_dist_mean": mean_hp_dist, "nhp_dist_mean": mean_nhp_dist,
                }
                for metric in MVP_EXTRA_METRICS:
                    summary[f"{metric}_mean"] = mean_extra[metric]
                _save_json(summary, os.path.join(json_dir, "summary.json"))

                # step-level NPZ for Fig.3/4
                raw_dir = os.path.join(save_root, "metric_raw", f"scenario_{scenario_id}")
                os.makedirs(raw_dir, exist_ok=True)
                np.savez(
                    os.path.join(raw_dir, f"ep_seed{seed}.npz"),
                    ep_rewards=np.array(ep_rewards),
                    hp_viols=np.array(ep_hp_viols),
                    nhp_viols=np.array(ep_nhp_viols),
                    step_hp_dist=np.array(seed_step_hp_dists),
                    step_nhp_dist=np.array(seed_step_nhp_dists),
                    total_viols=np.array(ep_extra["total_viol"]),
                    mean_intent_margins=np.array(ep_extra["mean_intent_margin"]),
                    violation_severities=np.array(ep_extra["violation_severity"]),
                    step_mean_intent_margin=np.array(seed_step_mean_margins),
                    step_violation_severity=np.array(seed_step_severities),
                    step_thr_viol=np.array(seed_step_qos["thr"]["viol"]),
                    step_lat_viol=np.array(seed_step_qos["lat"]["viol"]),
                    step_rel_viol=np.array(seed_step_qos["rel"]["viol"]),
                    step_thr_margin=np.array(seed_step_qos["thr"]["margin"]),
                    step_lat_margin=np.array(seed_step_qos["lat"]["margin"]),
                    step_rel_margin=np.array(seed_step_qos["rel"]["margin"]),
                    step_thr_severity=np.array(seed_step_qos["thr"]["severity"]),
                    step_lat_severity=np.array(seed_step_qos["lat"]["severity"]),
                    step_rel_severity=np.array(seed_step_qos["rel"]["severity"]),
                )

        scenario_summary = {
            "scenario_id": scenario_id, "seeds": test_seeds,
            "reward_mean": float(np.mean(seed_rewards)),
            "reward_std": float(np.std(seed_rewards)),
            "hp_viol_mean": float(np.mean(seed_hp_viols)),
            "hp_viol_std": float(np.std(seed_hp_viols)),
            "nhp_viol_mean": float(np.mean(seed_nhp_viols)),
            "nhp_viol_std": float(np.std(seed_nhp_viols)),
            "hp_dist_mean": float(np.mean(seed_hp_dists)),
            "hp_dist_std": float(np.std(seed_hp_dists)),
            "nhp_dist_mean": float(np.mean(seed_nhp_dists)),
            "nhp_dist_std": float(np.std(seed_nhp_dists)),
        }
        for metric in MVP_EXTRA_METRICS:
            scenario_summary[f"{metric}_mean"] = float(np.mean(seed_extra[metric]))
            scenario_summary[f"{metric}_std"] = float(np.std(seed_extra[metric]))
        all_scenario_results[f"scenario_{scenario_id}"] = scenario_summary
        print(f"\nScenario {scenario_id}: Reward={scenario_summary['reward_mean']:.2f}±{scenario_summary['reward_std']:.2f}, "
              f"HP_Dist={scenario_summary['hp_dist_mean']:.4f}±{scenario_summary['hp_dist_std']:.4f}")

        if save_results:
            _save_json(scenario_summary,
                       os.path.join(save_root, "metric_json", f"scenario_{scenario_id}", "summary.json"))

    print(f"\n=== PPO-HA Weighted (Discrete Teacher) Test Complete ===")
    return all_scenario_results


if __name__ == "__main__":
    pass