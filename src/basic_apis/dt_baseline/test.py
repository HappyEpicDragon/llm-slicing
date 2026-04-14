import os
import torch
import numpy as np
import json
import ray
from tqdm import tqdm
from omegaconf import OmegaConf

# === Local Imports ===
from src.basic_apis.dt_baseline.model_baseline_dt import DecisionTransformerBaseline, BaselineStateEncoder
from src.basic_apis.network_slicing_business.path_manager import PathManager
from src.basic_apis.ppo.ppo_baseline.env_ray import env_creator
from src.basic_apis.ppo.ppo_baseline.test_ppo_baseline import init_ray_and_register_env, register_only


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
    """
    flat_info = info if isinstance(info, dict) else {}
    hp_dist = nhp_dist = 0.0
    hp_viols = nhp_viols = hp_active = nhp_active = 0
    for s_idx in range(num_slices):
        is_hp = flat_info.get(f"meta/slice_{s_idx}_priority", 0) > 0
        drifts = [float(flat_info.get(f"drift/slice_{s_idx}_{met}", 0.0))
                  for met in ("thr", "rel", "lat")
                  if flat_info.get(f"meta/slice_{s_idx}_{met}_req", 0) > 0]
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


# ------------------------------------------------------------------------------
# 1. 辅助函数
# ------------------------------------------------------------------------------

def load_metadata(meta_path):
    """加载标准化统计量"""
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        mean = np.array(meta["obs_stats"]["flat"]["mean"], dtype=np.float32)
        std = np.array(meta["obs_stats"]["flat"]["std"], dtype=np.float32)
    return mean, std


def flatten_obs(obs_d):
    """
    将 MARL Dict Observation 展平 (需与 Dataset 收集逻辑完全一致)
    """
    flat = []
    # Player 0
    if "player_0" in obs_d:
        p0 = obs_d["player_0"]
        # 兼容处理：有些环境返回的是 Dict，有些可能是 Array
        val = p0["observations"] if isinstance(p0, dict) else p0
        flat.append(val)

    # Player 1-5
    for i in range(1, 6):
        key = f"player_{i}"
        if key in obs_d:
            p = obs_d[key]
            val = p["observations"] if isinstance(p, dict) else p
            flat.append(val)

    return np.concatenate(flat, axis=0).astype(np.float32)


def make_env(cfg, path_manager, seed=0, scenario_id=None):
    """创建 Ray 兼容的评估环境"""
    env_config = OmegaConf.to_container(cfg.environment, resolve=True)

    updates = cfg.env_updates
    mode = updates.mode  # usually 'testing'
    scenario_mode = updates.scenario_mode

    env_config['mode'] = mode
    env_config['scenario_mode'] = scenario_mode
    env_config['model_name'] = updates.model_name

    env_config["path_manager"] = path_manager
    env_config['seed'] = seed
    env_config['seed_test'] = seed

    if scenario_id is not None:
        env_config[scenario_mode][mode]['active_scenario_list'] = [scenario_id]
        env_config['state_config']['active_scenario_list'] = [scenario_id]
    else:
        target_list = updates[scenario_mode][mode].active_scenario_list
        env_config[scenario_mode][mode]['active_scenario_list'] = target_list
        env_config['state_config']['active_scenario_list'] = target_list

    return env_creator(env_config, only_env=True)


# ------------------------------------------------------------------------------
# 2. 核心测试逻辑
# ------------------------------------------------------------------------------

def _run_single_episode(env, model, obs_mean, obs_std, context_len, target_rtg, rtg_scale, device):
    """运行单个 episode，返回 (episode_reward, hp_viol, nhp_viol, step_hp_dists, step_nhp_dists)"""

    def reset_history():
        dummy_action = torch.zeros(10, device=device)
        return {"obs": [], "actions": [dummy_action], "rtg": [], "timesteps": []}

    def stack_pad(lst):
        seq = torch.stack(lst)
        curr_l = seq.shape[0]
        if curr_l < context_len:
            pad_len = context_len - curr_l
            shape = [pad_len] + list(seq.shape[1:])
            pad = torch.zeros(shape, dtype=seq.dtype, device=device)
            seq = torch.cat([pad, seq], dim=0)
        if seq.ndim == 1:
            seq = seq.unsqueeze(-1)
        return seq.unsqueeze(0)

    history = reset_history()
    obs_dict, _ = env.reset()
    cumulative_reward = 0.0
    ep_reward = 0.0
    curr_step = 0
    ep_hp_dist_sum = ep_nhp_dist_sum = 0.0
    ep_hp_viol_sum = ep_nhp_viol_sum = 0
    ep_hp_active = ep_nhp_active = 0
    step_hp_dists = []
    step_nhp_dists = []
    done = False

    while not done:
        with torch.no_grad():
            flat_obs = flatten_obs(obs_dict)
            norm_obs = np.clip((flat_obs - obs_mean) / (obs_std + 1e-6), -5.0, 5.0)
            t_obs = torch.from_numpy(norm_obs).float().to(device)
            history["obs"].append(t_obs)

            current_rtg_raw = target_rtg - cumulative_reward
            # RTG SymLog 变换（与 Dataset 训练时一致：只做 sym-log，不除以 rtg_scale）
            rtg_val = np.sign(current_rtg_raw) * np.log1p(np.abs(current_rtg_raw))
            t_rtg = torch.tensor([rtg_val], dtype=torch.float32, device=device)
            history["rtg"].append(t_rtg)

            t_step = torch.tensor([curr_step], dtype=torch.long, device=device)
            history["timesteps"].append(t_step)

            if len(history["obs"]) > context_len:
                for k in history:
                    history[k].pop(0)

            in_obs = stack_pad(history["obs"])
            in_rtg = stack_pad(history["rtg"])
            in_time = stack_pad(history["timesteps"]).squeeze(-1)

            act_seq = torch.stack(history["actions"])
            if act_seq.shape[0] < context_len:
                pad = torch.zeros((context_len - act_seq.shape[0], 10), device=device)
                act_seq = torch.cat([pad, act_seq], dim=0)
            in_act = act_seq.unsqueeze(0)

            real_len = min(len(history["obs"]), context_len)
            mask = torch.zeros((1, context_len), device=device)
            mask[0, -real_len:] = 1.0

            p_cont, p_disc_logits = model(in_obs, in_act, in_rtg, in_time, mask)
            last_cont = p_cont[0, -1, :]
            last_logits = p_disc_logits[0, -1]
            last_disc = torch.argmax(last_logits, dim=1).float()

            action_dict = {"player_0": last_cont.cpu().numpy()}
            disc_np = last_disc.cpu().numpy().astype(int)
            for i in range(5):
                action_dict[f"player_{i + 1}"] = disc_np[i]

            next_obs_dict, rewards, terminated, truncated, infos = env.step(action_dict)

            hybrid_act = torch.cat([last_cont, last_disc], dim=0)
            history["actions"].append(hybrid_act)

            r = rewards.get("player_0", 0.0)
            cumulative_reward += r
            ep_reward += r

            # 逐步累计 distance 和 violation（与 metric_value.py 对齐）
            step_info = infos.get("player_0", infos) if isinstance(infos, dict) else {}
            hp_d, nhp_d, hp_v, nhp_v, hp_a, nhp_a = _compute_step_metrics(step_info)
            ep_hp_dist_sum += hp_d
            ep_nhp_dist_sum += nhp_d
            ep_hp_viol_sum += hp_v
            ep_nhp_viol_sum += nhp_v
            ep_hp_active += hp_a
            ep_nhp_active += nhp_a
            step_hp_dists.append(hp_d / hp_a if hp_a > 0 else 0.0)
            step_nhp_dists.append(nhp_d / nhp_a if nhp_a > 0 else 0.0)

            obs_dict = next_obs_dict
            curr_step += 1
            done = terminated.get("__all__", False) or truncated.get("__all__", False)

    hp_dist_ep  = ep_hp_dist_sum  / ep_hp_active  if ep_hp_active  > 0 else 0.0
    nhp_dist_ep = ep_nhp_dist_sum / ep_nhp_active if ep_nhp_active > 0 else 0.0
    hp_viol_ep  = ep_hp_viol_sum  / ep_hp_active  if ep_hp_active  > 0 else 0.0
    nhp_viol_ep = ep_nhp_viol_sum / ep_nhp_active if ep_nhp_active > 0 else 0.0
    return ep_reward, hp_viol_ep, nhp_viol_ep, hp_dist_ep, nhp_dist_ep, step_hp_dists, step_nhp_dists


def test_dt_baseline(cfg, path_manager):
    """DT-Baseline 测试入口，支持 5 seeds × N 场景，保存标准 JSON。"""
    dt_cfg = cfg.train_dt if 'train_dt' in cfg else cfg
    device = torch.device("cpu")
    print(f"DT-Baseline Testing Device: {device}")

    base_dir = dt_cfg.dataset.path
    meta_path = os.path.join(base_dir, "metadata.json")
    model_path = cfg.model_path

    print(f"Loading Metadata from {meta_path}")
    obs_mean, obs_std = load_metadata(meta_path)
    obs_dim = obs_mean.shape[0]

    if not ray.is_initialized():
        ray.init(local_mode=True, ignore_reinit_error=True)
        register_only()

    # 读取多种子配置
    test_seeds = list(cfg.get('test_seeds', [0, 1, 2, 3, 4]))
    save_results = bool(cfg.get('save_results', True))
    save_root = str(cfg.get('save_root', 'data/channel_generality/dt_baseline'))

    # 确定场景列表
    scenario_mode = cfg.env_updates.scenario_mode
    mode = cfg.env_updates.mode
    scenario_list = list(cfg.env_updates[scenario_mode][mode].active_scenario_list)

    # 计算每个 episode 数量
    test_env_cfg = cfg.environment[scenario_mode][mode]
    n_episodes = test_env_cfg.max_episode - test_env_cfg.initial_episode

    print(f"Loading Weights from: {model_path}")
    state_encoder = BaselineStateEncoder(input_dim=obs_dim, embed_dim=dt_cfg.model.embed_dim)
    model = DecisionTransformerBaseline(
        state_encoder=state_encoder,
        state_dim=obs_dim,
        hidden_size=dt_cfg.model.embed_dim,
        max_length=dt_cfg.model.context_len,
        act_dim_cont=5,
        act_num_discrete=5,
        act_discrete_card=3
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    context_len = dt_cfg.model.context_len
    target_rtg = dt_cfg.evaluation.target_rtg
    rtg_scale = dt_cfg.dataset.rtg_scale

    print(f"Test Seeds: {test_seeds}, Scenarios: {scenario_list}")

    all_scenario_results = {}

    for scenario_id in scenario_list:
        print(f"\n{'='*60}\nEvaluating Scenario {scenario_id}\n{'='*60}")
        seed_rewards, seed_hp_viols, seed_nhp_viols = [], [], []
        seed_hp_dists, seed_nhp_dists = [], []

        for seed in test_seeds:
            print(f"  Seed {seed} / Scenario {scenario_id}")
            np.random.seed(seed)

            env = make_env(cfg, path_manager, seed=seed, scenario_id=scenario_id)

            ep_rewards, ep_hp_viols, ep_nhp_viols = [], [], []
            ep_hp_dists, ep_nhp_dists = [], []
            seed_step_hp_dists, seed_step_nhp_dists = [], []
            pbar = tqdm(total=n_episodes, unit="ep", desc=f"S{scenario_id} seed{seed}")

            try:
                for _ in range(n_episodes):
                    ep_r, hp_viol_ep, nhp_viol_ep, hp_dist_ep, nhp_dist_ep, step_hp, step_nhp = \
                        _run_single_episode(
                            env, model, obs_mean, obs_std, context_len, target_rtg, rtg_scale, device
                        )
                    ep_rewards.append(ep_r)
                    ep_hp_viols.append(float(hp_viol_ep))
                    ep_nhp_viols.append(float(nhp_viol_ep))
                    ep_hp_dists.append(float(hp_dist_ep))
                    ep_nhp_dists.append(float(nhp_dist_ep))
                    seed_step_hp_dists.extend(step_hp)
                    seed_step_nhp_dists.extend(step_nhp)
                    pbar.update(1)
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
            print(f"  Seed {seed}: Reward={mean_r:.2f}, HP_Viol={mean_hp:.4f}, "
                  f"HP_Dist={mean_hp_dist:.4f}, NHP_Dist={mean_nhp_dist:.4f}")

            seed_rewards.append(mean_r)
            seed_hp_viols.append(mean_hp)
            seed_nhp_viols.append(mean_nhp)
            seed_hp_dists.append(mean_hp_dist)
            seed_nhp_dists.append(mean_nhp_dist)

            if save_results:
                json_dir = os.path.join(save_root, "metric_json", f"scenario_{scenario_id}", f"seed_{seed}")
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
                )

        scenario_summary = {
            "scenario_id": scenario_id,
            "seeds": test_seeds,
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
            "per_seed_hp_dists": seed_hp_dists,
            "per_seed_nhp_dists": seed_nhp_dists,
        }
        all_scenario_results[f"scenario_{scenario_id}"] = scenario_summary
        print(f"\nScenario {scenario_id}: Reward={scenario_summary['reward_mean']:.2f}±{scenario_summary['reward_std']:.2f}, "
              f"HP_Dist={scenario_summary['hp_dist_mean']:.4f}±{scenario_summary['hp_dist_std']:.4f}")

        if save_results:
            _save_json(scenario_summary,
                       os.path.join(save_root, "metric_json", f"scenario_{scenario_id}", "summary.json"))

    if ray.is_initialized():
        ray.shutdown()

    print(f"\n=== DT-Baseline Test Complete ===")
    return all_scenario_results


if __name__ == "__main__":
    pass