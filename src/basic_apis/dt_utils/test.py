import os
import json
import shutil
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv

# === Local Imports ===
from src.basic_apis.dt_utils.model_ha_dt import HierarchicalStateEncoder, build_decision_model
from src.basic_apis.dt_utils.action_dims import resolve_action_dims_from_env_cfg
from src.basic_apis.network_slicing_business.path_manager import PathManager
from src.basic_apis.ppo.ppo_ha_weighted.hierarchical_slicing_env import HierarchicalSlicingEnv
from src.basic_apis.physics_probe import PhysicsProbe
from src.basic_apis.general_utils import pad_stack_tensor


def make_env(cfg, path_manager, rank=0, seed=0):
    """DT 测试环境工厂"""
    def _init():
        env_config = cfg.env_settings.copy()
        env = HierarchicalSlicingEnv(env_config, np.random.default_rng(seed + rank), path_manager)
        return env
    return _init


def _infer_num_slices_from_info(info, default=5):
    if not isinstance(info, dict):
        return default
    slice_ids = set()
    for k in info.keys():
        if k.startswith("meta/slice_"):
            parts = k.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                slice_ids.add(int(parts[1]))
    return (max(slice_ids) + 1) if slice_ids else default


def _compute_step_metrics(info, num_slices=None):
    """
    与 metric_value.calc_intent_distance / calc_slice_violations 对齐的单步指标计算：
    - Distance: 每个活跃 slice 的 mean(drifts)（当 mean < 0 时计入），HP/NHP 分别求和
    - Violation: 每个活跃 slice 整体判断（mean(drifts) < 0 → 1次违约），HP/NHP 分别计数
    返回: (hp_dist, nhp_dist, hp_viols, nhp_viols, hp_active, nhp_active)
    """
    hp_dist = nhp_dist = 0.0
    hp_viols = nhp_viols = hp_active = nhp_active = 0
    resolved_num_slices = _infer_num_slices_from_info(info) if num_slices is None else num_slices
    for s_idx in range(resolved_num_slices):
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


def _load_obs_stats(meta_path):
    """Load observation normalization stats from metadata.json,
    consistent with HierarchicalDTDataset."""
    if not meta_path or not os.path.exists(meta_path):
        return None
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    obs_stats_raw = meta.get("obs_stats", meta)
    stats = {}
    for k in ['inter', 'intra', 'global']:
        if k in obs_stats_raw:
            stats[k] = {
                'mean': np.array(obs_stats_raw[k]['mean'], dtype=np.float32),
                'std': np.array(obs_stats_raw[k]['std'], dtype=np.float32) + 1e-6,
            }
    return stats if stats else None


def _normalize_obs(raw, feat_key, obs_stats):
    """Normalize a raw observation array, matching HierarchicalDTDataset logic."""
    if obs_stats is None:
        return raw
    stat_key = feat_key.split('_')[0]
    if stat_key not in obs_stats:
        return raw
    mean = obs_stats[stat_key]['mean']
    std = obs_stats[stat_key]['std']
    safe_std = std.copy()
    safe_std[safe_std < 1e-2] = 1.0
    normed = (raw - mean) / safe_std
    return np.clip(normed, -5.0, 5.0).astype(np.float32)


def _compute_viol_reward_from_obs(obs_raw, hp_weight=10.0, nhp_weight=1.0):
    """从 raw obs 的 inter_feat 计算 violation-based reward"""
    inter = obs_raw["inter_feat"]
    if inter.ndim == 3:
        inter = inter[0]
    vr = 0.0
    for s in range(inter.shape[0]):
        drift = float(inter[s, 1])
        if drift < 0:
            w = hp_weight if inter[s, 0] > 0 else nhp_weight
            vr += drift * w
    return vr


def _run_single_episode(
    env, model, device, target_rtg, rtg_scale, context_len, action_dims,
    obs_stats=None, force_dim0=None, force_intra=None, use_viol_rtg=False,
):
    """运行单个 episode，返回 (ep_reward, step_rewards, ep_hp_dist, ep_nhp_dist,
                                ep_hp_viol, ep_nhp_viol, ep_hp_active, ep_nhp_active,
                                step_hp_dists, step_nhp_dists)
    step_hp/nhp_dists 为归一化后的 per-step 值（供 NPZ 保存）。
    """

    def reset_buffers():
        dummy_action = torch.zeros(len(action_dims), dtype=torch.long, device=device)
        return {
            "inter": [], "intra": [], "global": [],
            "actions": [dummy_action],
            "rtg": [], "timesteps": []
        }

    obs = env.reset()
    history = reset_buffers()
    cumulative_reward = 0.0
    current_ep_reward = 0.0
    curr_step = 0
    step_rewards = []
    step_hp_dists = []
    step_nhp_dists = []
    ep_hp_dist_sum = ep_nhp_dist_sum = 0.0
    ep_hp_viol_sum = ep_nhp_viol_sum = 0
    ep_hp_active = ep_nhp_active = 0
    done = False

    while not done:
        with torch.no_grad():
            t_inter = torch.from_numpy(_normalize_obs(obs['inter_feat'][0], 'inter_feat', obs_stats)).float().to(device)
            t_intra = torch.from_numpy(_normalize_obs(obs['intra_feat'][0], 'intra_feat', obs_stats)).float().to(device)
            t_glob = torch.from_numpy(_normalize_obs(obs['global_feat'][0], 'global_feat', obs_stats)).float().to(device)

            history["inter"].append(t_inter)
            history["intra"].append(t_intra)
            history["global"].append(t_glob)

            # RTG SymLog 变换（与 Dataset 训练时一致：只做 sym-log，不除以 rtg_scale）
            current_rtg_raw = target_rtg - cumulative_reward
            rtg_input_val = np.sign(current_rtg_raw) * np.log1p(np.abs(current_rtg_raw))
            t_rtg = torch.tensor([rtg_input_val], dtype=torch.float32, device=device)
            history["rtg"].append(t_rtg)

            t_step = torch.tensor([curr_step], dtype=torch.long, device=device)
            history["timesteps"].append(t_step)

            if len(history["rtg"]) > context_len:
                for k in history:
                    history[k].pop(0)

            in_inter = pad_stack_tensor(history["inter"], context_len, None, device)
            in_intra = pad_stack_tensor(history["intra"], context_len, None, device)
            in_global = pad_stack_tensor(history["global"], context_len, 2, device)
            in_rtg = pad_stack_tensor(history["rtg"], context_len, 1, device)
            in_steps = pad_stack_tensor(history["timesteps"], context_len, 1, device).squeeze(-1).long()

            act_seq = torch.stack(history["actions"])
            if act_seq.shape[0] < context_len:
                pad_len = context_len - act_seq.shape[0]
                pad = torch.zeros((pad_len, len(action_dims)), dtype=torch.long, device=device)
                act_seq = torch.cat([pad, act_seq], dim=0)
            in_actions = act_seq.unsqueeze(0)

            states_in = {'inter': in_inter, 'intra': in_intra, 'global': in_global}

            real_len = min(len(history["rtg"]), context_len)
            mask = torch.zeros((1, context_len), device=device)
            mask[0, -real_len:] = 1.0

            action_preds = model(
                states=states_in,
                actions=in_actions,
                returns=in_rtg,
                timesteps=in_steps,
                attention_mask=mask
            )

            last_step_logits = action_preds[0, -1, :]
            pred_actions = []
            start_idx = 0
            for dim_size in action_dims:
                end_idx = start_idx + dim_size
                dim_act = torch.argmax(last_step_logits[start_idx:end_idx]).item()
                pred_actions.append(dim_act)
                start_idx = end_idx

            if force_dim0 is not None:
                pred_actions[0] = force_dim0
            if force_intra is not None:
                for i in range(1, len(pred_actions)):
                    pred_actions[i] = force_intra

            vec_action = np.array(pred_actions, dtype=np.int32)
            env_action = np.expand_dims(vec_action, axis=0)
            next_obs, rewards, dones, infos = env.step(env_action)

            action_tensor = torch.tensor(vec_action, dtype=torch.long, device=device)
            history["actions"].append(action_tensor)

            reward = float(rewards[0])
            current_ep_reward += reward
            step_rewards.append(reward)

            if use_viol_rtg:
                viol_r = _compute_viol_reward_from_obs(next_obs)
                cumulative_reward += viol_r
            else:
                cumulative_reward += reward

            # 逐步累计 distance 和 violation（与 metric_value.py 对齐）
            step_info = infos[0] if isinstance(infos, (list, tuple)) else infos
            hp_d, nhp_d, hp_v, nhp_v, hp_a, nhp_a = _compute_step_metrics(step_info)
            ep_hp_dist_sum += hp_d
            ep_nhp_dist_sum += nhp_d
            ep_hp_viol_sum += hp_v
            ep_nhp_viol_sum += nhp_v
            ep_hp_active += hp_a
            ep_nhp_active += nhp_a
            step_hp_dists.append(hp_d / hp_a if hp_a > 0 else 0.0)
            step_nhp_dists.append(nhp_d / nhp_a if nhp_a > 0 else 0.0)

            obs = next_obs
            curr_step += 1

            if dones[0]:
                done = True

    return (current_ep_reward, step_rewards,
            ep_hp_dist_sum, ep_nhp_dist_sum,
            ep_hp_viol_sum, ep_nhp_viol_sum,
            ep_hp_active, ep_nhp_active,
            step_hp_dists, step_nhp_dists)


def test_dt_process(cfg, path_manager):
    """
    DT 测试主入口。
    支持多种子循环（test_seeds）和多场景循环（active_scenario_list）。
    结果以标准 JSON + NPZ 格式保存，供绘图脚本自动发现。
    """
    dt_cfg = cfg
    if 'train_dt' in cfg:
        dt_cfg = cfg.train_dt

    device_str = cfg.get("device", "cpu")
    device = torch.device(device_str)

    torch.set_num_threads(2)
    os.environ.setdefault("OMP_NUM_THREADS", "2")

    print(f"DT Testing Device: {device}")

    model_path = cfg.model_path if 'model_path' in cfg else dt_cfg.saving.save_path + "/best_dt_model.pth"
    target_rtg = dt_cfg.evaluation.target_rtg if 'target_rtg' in dt_cfg.evaluation else -60000.0
    rtg_scale = dt_cfg.dataset.rtg_scale
    context_len = dt_cfg.model.context_len

    # 多种子配置
    test_seeds = list(cfg.get('test_seeds', [42]))
    save_results = bool(cfg.get('save_results', False))

    # 诊断实验用：强制覆盖动作维度
    force_dim0_raw = cfg.get('force_dim0', None)
    force_dim0 = int(force_dim0_raw) if force_dim0_raw is not None else None
    force_intra_raw = cfg.get('force_intra', None)
    force_intra = int(force_intra_raw) if force_intra_raw is not None else None
    use_viol_rtg = bool(cfg.get('use_viol_rtg', False))

    print(f"Target RTG (Raw): {target_rtg}")
    print(f"Loading Model from: {model_path}")
    print(f"Test Seeds: {test_seeds}")
    if force_dim0 is not None:
        print(f"⚠️  Force Dim0 = act{force_dim0} (diagnostic mode)")
    if force_intra is not None:
        names = {0: "RR", 1: "PF", 2: "MT"}
        print(f"⚠️  Force Intra = {names.get(force_intra, force_intra)} (diagnostic mode)")
    if use_viol_rtg:
        print(f"⚠️  Violation-based RTG mode (target_rtg=0 recommended)")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DT model not found: {model_path}")

    # 确定场景列表
    env_updates = cfg.env_updates
    mode = env_updates.mode
    scenario_mode = env_updates.scenario_mode
    model_name = env_updates.model_name

    scenario_list = list(env_updates[scenario_mode][mode].active_scenario_list)
    print(f"Scenario List: {scenario_list}")

    # 结果保存根目录
    save_root = cfg.get('save_root', 'data/channel_generality/dt')
    clean_before_save = bool(cfg.get('clean_before_save', False))
    if save_results and clean_before_save:
        for subdir in ("metric_json", "metric_raw"):
            old_dir = os.path.join(save_root, subdir)
            if os.path.exists(old_dir):
                shutil.rmtree(old_dir)
        print(f"[Asset] cleaned old metrics under: {save_root}")

    # 加载模型：从环境配置动态推导 action dims，避免硬编码 M=11。
    action_dims = resolve_action_dims_from_env_cfg(cfg.environment, intra_mode_count=3)

    state_encoder = HierarchicalStateEncoder(embed_dim=dt_cfg.model.embed_dim)
    model = build_decision_model(
        model_cfg=dt_cfg.model,
        state_encoder=state_encoder,
        action_dims=action_dims,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载观测归一化统计量（与训练时 HierarchicalDTDataset 保持一致）
    meta_path = dt_cfg.dataset.get('meta_path', None) if 'dataset' in dt_cfg else None
    obs_stats = _load_obs_stats(meta_path)
    if obs_stats:
        print(f"[Norm] Loaded obs normalization stats from: {meta_path}")
    else:
        print("[Norm] WARNING: No obs normalization stats loaded. Inputs will NOT be normalized.")

    # =====================================================================
    # 外层：场景循环（M3）
    # =====================================================================
    all_scenario_results = {}

    for scenario_id in scenario_list:
        print(f"\n{'='*60}")
        print(f"Evaluating Scenario {scenario_id}")
        print(f"{'='*60}")

        # 为当前场景创建独立的环境配置
        cfg.environment.env_settings.mode = mode
        cfg.environment.env_settings.scenario_mode = scenario_mode
        cfg.environment.env_settings.model_name = model_name
        cfg.environment.env_settings[scenario_mode][mode].active_scenario_list = [scenario_id]

        scenario_seed_rewards = []
        scenario_hp_viols = []
        scenario_nhp_viols = []
        scenario_hp_dists = []
        scenario_nhp_dists = []
        scenario_step_rewards_all = []  # 每个 seed 的 step-level rewards

        # =====================================================================
        # 内层：seed 循环（M1）
        # =====================================================================
        for seed in test_seeds:
            print(f"\n  Seed {seed} / Scenario {scenario_id}")
            np.random.seed(seed)

            env = DummyVecEnv([make_env(cfg.environment, path_manager, rank=0, seed=seed)])

            # 读取 episode 数量
            scenario_mode_local = cfg.environment.env_settings.scenario_mode
            test_env_cfg = cfg.environment.env_settings[scenario_mode_local][mode]
            n_scenarios = test_env_cfg.max_scenario_episodes - test_env_cfg.init_scenario_episode

            ep_rewards = []
            ep_hp_viols = []
            ep_nhp_viols = []
            ep_hp_dists = []
            ep_nhp_dists = []
            seed_step_rewards = []
            seed_step_hp_dists = []
            seed_step_nhp_dists = []

            pbar = tqdm(total=n_scenarios, unit="ep", desc=f"S{scenario_id} seed{seed}")
            episodes_done = 0

            try:
                while episodes_done < n_scenarios:
                    (ep_reward, step_rewards,
                     hp_dist_sum, nhp_dist_sum,
                     hp_viol_sum, nhp_viol_sum,
                     hp_active, nhp_active,
                     step_hp_dists, step_nhp_dists) = _run_single_episode(
                        env, model, device, target_rtg, rtg_scale, context_len, action_dims,
                        obs_stats=obs_stats, force_dim0=force_dim0,
                        force_intra=force_intra, use_viol_rtg=use_viol_rtg,
                    )

                    # 归一化（与 metric_value.py 一致）
                    hp_dist_ep  = hp_dist_sum  / hp_active  if hp_active  > 0 else 0.0
                    nhp_dist_ep = nhp_dist_sum / nhp_active if nhp_active > 0 else 0.0
                    hp_viol_ep  = hp_viol_sum  / hp_active  if hp_active  > 0 else 0.0
                    nhp_viol_ep = nhp_viol_sum / nhp_active if nhp_active > 0 else 0.0

                    ep_rewards.append(ep_reward)
                    ep_hp_viols.append(float(hp_viol_ep))
                    ep_nhp_viols.append(float(nhp_viol_ep))
                    ep_hp_dists.append(float(hp_dist_ep))
                    ep_nhp_dists.append(float(nhp_dist_ep))
                    seed_step_rewards.extend(step_rewards)
                    seed_step_hp_dists.extend(step_hp_dists)
                    seed_step_nhp_dists.extend(step_nhp_dists)

                    episodes_done += 1
                    pbar.update(1)

            except KeyboardInterrupt:
                print("  Interrupted.")
            finally:
                env.close()
                pbar.close()

            mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            mean_hp = float(np.mean(ep_hp_viols)) if ep_hp_viols else 0.0
            mean_nhp = float(np.mean(ep_nhp_viols)) if ep_nhp_viols else 0.0
            mean_hp_dist = float(np.mean(ep_hp_dists)) if ep_hp_dists else 0.0
            mean_nhp_dist = float(np.mean(ep_nhp_dists)) if ep_nhp_dists else 0.0
            print(f"  Seed {seed}: Avg Reward={mean_reward:.2f}, HP_Viol={mean_hp:.2f}, "
                  f"HP_Dist={mean_hp_dist:.4f}, NHP_Dist={mean_nhp_dist:.4f}")

            scenario_seed_rewards.append(mean_reward)
            scenario_hp_viols.append(mean_hp)
            scenario_nhp_viols.append(mean_nhp)
            scenario_hp_dists.append(mean_hp_dist)
            scenario_nhp_dists.append(mean_nhp_dist)
            scenario_step_rewards_all.append(seed_step_rewards)

            # === 保存 seed 级别的 metric_json ===
            if save_results:
                json_dir = os.path.join(save_root, "metric_json", f"scenario_{scenario_id}", f"seed_{seed}")
                _save_json(
                    {"hp_violations": ep_hp_viols, "mean": mean_hp, "episodes": len(ep_hp_viols)},
                    os.path.join(json_dir, "hp_violations.json")
                )
                _save_json(
                    {"nhp_violations": ep_nhp_viols, "mean": mean_nhp, "episodes": len(ep_nhp_viols)},
                    os.path.join(json_dir, "nhp_violations.json")
                )
                _save_json(
                    {"rewards": ep_rewards, "mean": mean_reward, "episodes": len(ep_rewards)},
                    os.path.join(json_dir, "episode_rewards.json")
                )
                _save_json(
                    {"hp_distance": ep_hp_dists, "mean": mean_hp_dist, "episodes": len(ep_hp_dists)},
                    os.path.join(json_dir, "hp_distance.json")
                )
                _save_json(
                    {"nhp_distance": ep_nhp_dists, "mean": mean_nhp_dist, "episodes": len(ep_nhp_dists)},
                    os.path.join(json_dir, "nhp_distance.json")
                )

                # step-level data (metric_raw) for Fig.3/4
                raw_dir = os.path.join(save_root, "metric_raw", f"scenario_{scenario_id}")
                os.makedirs(raw_dir, exist_ok=True)
                np.savez(
                    os.path.join(raw_dir, f"ep_seed{seed}.npz"),
                    step_rewards=np.array(seed_step_rewards),
                    ep_rewards=np.array(ep_rewards),
                    hp_viols=np.array(ep_hp_viols),
                    nhp_viols=np.array(ep_nhp_viols),
                    step_hp_dist=np.array(seed_step_hp_dists),
                    step_nhp_dist=np.array(seed_step_nhp_dists),
                )

        # === 场景汇总：mean ± std across seeds ===
        scenario_summary = {
            "scenario_id": scenario_id,
            "seeds": test_seeds,
            "reward_mean": float(np.mean(scenario_seed_rewards)),
            "reward_std": float(np.std(scenario_seed_rewards)),
            "hp_viol_mean": float(np.mean(scenario_hp_viols)),
            "hp_viol_std": float(np.std(scenario_hp_viols)),
            "nhp_viol_mean": float(np.mean(scenario_nhp_viols)),
            "nhp_viol_std": float(np.std(scenario_nhp_viols)),
            "hp_dist_mean": float(np.mean(scenario_hp_dists)),
            "hp_dist_std": float(np.std(scenario_hp_dists)),
            "nhp_dist_mean": float(np.mean(scenario_nhp_dists)),
            "nhp_dist_std": float(np.std(scenario_nhp_dists)),
            "per_seed_rewards": scenario_seed_rewards,
            "per_seed_hp_viols": scenario_hp_viols,
            "per_seed_nhp_viols": scenario_nhp_viols,
            "per_seed_hp_dists": scenario_hp_dists,
            "per_seed_nhp_dists": scenario_nhp_dists,
        }
        all_scenario_results[f"scenario_{scenario_id}"] = scenario_summary

        print(f"\nScenario {scenario_id} Summary:")
        print(f"  Reward:   {scenario_summary['reward_mean']:.2f} ± {scenario_summary['reward_std']:.2f}")
        print(f"  HP Viol:  {scenario_summary['hp_viol_mean']:.2f} ± {scenario_summary['hp_viol_std']:.2f}")
        print(f"  NHP Viol: {scenario_summary['nhp_viol_mean']:.2f} ± {scenario_summary['nhp_viol_std']:.2f}")
        print(f"  HP Dist:  {scenario_summary['hp_dist_mean']:.4f} ± {scenario_summary['hp_dist_std']:.4f}")
        print(f"  NHP Dist: {scenario_summary['nhp_dist_mean']:.4f} ± {scenario_summary['nhp_dist_std']:.4f}")

        if save_results:
            summary_dir = os.path.join(save_root, "metric_json", f"scenario_{scenario_id}")
            _save_json(scenario_summary, os.path.join(summary_dir, "summary.json"))

    # === 全局汇总 ===
    print(f"\n{'='*60}")
    print("=== Decision Transformer Test Complete ===")
    all_rewards = [v["reward_mean"] for v in all_scenario_results.values()]
    all_hp = [v["hp_viol_mean"] for v in all_scenario_results.values()]
    all_nhp = [v["nhp_viol_mean"] for v in all_scenario_results.values()]
    print(f"Avg Reward (all scenarios): {np.mean(all_rewards):.2f}")
    print(f"Avg HP Viol:  {np.mean(all_hp):.2f}")
    print(f"Avg NHP Viol: {np.mean(all_nhp):.2f}")

    if save_results:
        global_summary = {
            "scenarios": list(all_scenario_results.keys()),
            "overall_reward_mean": float(np.mean(all_rewards)),
            "overall_hp_viol_mean": float(np.mean(all_hp)),
            "overall_nhp_viol_mean": float(np.mean(all_nhp)),
            "per_scenario": all_scenario_results,
        }
        _save_json(global_summary, os.path.join(save_root, "metric_json", "global_summary.json"))
        print(f"Results saved to: {os.path.join(save_root, 'metric_json')}")

    return all_scenario_results


if __name__ == "__main__":
    pass
