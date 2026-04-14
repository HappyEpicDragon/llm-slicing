import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.basic_apis.explainable_ai_utils.xai_utils import make_env, load_models, resolve_project_path


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


def _count_hp_violations_from_env(env):
    """从 HierarchicalSlicingEnv 的最近 episode 统计 HP Violation 次数"""
    try:
        num_slices = env.num_slices
        priorities = np.array([env.slice_priorities[i] for i in range(num_slices)])
        # 从最后一次 step 的 info 计数（简化：读取 env 内部统计）
        # 使用 env.violation_counts 如果存在，否则返回 None
        if hasattr(env, 'episode_hp_viol_count'):
            return float(env.episode_hp_viol_count)
    except Exception:
        pass
    return None


def evaluate_ppo(model, env_cfg, pm, seeds, cfg_xai):
    """评估 PPO 作为 Baseline（水平参考线）"""
    rewards = []
    hp_viols = []
    print("Evaluating PPO Baseline...")
    for seed in seeds:
        env = make_env(env_cfg, pm, cfg_xai.scenario_id, seed)
        obs, _ = env.reset()
        done = False
        total_reward = 0
        ep_hp = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if isinstance(info, dict):
                for s_idx in range(_infer_num_slices_from_info(info)):
                    if info.get(f"meta/slice_{s_idx}_priority", 0) > 0:
                        for met in ("thr", "rel", "lat"):
                            if info.get(f"meta/slice_{s_idx}_{met}_req", 0) > 0:
                                if info.get(f"drift/slice_{s_idx}_{met}", 0.0) < 0:
                                    ep_hp += 1
        rewards.append(total_reward)
        hp_viols.append(ep_hp)
        env.close()
    return np.mean(rewards), np.std(rewards), np.mean(hp_viols), np.std(hp_viols)


def evaluate_dt_single_rtg(model, env_cfg, pm, target_rtg, seeds, cfg_xai):
    """评估 IDT 在特定 RTG 指令下的表现，同时返回 HP Violation 数量"""
    device = cfg_xai.device
    rewards = []
    hp_viols = []
    rtg_scale = float(getattr(cfg_xai, "rtg_scale", 100000.0))
    context_len = 20
    action_dims = model.action_dims
    dummy_action = torch.zeros(len(action_dims), dtype=torch.long, device=device)

    for seed in seeds:
        env = make_env(env_cfg, pm, cfg_xai.scenario_id, seed)
        obs, _ = env.reset()

        history = {
            "inter": [], "intra": [], "global": [],
            "actions": [dummy_action],
            "rtg": [], "timesteps": []
        }

        curr_rtg = target_rtg
        total_reward = 0
        ep_hp_viol = 0
        done = False
        curr_step = 0

        while not done:
            t_inter = torch.from_numpy(obs['inter_feat']).float().to(device)
            t_intra = torch.from_numpy(obs['intra_feat']).float().to(device)
            t_glob = torch.from_numpy(obs['global_feat']).float().to(device)

            history["inter"].append(t_inter)
            history["intra"].append(t_intra)
            history["global"].append(t_glob)

            # 与主训练/测试链路对齐：SymLog + rtg_scale
            rtg_input = np.sign(curr_rtg) * np.log1p(np.abs(curr_rtg)) / rtg_scale

            history["rtg"].append(torch.tensor([rtg_input], dtype=torch.float32, device=device))
            history["timesteps"].append(torch.tensor([curr_step], dtype=torch.long, device=device))

            if len(history["inter"]) > context_len:
                for k in ["inter", "intra", "global", "rtg", "timesteps", "actions"]:
                    history[k].pop(0)

            def stack_helper(lst):
                return torch.stack(lst).unsqueeze(0)  # [1, T, ...]

            states = {
                'inter': stack_helper(history['inter']),
                'intra': stack_helper(history['intra']),
                'global': stack_helper(history['global'])
            }
            actions_in = torch.stack(history['actions']).unsqueeze(0)
            if actions_in.shape[1] < states['inter'].shape[1]:
                actions_in = torch.cat(
                    [torch.zeros((1, 1, len(action_dims)), device=device), actions_in], dim=1
                )
            actions_in = actions_in[:, -states['inter'].shape[1]:, :]

            returns_in = stack_helper(history['rtg'])
            timesteps_in = stack_helper(history['timesteps']).squeeze(-1)

            B, K = states['inter'].shape[:2]
            mask = torch.ones((B, K), device=device)

            with torch.no_grad():
                preds = model(states, actions_in, returns_in, timesteps_in, mask)
                last_logits = preds[0, -1, :]
                pred_actions = []
                idx = 0
                for d in action_dims:
                    pred_actions.append(torch.argmax(last_logits[idx:idx + d]).item())
                    idx += d
                action = np.array(pred_actions, dtype=np.int32)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            curr_rtg -= reward

            # 统计 HP Violation
            if isinstance(info, dict):
                for s_idx in range(_infer_num_slices_from_info(info)):
                    if info.get(f"meta/slice_{s_idx}_priority", 0) > 0:
                        for met in ("thr", "rel", "lat"):
                            if info.get(f"meta/slice_{s_idx}_{met}_req", 0) > 0:
                                if info.get(f"drift/slice_{s_idx}_{met}", 0.0) < 0:
                                    ep_hp_viol += 1

            history["actions"].append(torch.tensor(action, dtype=torch.long, device=device))
            curr_step += 1

        rewards.append(total_reward)
        hp_viols.append(ep_hp_viol)
        env.close()

    return np.mean(rewards), np.std(rewards), np.mean(hp_viols), np.std(hp_viols)


def plot_sweep_results(rtg_targets, dt_means, dt_stds,
                       dt_hp_means, dt_hp_stds,
                       ppo_mean, ppo_std,
                       ppo_hp_mean, ppo_hp_std,
                       save_path):
    """绘制 RTG Sweep 双子图：(a) Episode Return, (b) HP Violation"""
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    idt_color = '#1f77b4'
    ppo_color = '#555555'
    rtg_arr = np.array(rtg_targets)

    # ── 上子图：Episode Return ──────────────────────────────────────
    ax1.axhline(y=ppo_mean, color=ppo_color, linestyle='--', linewidth=2,
                label=f"PPO Baseline ({ppo_mean:.0f})")
    ax1.fill_between([rtg_arr.min(), rtg_arr.max()],
                     ppo_mean - ppo_std, ppo_mean + ppo_std,
                     color=ppo_color, alpha=0.1)
    ax1.plot(rtg_arr, dt_means, '-o', color=idt_color, linewidth=2, markersize=8,
             label="Proposed IDT")
    ax1.fill_between(rtg_arr,
                     np.array(dt_means) - np.array(dt_stds),
                     np.array(dt_means) + np.array(dt_stds),
                     color=idt_color, alpha=0.2)
    ax1.set_title("(a) RTG Controllability — Episode Return", fontweight='bold')
    ax1.set_xlabel("Target Return-to-Go (Prompt)")
    ax1.set_ylabel("Achieved Episode Return")
    ax1.legend(loc='lower right')

    # 添加区域标注
    n = len(rtg_arr)
    if n >= 6:
        ax1.text(rtg_arr[2], dt_means[2] + abs(dt_means[2]) * 0.05,
                 "Linear Region\n(Controllable)", color='green', fontsize=10)
        ax1.text(rtg_arr[-3], dt_means[-3] - abs(dt_means[-3]) * 0.05,
                 "Saturation Region\n(Physical Limit)", color='red', fontsize=10)

    # ── 下子图：HP Violation ────────────────────────────────────────
    ax2.axhline(y=ppo_hp_mean, color=ppo_color, linestyle='--', linewidth=2,
                label=f"PPO Baseline ({ppo_hp_mean:.1f})")
    ax2.fill_between([rtg_arr.min(), rtg_arr.max()],
                     ppo_hp_mean - ppo_hp_std, ppo_hp_mean + ppo_hp_std,
                     color=ppo_color, alpha=0.1)
    ax2.plot(rtg_arr, dt_hp_means, '-o', color=idt_color, linewidth=2, markersize=8,
             label="Proposed IDT")
    ax2.fill_between(rtg_arr,
                     np.array(dt_hp_means) - np.array(dt_hp_stds),
                     np.array(dt_hp_means) + np.array(dt_hp_stds),
                     color=idt_color, alpha=0.2)
    ax2.set_title("(b) RTG Controllability — HP Violation Count", fontweight='bold')
    ax2.set_xlabel("Target Return-to-Go (Prompt)")
    ax2.set_ylabel("HP Violation Count")
    ax2.legend(loc='upper right')

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    for ext in ('png', 'pdf'):
        out = os.path.join(save_path, f"rtg_sweep.{ext}")
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {out}")
    plt.close(fig)


def execute(cfg_xai, pm, env_cfg):
    ppo, dt = load_models(cfg_xai, env_cfg=env_cfg)
    asset_dir = resolve_project_path(pm, cfg_xai.asset_dir)
    save_dir = resolve_project_path(pm, cfg_xai.output_dir)
    os.makedirs(asset_dir, exist_ok=True)

    if ppo and dt:
        test_seeds = list(range(cfg_xai.seeds_per_point))
        ppo_mean, ppo_std, ppo_hp_mean, ppo_hp_std = evaluate_ppo(ppo, env_cfg, pm, test_seeds, cfg_xai)
        print(f"PPO Result: {ppo_mean:.2f} ± {ppo_std:.2f}, HP Viol: {ppo_hp_mean:.1f}")

        targets = np.linspace(cfg_xai.rtg_min, cfg_xai.rtg_max, cfg_xai.num_points)
        dt_means, dt_stds = [], []
        dt_hp_means, dt_hp_stds = [], []

        print(f"Starting RTG Sweep ({cfg_xai.num_points} points)...")
        for t in tqdm(targets):
            mean, std, hp_mean, hp_std = evaluate_dt_single_rtg(dt, env_cfg, pm, t, test_seeds, cfg_xai)
            dt_means.append(mean)
            dt_stds.append(std)
            dt_hp_means.append(hp_mean)
            dt_hp_stds.append(hp_std)

        np.savez(
            os.path.join(asset_dir, "rtg_sweep_metrics.npz"),
            rtg_targets=np.array(targets),
            dt_return_mean=np.array(dt_means),
            dt_return_std=np.array(dt_stds),
            dt_hp_mean=np.array(dt_hp_means),
            dt_hp_std=np.array(dt_hp_stds),
            ppo_return_mean=np.array([ppo_mean]),
            ppo_return_std=np.array([ppo_std]),
            ppo_hp_mean=np.array([ppo_hp_mean]),
            ppo_hp_std=np.array([ppo_hp_std]),
        )

        plot_sweep_results(targets, dt_means, dt_stds,
                           dt_hp_means, dt_hp_stds,
                           ppo_mean, ppo_std,
                           ppo_hp_mean, ppo_hp_std,
                           save_path=save_dir)


if __name__ == "__main__":
    pass
