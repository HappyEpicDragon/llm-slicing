"""
X1: Attention Heatmap — SliceAttn 自注意力权重可视化

从 DT E4（SliceAttn）的状态编码器提取自注意力权重，
展示 5 个切片 Token + 1 个全局 Token 之间的注意力模式。

用法：
    pixi run python scripts/plot_xai_attention.py
"""
import os
import sys
import json
import pickle
import random
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.basic_apis.dt_v2.model_v2 import build_dt_v2, SliceAttnStateEncoderV2
from src.basic_apis.dt_v2.env_v2 import INTER_DIM, INTRA_DIM, HierarchicalSlicingEnvV2
from src.basic_apis.ppo.ppo_ha_weighted.agent_hierarchical import (
    HierarchicalSmartPolicy,
    HierarchicalMLPPolicy,
    HierarchicalPooledAttentionPolicy,
    HierarchicalSliceAttnPolicy,
)
from src.basic_apis.network_slicing_business.path_manager import PathManager
from stable_baselines3 import PPO

# ─────────────────────────── 配置 ──────────────────────────────
MODEL_PATH = "data/channel_generality/dt_v2_tiny/models/b2_e2/final_dt_v2.pth"
PPO_MODEL_PATH = "data/channel_generality/dt_v2_mean_rwd/ppo_joint_slice_attn/best_model/best_model.zip"
META_PATH  = "data/channel_generality/dt_v2_8exp/dataset_D-B/metadata.json"
DATASET_DIR = "data/channel_generality/dt_v2_8exp/dataset_D-B/training"
OUTPUT_DIR = "outputs/figures/channel_generality/xai"

NUM_SLICES = 5
ACTION_DIMS = [11] + [5] * NUM_SLICES + [3] * NUM_SLICES
EMBED_DIM = 512
N_SAMPLES_PER_SCENARIO = 60     # 每场景采样的时间步数
RANDOM_SEED = 42

TOKEN_LABELS = ["Slice 0", "Slice 1", "Slice 2", "Slice 3", "Slice 4", "Global"]
SCENARIO_LABELS = {0: "s0", 1: "s1", 2: "s2", 3: "s3", 4: "s4"}
# ───────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DT and PPO attention heatmaps on sampled dataset states.")
    parser.add_argument("--dt_model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--ppo_model_path", type=str, default=PPO_MODEL_PATH)
    parser.add_argument("--meta_path", type=str, default=META_PATH)
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR)
    parser.add_argument("--obs_source", choices=["dataset", "env"], default="dataset")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--scenario", type=int, default=7)
    parser.add_argument("--samples_per_scenario", type=int, default=N_SAMPLES_PER_SCENARIO)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--dt_encoder_type", choices=["cross_attn", "mlp", "slice_attn"], default="slice_attn")
    parser.add_argument("--dt_n_layer", type=int, default=6)
    parser.add_argument("--dt_n_head", type=int, default=8)
    parser.add_argument("--dt_embed_dim", type=int, default=EMBED_DIM)
    parser.add_argument("--dt_encoder_hidden_dim", type=int, default=64)
    parser.add_argument("--dt_encoder_num_heads", type=int, default=4)
    parser.add_argument("--out_prefix", type=str, default="x1_attention")
    parser.add_argument("--env_init_episode", type=int, default=0)
    parser.add_argument("--env_max_episode", type=int, default=20)
    return parser.parse_args()


def load_obs_stats(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    stats = {}
    for k in ["inter", "intra", "global"]:
        if k in meta["obs_stats"]:
            stats[k] = {
                "mean": np.array(meta["obs_stats"][k]["mean"], dtype=np.float32),
                "std":  np.array(meta["obs_stats"][k]["std"],  dtype=np.float32) + 1e-6,
            }
    return stats


def normalize(arr, key, obs_stats):
    if key not in obs_stats:
        return arr.astype(np.float32)
    mean = obs_stats[key]["mean"]
    std  = obs_stats[key]["std"].copy()
    std[std < 1e-2] = 1.0
    return np.clip((arr - mean) / std, -5.0, 5.0).astype(np.float32)


def load_trajectories_by_scenario(dataset_dir, n_per_scenario=60, seed=42):
    """从训练集按场景分组加载观测。返回 dict: scenario_idx -> list of obs dicts."""
    rng = random.Random(seed)
    all_files = [f for f in os.listdir(dataset_dir) if f.endswith(".pkl")]

    # 解析文件名中的场景编号（文件名形如 S0_single_...）
    scen_files = {}
    for fn in all_files:
        try:
            scen = int(fn.split("_single_")[0].lstrip("S"))
        except Exception:
            continue
        scen_files.setdefault(scen, []).append(fn)

    result = {}
    for scen, files in sorted(scen_files.items()):
        selected = rng.sample(files, min(len(files), max(1, n_per_scenario // 10 + 1)))
        obs_list = []
        for fn in selected:
            try:
                traj = pickle.load(open(os.path.join(dataset_dir, fn), "rb"))
                obs_seq = traj["observations"]          # list of T dicts
                steps_to_take = n_per_scenario // len(selected) + 1
                step_ids = rng.sample(range(len(obs_seq)), min(steps_to_take, len(obs_seq)))
                for t in step_ids:
                    obs_list.append(obs_seq[t])
                    if len(obs_list) >= n_per_scenario:
                        break
            except Exception as e:
                print(f"  Warning: failed to load {fn}: {e}")
                continue
            if len(obs_list) >= n_per_scenario:
                break
        result[scen] = obs_list[:n_per_scenario]
        print(f"  Loaded {len(result[scen])} obs for scenario {scen}")
    return result


def load_obs_from_env_scenario(scenario, n_samples=60, seed=42, init_episode=0, max_episode=20):
    """在 EnvV2 测试模式下采样指定场景观测（用于 OOD 场景，如 s7）。"""
    env_cfg = OmegaConf.load("conf/environment/env_ha.yaml")
    env_settings = env_cfg.env_settings.copy()
    env_settings.mode = "testing"
    env_settings.scenario_mode = "inside"
    env_settings.inside.testing.active_scenario_list = [int(scenario)]
    env_settings.inside.testing.init_scenario_episode = int(init_episode)
    env_settings.inside.testing.max_scenario_episodes = int(max_episode)

    pm = PathManager(os.getcwd())
    env = HierarchicalSlicingEnvV2(env_settings, np.random.default_rng(seed), pm)

    obs_list = []
    obs, _ = env.reset()
    while len(obs_list) < n_samples:
        obs_list.append({
            "inter_feat": obs["inter_feat"].copy(),
            "intra_feat": obs["intra_feat"].copy(),
            "global_feat": obs["global_feat"].copy(),
        })
        action = env.action_space.sample().astype(np.int32)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return obs_list


def obs_list_to_tensors(obs_list, obs_stats, device):
    """将观测列表转换为 (B,1,5,8), (B,1,25,7), (B,1,2) 张量。"""
    inter_arr  = np.stack([normalize(o["inter_feat"],  "inter",  obs_stats) for o in obs_list])  # (B,5,8)
    intra_arr  = np.stack([normalize(o["intra_feat"],  "intra",  obs_stats) for o in obs_list])  # (B,25,7)
    global_arr = np.stack([normalize(o["global_feat"], "global", obs_stats) for o in obs_list])  # (B,2)

    inter  = torch.from_numpy(inter_arr).float().unsqueeze(1).to(device)   # (B,1,5,8)
    intra  = torch.from_numpy(intra_arr).float().unsqueeze(1).to(device)   # (B,1,25,7)
    glob   = torch.from_numpy(global_arr).float().unsqueeze(1).to(device)  # (B,1,2)
    return inter, intra, glob


def extract_attn_weights(encoder, obs_list, obs_stats, device, batch_size=64):
    """分批提取注意力权重，返回 (N, 6, 6) numpy 数组。"""
    all_weights = []
    for i in range(0, len(obs_list), batch_size):
        batch = obs_list[i:i + batch_size]
        inter, intra, glob = obs_list_to_tensors(batch, obs_stats, device)
        with torch.no_grad():
            _, attn = encoder(inter, intra, glob, return_attn=True)
        # attn: (B, K=1, 6, 6)
        all_weights.append(attn[:, 0, :, :].cpu().numpy())
    return np.concatenate(all_weights, axis=0)   # (N, 6, 6)


def extract_ppo_attn_weights(ppo_model, obs_list, device, batch_size=64):
    """提取 PPO feature extractor 的 self-attention 权重，期望形状 (N,6,6)。"""
    feat_extractor = ppo_model.policy.features_extractor.to(device)
    feat_extractor.eval()
    all_weights = []
    for i in range(0, len(obs_list), batch_size):
        batch = obs_list[i:i + batch_size]
        obs_dict = {
            "inter_feat": torch.from_numpy(np.stack([o["inter_feat"].astype(np.float32) for o in batch])).to(device),
            "intra_feat": torch.from_numpy(np.stack([o["intra_feat"].astype(np.float32) for o in batch])).to(device),
            "global_feat": torch.from_numpy(np.stack([o["global_feat"].astype(np.float32) for o in batch])).to(device),
        }
        with torch.no_grad():
            _, attn = feat_extractor(obs_dict, return_attn=True)
        all_weights.append(attn.cpu().numpy())
    return np.concatenate(all_weights, axis=0)


def plot_heatmap(ax, attn_mean, title, vmin=None, vmax=None, show_cbar=False, fig=None):
    """在指定 ax 上绘制 6×6 注意力热图。"""
    if vmin is None:
        vmin = attn_mean.min()
    if vmax is None:
        vmax = attn_mean.max()

    im = ax.imshow(attn_mean, aspect="equal", cmap="YlOrRd", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(TOKEN_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(TOKEN_LABELS, fontsize=8)
    ax.set_xlabel("Key Token", fontsize=9)
    ax.set_ylabel("Query Token", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")

    # 在格子内标注数值
    for r in range(6):
        for c in range(6):
            val = attn_mean[r, c]
            color = "white" if val > (vmin + (vmax - vmin) * 0.6) else "black"
            ax.text(c, r, f"{val:.3f}", ha="center", va="center", fontsize=6.5, color=color)

    if show_cbar and fig is not None:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 1. 加载模型（仅用 state_encoder） ──────────────────────────
    print("Loading model...")
    obs_stats = load_obs_stats(args.meta_path)
    act_dim = sum(ACTION_DIMS)
    model_cfg = OmegaConf.create({
        "context_len": 20, "embed_dim": int(args.dt_embed_dim),
        "n_layer": int(args.dt_n_layer), "n_head": int(args.dt_n_head), "activation": "relu",
        "dropout": 0.1, "act_dim": act_dim,
    })
    model = build_dt_v2(
        model_cfg=model_cfg, action_dims=ACTION_DIMS,
        inter_dim=INTER_DIM, intra_dim=INTRA_DIM,
        embed_dim=int(args.dt_embed_dim), encoder_type=str(args.dt_encoder_type),
        encoder_hidden_dim=int(args.dt_encoder_hidden_dim),
        encoder_num_heads=int(args.dt_encoder_num_heads),
    ).to(device)
    state_dict = torch.load(args.dt_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    encoder: SliceAttnStateEncoderV2 = model.state_encoder
    print(f"  DT model loaded: {args.dt_model_path}")

    print("Loading PPO model...")
    policy_map = {
        "attention": HierarchicalSmartPolicy,
        "mlp": HierarchicalMLPPolicy,
        "pooled_attention": HierarchicalPooledAttentionPolicy,
        "slice_attn": HierarchicalSliceAttnPolicy,
    }
    custom_objects = {"policy_class": policy_map["slice_attn"]}
    ppo_model = PPO.load(args.ppo_model_path, device=device, custom_objects=custom_objects)
    print(f"  PPO model loaded: {args.ppo_model_path}")

    # ── 2. 加载训练轨迹 ────────────────────────────────────────────
    if args.obs_source == "dataset":
        print("Loading training trajectories...")
        scen_obs = load_trajectories_by_scenario(
            args.dataset_dir, n_per_scenario=int(args.samples_per_scenario), seed=int(args.seed)
        )
    else:
        print(f"Sampling observations from EnvV2 scenario s{args.scenario}...")
        scen_obs = {
            int(args.scenario): load_obs_from_env_scenario(
                scenario=int(args.scenario),
                n_samples=int(args.samples_per_scenario),
                seed=int(args.seed),
                init_episode=int(args.env_init_episode),
                max_episode=int(args.env_max_episode),
            )
        }
        print(f"  Sampled {len(scen_obs[int(args.scenario)])} obs from env scenario s{args.scenario}")

    # ── 3. 提取各场景注意力权重 ─────────────────────────────────────
    print("Extracting attention weights...")
    scen_attn = {}
    for scen, obs_list in sorted(scen_obs.items()):
        if not obs_list:
            continue
        weights = extract_attn_weights(encoder, obs_list, obs_stats, device)
        scen_attn[scen] = weights
        print(f"  Scenario {scen}: mean attn shape {weights.shape}")

    # 汇总均值（所有场景平均）
    all_weights = np.concatenate(list(scen_attn.values()), axis=0)
    overall_mean = all_weights.mean(axis=0)    # (6, 6)
    overall_std  = all_weights.std(axis=0)

    global_vmin = 0.0
    global_vmax = float(np.percentile(all_weights, 95))

    # ── 4. 绘图 ─────────────────────────────────────────────────────
    print("Plotting...")
    n_scen = len(scen_attn)
    fig = plt.figure(figsize=(5 * (n_scen + 1), 5.5))
    gs  = gridspec.GridSpec(1, n_scen + 1, wspace=0.35)

    axes = [fig.add_subplot(gs[0, i]) for i in range(n_scen + 1)]

    # 各场景子图
    for idx, (scen, weights) in enumerate(sorted(scen_attn.items())):
        mean_w = weights.mean(axis=0)
        plot_heatmap(axes[idx], mean_w, f"Scenario s{scen} (n={len(weights)})",
                     vmin=global_vmin, vmax=global_vmax)

    # 最后一列：汇总均值
    im = plot_heatmap(axes[-1], overall_mean, "Overall Mean (s0–s4)",
                      vmin=global_vmin, vmax=global_vmax, show_cbar=True, fig=fig)

    fig.suptitle(
        "IDT-v2 (E4) SliceAttn Self-Attention Weights\n(Query→Key attention, 5 Slice Tokens + 1 Global Token)",
        fontsize=13, fontweight="bold", y=1.02
    )

    scen_keys = sorted(scen_attn.keys())
    if len(scen_keys) == 1:
        scen_tag = f"s{scen_keys[0]}"
    else:
        scen_tag = f"s{scen_keys[0]}_s{scen_keys[-1]}"
    out_png = os.path.join(args.output_dir, f"{args.out_prefix}_b2e2_{scen_tag}.png")
    out_pdf = os.path.join(args.output_dir, f"{args.out_prefix}_b2e2_{scen_tag}.pdf")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")

    # ── 5. 附加：整体均值 + 标准差 热图（2列并排）──────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    plot_heatmap(axes2[0], overall_mean, "Mean Attention (all scenarios)",
                 vmin=global_vmin, vmax=global_vmax, show_cbar=True, fig=fig2)
    plot_heatmap(axes2[1], overall_std,  "Std Dev of Attention",
                 show_cbar=True, fig=fig2)
    fig2.suptitle("IDT-v2 (E4) SliceAttn — Mean & Std Attention", fontsize=12, fontweight="bold")

    out2_png = os.path.join(args.output_dir, f"{args.out_prefix}_b2e2_mean_std.png")
    out2_pdf = os.path.join(args.output_dir, f"{args.out_prefix}_b2e2_mean_std.pdf")
    fig2.savefig(out2_png, dpi=180, bbox_inches="tight")
    fig2.savefig(out2_pdf, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out2_png}")

    # ── 6. 打印汇总统计 ──────────────────────────────────────────────
    print("\n=== Overall Attention Statistics ===")
    print(f"Mean attention matrix:\n{np.round(overall_mean, 4)}")
    print(f"\nDominant attention (row=query, col=key):")
    for r, q_label in enumerate(TOKEN_LABELS):
        top_k = np.argsort(overall_mean[r])[::-1][:3]
        top_str = ", ".join(f"{TOKEN_LABELS[c]}={overall_mean[r, c]:.3f}" for c in top_k)
        print(f"  {q_label:8s} attends to: {top_str}")

    # ── 7. X1 主图：DT vs PPO（固定场景）────────────────────────────
    target_scen = int(args.scenario)
    if target_scen not in scen_obs or not scen_obs[target_scen]:
        fallback_scen = sorted(scen_obs.keys())[0]
        print(f"Warning: scenario s{target_scen} not in sampled dataset; fallback to s{fallback_scen}.")
        target_scen = fallback_scen
    obs_list = scen_obs[target_scen]
    dt_weights = extract_attn_weights(encoder, obs_list, obs_stats, device)
    dt_mean = dt_weights.mean(axis=0)  # (6,6)

    ppo_weights = extract_ppo_attn_weights(ppo_model, obs_list, device)
    ppo_mean = ppo_weights.mean(axis=0)
    if ppo_mean.shape != (6, 6):
        raise ValueError(
            f"PPO attention shape is {ppo_mean.shape}, expected (6, 6) for one-to-one fair comparison."
        )

    fig3 = plt.figure(figsize=(15, 6))
    gs3 = gridspec.GridSpec(1, 2, wspace=0.35)
    ax1 = fig3.add_subplot(gs3[0, 0])
    ax2 = fig3.add_subplot(gs3[0, 1])

    plot_heatmap(
        ax1, dt_mean,
        f"IDT-v2 b2_e2 SliceAttn (s{target_scen}, n={len(obs_list)})",
        vmin=0.0, vmax=float(max(np.percentile(dt_weights, 95), np.percentile(ppo_weights, 95))),
    )

    plot_heatmap(
        ax2, ppo_mean,
        f"PPO-teacher SliceAttn (s{target_scen}, n={len(obs_list)})",
        vmin=0.0, vmax=float(max(np.percentile(dt_weights, 95), np.percentile(ppo_weights, 95))),
        show_cbar=True, fig=fig3,
    )

    fig3.suptitle(
        "X1 Fair Attention Comparison (same 6x6 token self-attention): IDT-v2 b2_e2 vs PPO-teacher",
        fontsize=12, fontweight="bold"
    )
    out3_png = os.path.join(args.output_dir, f"{args.out_prefix}_b2e2_vs_ppo_s{target_scen}.png")
    out3_pdf = os.path.join(args.output_dir, f"{args.out_prefix}_b2e2_vs_ppo_s{target_scen}.pdf")
    fig3.savefig(out3_png, dpi=180, bbox_inches="tight")
    fig3.savefig(out3_pdf, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {out3_png}")
    print(f"  Saved: {out3_pdf}")

    print("\nDone!")


if __name__ == "__main__":
    main()
