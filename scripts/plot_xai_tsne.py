"""
X2: t-SNE 语义流形 — DT-E4 vs PPO-teacher 隐层对比

从 DT（E4）的 Transformer 上下文嵌入 与 PPO-teacher 的特征提取器输出
分别提取高维特征，用 t-SNE 降维到 2D 进行可视化。

颜色编码：
  - 每个场景 s0–s4 用不同颜色（训练场景）
  - 不同标记形状：DT vs PPO

用法：
    pixi run python scripts/plot_xai_tsne.py
"""
import os
import sys
import json
import pickle
import random
import warnings
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from omegaconf import OmegaConf
from stable_baselines3 import PPO

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

matplotlib.rcParams.update(
    {
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)

from src.basic_apis.dt_v2.model_v2 import build_dt_v2
from src.basic_apis.dt_v2.env_v2 import INTER_DIM, INTRA_DIM, HierarchicalSlicingEnvV2
from src.basic_apis.network_slicing_business.path_manager import PathManager

# ─────────────────────────── 配置 ──────────────────────────────
DT_MODEL_PATH  = "data/channel_generality/dt_v2_tiny/models/b2_e2/final_dt_v2.pth"
PPO_MODEL_PATH = "data/channel_generality/dt_v2_mean_rwd/ppo_joint_slice_attn/best_model/best_model.zip"
META_PATH      = "data/channel_generality/dt_v2_8exp/dataset_D-B/metadata.json"
DATASET_DIR    = "data/channel_generality/dt_v2_8exp/dataset_D-B/training"
OUTPUT_DIR     = "outputs/figures/channel_generality/xai"

NUM_SLICES  = 5
ACTION_DIMS = [11] + [5] * NUM_SLICES + [3] * NUM_SLICES
EMBED_DIM   = 512
N_PER_SCENARIO = 100     # 每场景采样时间步数（用于 t-SNE）
CONTEXT_LEN = 20
RANDOM_SEED = 42
TSNE_PERPLEXITY = 30
# ───────────────────────────────────────────────────────────────

SCENARIO_COLORS = {0: "#2196F3", 1: "#4CAF50", 2: "#FF9800", 3: "#9C27B0", 4: "#F44336"}
SCENARIO_NAMES  = {s: f"s{s}" for s in range(5)}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot t-SNE manifold for DT-v2 b2_e2 vs PPO teacher.")
    parser.add_argument("--dt_model_path", type=str, default=DT_MODEL_PATH)
    parser.add_argument("--ppo_model_path", type=str, default=PPO_MODEL_PATH)
    parser.add_argument("--meta_path", type=str, default=META_PATH)
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR)
    parser.add_argument("--obs_source", choices=["dataset", "env"], default="dataset")
    parser.add_argument("--scenarios", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--env_init_episode", type=int, default=0)
    parser.add_argument("--env_max_episode", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--n_per_scenario", type=int, default=N_PER_SCENARIO)
    parser.add_argument("--context_len", type=int, default=CONTEXT_LEN)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--perplexity", type=int, default=TSNE_PERPLEXITY)
    parser.add_argument("--dt_encoder_type", choices=["cross_attn", "mlp", "slice_attn"], default="slice_attn")
    parser.add_argument("--dt_n_layer", type=int, default=6)
    parser.add_argument("--dt_n_head", type=int, default=8)
    parser.add_argument("--dt_embed_dim", type=int, default=EMBED_DIM)
    parser.add_argument("--dt_encoder_hidden_dim", type=int, default=64)
    parser.add_argument("--dt_encoder_num_heads", type=int, default=4)
    parser.add_argument("--out_prefix", type=str, default="x2_tsne")
    return parser.parse_args()


def get_scenario_color(s):
    if s in SCENARIO_COLORS:
        return SCENARIO_COLORS[s]
    palette = ["#17becf", "#bcbd22", "#8c564b", "#e377c2", "#7f7f7f"]
    return palette[s % len(palette)]


def scenario_display_name(s: int) -> str:
    """Map scenario IDs to paper labels when possible (s5-s9 -> A-E)."""
    mapping_ood = {5: "A", 6: "B", 7: "C", 8: "D", 9: "E"}
    if s in mapping_ood:
        return mapping_ood[s]
    if s in SCENARIO_NAMES:
        return SCENARIO_NAMES[s]
    return f"s{s}"


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


def load_obs_by_scenario(dataset_dir, n_per_scenario=100, seed=42):
    """返回 {scenario: list of obs_dicts}"""
    rng = random.Random(seed)
    all_files = [f for f in os.listdir(dataset_dir) if f.endswith(".pkl")]

    scen_files = {}
    for fn in all_files:
        try:
            scen = int(fn.split("_single_")[0].lstrip("S"))
        except Exception:
            continue
        scen_files.setdefault(scen, []).append(fn)

    result = {}
    for scen, files in sorted(scen_files.items()):
        selected = rng.sample(files, min(len(files), max(1, n_per_scenario // 8 + 2)))
        obs_list = []
        for fn in selected:
            try:
                traj = pickle.load(open(os.path.join(dataset_dir, fn), "rb"))
                obs_seq = traj["observations"]
                n_take = max(1, n_per_scenario // len(selected))
                idxs = rng.sample(range(len(obs_seq)), min(n_take, len(obs_seq)))
                for t in sorted(idxs):
                    obs_list.append(obs_seq[t])
                    if len(obs_list) >= n_per_scenario:
                        break
            except Exception:
                continue
            if len(obs_list) >= n_per_scenario:
                break
        result[scen] = obs_list[:n_per_scenario]
    return result


def load_obs_from_env_scenarios(scenarios, n_per_scenario=100, seed=42, init_episode=0, max_episode=20):
    """从 EnvV2 采样指定场景观测（适用于 OOD s5-s9）。"""
    env_cfg = OmegaConf.load("conf/environment/env_ha.yaml")
    pm = PathManager(os.getcwd())
    result = {}
    for scen in scenarios:
        env_settings = env_cfg.env_settings.copy()
        env_settings.mode = "testing"
        env_settings.scenario_mode = "inside"
        env_settings.inside.testing.active_scenario_list = [int(scen)]
        env_settings.inside.testing.init_scenario_episode = int(init_episode)
        env_settings.inside.testing.max_scenario_episodes = int(max_episode)
        env = HierarchicalSlicingEnvV2(env_settings, np.random.default_rng(seed + int(scen)), pm)
        obs_list = []
        obs, _ = env.reset()
        while len(obs_list) < n_per_scenario:
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
        result[int(scen)] = obs_list
    return result


def obs_list_to_arrays(obs_list, obs_stats):
    """返回 (B,5,8), (B,25,7), (B,2)"""
    inter  = np.stack([normalize(o["inter_feat"],  "inter",  obs_stats) for o in obs_list])
    intra  = np.stack([normalize(o["intra_feat"],  "intra",  obs_stats) for o in obs_list])
    glob   = np.stack([normalize(o["global_feat"], "global", obs_stats) for o in obs_list])
    return inter, intra, glob


# ── DT 嵌入提取 ─────────────────────────────────────────────────
def extract_dt_embeddings(model, scen_obs, obs_stats, device, batch_size=64):
    """提取 DT 的 state encoder 输出（512维）。"""
    all_embs = []
    all_labels = []

    for scen, obs_list in sorted(scen_obs.items()):
        inter_arr, intra_arr, glob_arr = obs_list_to_arrays(obs_list, obs_stats)
        for i in range(0, len(obs_list), batch_size):
            b_inter = torch.from_numpy(inter_arr[i:i+batch_size]).float().unsqueeze(1).to(device)
            b_intra = torch.from_numpy(intra_arr[i:i+batch_size]).float().unsqueeze(1).to(device)
            b_glob  = torch.from_numpy(glob_arr[i:i+batch_size]).float().unsqueeze(1).to(device)
            with torch.no_grad():
                emb = model.state_encoder(b_inter, b_intra, b_glob)  # (B,1,512)
            all_embs.append(emb[:, 0, :].cpu().numpy())
            all_labels.extend([scen] * (i+batch_size - i if i+batch_size<=len(obs_list) else len(obs_list)-i))
        print(f"  DT: scenario {scen}, {len(obs_list)} embeddings extracted")

    return np.concatenate(all_embs, axis=0), np.array(all_labels)


# ── PPO 嵌入提取 ─────────────────────────────────────────────────
def extract_ppo_embeddings(ppo_model, scen_obs, device, batch_size=128):
    """提取 PPO feature extractor 输出（256维）。"""
    import torch

    feat_extractor = ppo_model.policy.features_extractor.to(device)
    feat_extractor.eval()

    all_embs   = []
    all_labels = []

    for scen, obs_list in sorted(scen_obs.items()):
        for i in range(0, len(obs_list), batch_size):
            batch = obs_list[i:i+batch_size]
            obs_dict = {
                "inter_feat":  torch.from_numpy(
                    np.stack([o["inter_feat"].astype(np.float32) for o in batch])
                ).to(device),
                "intra_feat":  torch.from_numpy(
                    np.stack([o["intra_feat"].astype(np.float32) for o in batch])
                ).to(device),
                "global_feat": torch.from_numpy(
                    np.stack([o["global_feat"].astype(np.float32) for o in batch])
                ).to(device),
            }
            with torch.no_grad():
                emb = feat_extractor(obs_dict)   # (B, 256)
            all_embs.append(emb.cpu().numpy())
            n_actual = len(batch)
            all_labels.extend([scen] * n_actual)
        print(f"  PPO: scenario {scen}, {len(obs_list)} embeddings extracted")

    return np.concatenate(all_embs, axis=0), np.array(all_labels)


# ── t-SNE 辅助 ──────────────────────────────────────────────────
def run_tsne(embeddings, seed=42, perplexity=30, n_iter=1000):
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity,
                learning_rate="auto", init="pca")
    return tsne.fit_transform(emb_scaled)


def scatter_tsne(ax, xy, labels, title, alpha=0.55, s=20):
    for scen in sorted(np.unique(labels)):
        mask = labels == scen
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   color=get_scenario_color(int(scen)), label=scenario_display_name(int(scen)),
                   alpha=alpha, s=s, linewidths=0)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("t-SNE dim 1", fontsize=13)
    ax.set_ylabel("t-SNE dim 2", fontsize=13)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    legend = ax.legend(title="Scenario", loc="upper right", fontsize=11,
                       markerscale=1.5, framealpha=0.8)
    legend.get_title().set_fontsize(12)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 1. 加载模型 ────────────────────────────────────────────────
    print("Loading DT model...")
    obs_stats = load_obs_stats(args.meta_path)
    act_dim = sum(ACTION_DIMS)
    model_cfg = OmegaConf.create({
        "context_len": int(args.context_len), "embed_dim": int(args.dt_embed_dim),
        "n_layer": int(args.dt_n_layer), "n_head": int(args.dt_n_head), "activation": "relu",
        "dropout": 0.1, "act_dim": act_dim,
    })
    dt_model = build_dt_v2(
        model_cfg=model_cfg, action_dims=ACTION_DIMS,
        inter_dim=INTER_DIM, intra_dim=INTRA_DIM,
        embed_dim=int(args.dt_embed_dim), encoder_type=str(args.dt_encoder_type),
        encoder_hidden_dim=int(args.dt_encoder_hidden_dim),
        encoder_num_heads=int(args.dt_encoder_num_heads),
    ).to(device)
    dt_model.load_state_dict(torch.load(args.dt_model_path, map_location=device))
    dt_model.eval()
    print(f"  DT model loaded: {args.dt_model_path}")

    print("Loading PPO teacher...")
    ppo_model = PPO.load(args.ppo_model_path, device=device)
    print(f"  PPO model loaded: {args.ppo_model_path}")

    # ── 2. 加载观测 ────────────────────────────────────────────────
    if args.obs_source == "dataset":
        print(f"Loading {args.n_per_scenario} obs/scenario from dataset...")
        scen_obs = load_obs_by_scenario(args.dataset_dir, n_per_scenario=int(args.n_per_scenario), seed=int(args.seed))
        if args.scenarios:
            keep = set(int(x) for x in args.scenarios)
            scen_obs = {k: v for k, v in scen_obs.items() if k in keep}
    else:
        print(f"Sampling {args.n_per_scenario} obs/scenario from EnvV2...")
        scen_obs = load_obs_from_env_scenarios(
            scenarios=[int(s) for s in args.scenarios],
            n_per_scenario=int(args.n_per_scenario),
            seed=int(args.seed),
            init_episode=int(args.env_init_episode),
            max_episode=int(args.env_max_episode),
        )

    # ── 3. 提取嵌入 ────────────────────────────────────────────────
    print("Extracting DT embeddings (state encoder output)...")
    dt_embs, dt_labels = extract_dt_embeddings(dt_model, scen_obs, obs_stats, device)
    print(f"  DT embeddings: {dt_embs.shape}")

    print("Extracting PPO embeddings (features extractor output)...")
    ppo_embs, ppo_labels = extract_ppo_embeddings(ppo_model, scen_obs, device)
    print(f"  PPO embeddings: {ppo_embs.shape}")

    # ── 4. 运行 t-SNE ──────────────────────────────────────────────
    print(f"Running t-SNE (perplexity={args.perplexity})...")
    print("  DT t-SNE...")
    dt_xy = run_tsne(dt_embs, seed=int(args.seed), perplexity=int(args.perplexity))
    print("  PPO t-SNE...")
    ppo_xy = run_tsne(ppo_embs, seed=int(args.seed), perplexity=int(args.perplexity))

    # ── 5. 联合 t-SNE（DT+PPO 同一嵌入空间，用不同标记）─────────────
    # 用 PCA 将两者统一到 min_dim 维，再联合做 t-SNE。
    from sklearn.decomposition import PCA
    min_dim = min(dt_embs.shape[1], ppo_embs.shape[1])
    dt_embs_scaled = StandardScaler().fit_transform(dt_embs)
    ppo_embs_scaled = StandardScaler().fit_transform(ppo_embs)
    if dt_embs_scaled.shape[1] != min_dim:
        dt_embs_proj = PCA(n_components=min_dim, random_state=int(args.seed)).fit_transform(dt_embs_scaled)
    else:
        dt_embs_proj = dt_embs_scaled
    if ppo_embs_scaled.shape[1] != min_dim:
        ppo_embs_proj = PCA(n_components=min_dim, random_state=int(args.seed)).fit_transform(ppo_embs_scaled)
    else:
        ppo_embs_proj = ppo_embs_scaled

    combined_embs   = np.concatenate([dt_embs_proj, ppo_embs_proj], axis=0)
    combined_labels = np.concatenate([dt_labels, ppo_labels], axis=0)
    combined_source = np.array(["IDT"] * len(dt_labels) + ["PPO-Discrete"] * len(ppo_labels))

    print("  Combined t-SNE...")
    combined_xy = run_tsne(combined_embs, seed=int(args.seed), perplexity=int(args.perplexity))
    dt_xy_combined  = combined_xy[:len(dt_labels)]
    ppo_xy_combined = combined_xy[len(dt_labels):]

    # ── 6. 绘图 ─────────────────────────────────────────────────────
    print("Plotting...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    plt.subplots_adjust(wspace=0.25, top=0.78)

    # 子图 A: DT state encoder
    scatter_tsne(axes[0], dt_xy, dt_labels,
                 "IDT State Encoder\n(512-dim → t-SNE)")

    # 子图 B: PPO feature extractor
    scatter_tsne(axes[1], ppo_xy, ppo_labels,
                 "PPO-Discrete Feature Extractor\n(256-dim → t-SNE)")

    # 子图 C: 联合（DT vs PPO，同一空间，两种形状）
    ax = axes[2]
    markers = {"IDT": "o", "PPO-Discrete": "^"}
    for src, mk in markers.items():
        mask_src = combined_source == src
        xy = combined_xy[mask_src]
        lbls = combined_labels[mask_src]
        for scen in sorted(np.unique(lbls)):
            mask_s = lbls == scen
            ax.scatter(xy[mask_s, 0], xy[mask_s, 1],
                       color=get_scenario_color(int(scen)), marker=mk,
                       alpha=0.45, s=18, linewidths=0,
                       label=f"s{int(scen)} ({src})" if scen == sorted(np.unique(lbls))[0] else None)

    # 图例：颜色=场景，形状=来源
    scenarios_present = sorted(np.unique(combined_labels).tolist())
    color_patches = [mpatches.Patch(color=get_scenario_color(int(s)), label=scenario_display_name(int(s)))
                     for s in scenarios_present]
    from matplotlib.lines import Line2D
    source_handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
              markersize=8, label="IDT"),
        Line2D([0], [0], marker="^", color="gray", linestyle="None",
              markersize=8, label="PPO-Discrete"),
    ]
    ax.legend(handles=color_patches + source_handles,
              loc="upper right", fontsize=11, framealpha=0.85,
              ncol=2, handletextpad=0.4, columnspacing=0.8)
    ax.set_title("Joint Embedding Space\n(IDT & PPO-Discrete, PCA-aligned)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("t-SNE dim 1", fontsize=13)
    ax.set_ylabel("t-SNE dim 2", fontsize=13)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    fig.suptitle(
        "t-SNE Semantic Manifold: IDT vs PPO-Discrete Hidden Representations\n"
        f"(Scenarios {','.join([scenario_display_name(int(s)) for s in scenarios_present])}, colored by scenario)",
        fontsize=15, fontweight="bold", y=0.98
    )

    out_png = os.path.join(args.output_dir, f"{args.out_prefix}_b2e2_vs_ppo.png")
    out_pdf = os.path.join(args.output_dir, f"{args.out_prefix}_b2e2_vs_ppo.pdf")
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")

    # ── 7. 打印簇间距离统计 ─────────────────────────────────────────
    print("\n=== Scenario Cluster Separation (DT, L2 between centroids) ===")
    dt_scenarios = sorted(np.unique(dt_labels).tolist())
    dt_centroids = {int(s): dt_xy[dt_labels == s].mean(axis=0) for s in dt_scenarios}
    for i, s1 in enumerate(dt_scenarios):
        for s2 in dt_scenarios[i + 1:]:
            d = np.linalg.norm(dt_centroids[s1] - dt_centroids[s2])
            print(f"  s{int(s1)} ↔ s{int(s2)}: {d:.2f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
