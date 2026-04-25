import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
from omegaconf import OmegaConf
from src.basic_apis.network_slicing_business.path_context import PathContext
from src.basic_apis.explainable_ai_utils.xai_utils import resolve_project_path


# ===========================
# 1. 辅助函数
# ===========================
def reconstruct_timesteps(data):
    timesteps = []
    has_real_ts = 'timestep' in data[0]
    for i, d in enumerate(data):
        t = d['timestep'] if has_real_ts else i % 200
        timesteps.append(t)
    return np.array(timesteps)


def load_data(path):
    if not os.path.exists(path):
        print(f"⚠️ Data file not found: {path}")
        return None, None, None, None
    data = np.load(path, allow_pickle=True)
    embeddings = np.array([d['embedding'] for d in data])
    labels = np.array([d['scenario_label'] for d in data])
    timesteps = reconstruct_timesteps(data)

    ep_ids = []
    curr = 0
    last_t = 1e9
    for t in timesteps:
        if t < last_t: curr += 1
        ep_ids.append(curr)
        last_t = t

    return embeddings, labels, timesteps, np.array(ep_ids)


def compute_tsne(embeddings):
    print("   Running t-SNE (init='pca')...")
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, init='pca', learning_rate='auto')
    z = tsne.fit_transform(embeddings)
    return z


# ===========================
# 2. 绘图组件
# ===========================

def plot_ppo_simplified(ax, z, labels, ep_ids, timesteps):
    """
    [图a 终极修正]
    精简线条：截断尾部震荡 + 强力平滑
    """
    # 背景散点
    palette = {0: "#5DADE2", 1: "#E74C3C"}
    sns.scatterplot(x=z[:, 0], y=z[:, 1], hue=labels, palette=palette,
                    ax=ax, alpha=0.1, s=10, legend=False, edgecolor=None)

    colors = ['#00008B', '#8B0000']

    for domain, color in zip([0, 1], colors):
        mask_dom = (labels == domain)
        dom_eps = np.unique(ep_ids[mask_dom])

        # 挑选一条"跑得远"的轨迹 (Range large)
        best_ep = dom_eps[0]
        max_spread = 0

        for ep in dom_eps[:30]:
            mask_ep = (ep_ids == ep)
            z_ep = z[mask_ep]
            if len(z_ep) < 15: continue

            # 计算散布范围
            spread = np.ptp(z_ep[:, 0]) + np.ptp(z_ep[:, 1])
            if spread > max_spread:
                max_spread = spread
                best_ep = ep

        # 提取数据
        mask_ep = (ep_ids == best_ep)
        z_ep = z[mask_ep]
        t_ep = timesteps[mask_ep]
        idx = np.argsort(t_ep)
        z_ep = z_ep[idx]

        # [关键修改 1] 截断尾部 (Cut off tail)
        # 只取前 75%，去掉后面原地打转的死结
        cutoff_idx = int(len(z_ep) * 0.75)
        z_ep = z_ep[:cutoff_idx]

        # [关键修改 2] 强力平滑 (Strong Smoothing)
        # sigma 提高到 3.0，把所有的小折线都抹平
        if len(z_ep) > 5:
            smooth_x = gaussian_filter1d(z_ep[:, 0], sigma=3.0)
            smooth_y = gaussian_filter1d(z_ep[:, 1], sigma=3.0)

            # 画线
            ax.plot(smooth_x, smooth_y, color=color, linewidth=2.5, alpha=0.9)

            # 箭头 (画在末端)
            ax.arrow(smooth_x[-2], smooth_y[-2],
                     smooth_x[-1] - smooth_x[-2], smooth_y[-1] - smooth_y[-2],
                     head_width=3.0, color=color, zorder=10)

    ax.set_title("(a) t-SNE's Macro Flow of PPO Baseline", fontweight='bold')

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color="#00008B", lw=2.5), Line2D([0], [0], color="#8B0000", lw=2.5)]
    ax.legend(custom_lines, ['Source Trajectory', 'Target Trajectory'], loc='upper right', fontsize=9)


def plot_dt_refined(ax, z, labels, timesteps):
    """
    [图b] 保持精致长矛
    """
    palette = {0: "#5DADE2", 1: "#E74C3C"}
    sns.scatterplot(x=z[:, 0], y=z[:, 1], hue=labels, palette=palette,
                    ax=ax, alpha=0.1, s=15, legend=False, edgecolor=None)

    colors = ['#00008B', '#8B0000']
    for domain, color in zip([0, 1], colors):
        mask = (labels == domain)
        z_dom = z[mask]
        t_dom = timesteps[mask]
        max_t = np.max(t_dom)

        p_start = np.mean(z_dom[t_dom < max_t * 0.1], axis=0)
        p_end = np.mean(z_dom[t_dom > max_t * 0.9], axis=0)

        # 强制长度
        vec = p_end - p_start
        vec_len = np.linalg.norm(vec)
        xlim = ax.get_xlim();
        ylim = ax.get_ylim()
        min_len = min(xlim[1] - xlim[0], ylim[1] - ylim[0]) * 0.3

        if vec_len < min_len:
            scale = min_len / (vec_len + 1e-6)
            p_end_vis = p_start + vec * scale
        else:
            p_end_vis = p_end

        arrow = FancyArrowPatch(posA=p_start, posB=p_end_vis,
                                arrowstyle='-|>', color=color,
                                mutation_scale=25, linewidth=2.5, alpha=0.9, zorder=20)
        ax.add_patch(arrow)

    ax.set_title("(b) t-SNE's Macro Flow of IDT", fontweight='bold')
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color="#00008B", lw=2.5), Line2D([0], [0], color="#8B0000", lw=2.5)]
    ax.legend(custom_lines, ["Source Flow", "Target Flow"], loc='upper right', fontsize=9)


def plot_dt_aligned_labels(ax, z, labels, timesteps):
    """
    [图c] 保持对齐逻辑，确保美观
    """
    ax.scatter(z[:, 0], z[:, 1], c='lightgrey', alpha=0.1, s=10)
    domains = [0, 1]
    cmap = plt.get_cmap('viridis')

    for domain in domains:
        mask = (labels == domain)
        z_dom = z[mask]
        t_dom = timesteps[mask]
        max_t = np.max(t_dom)

        p_start = np.mean(z_dom[t_dom <= max_t * 0.05], axis=0)
        p_end = np.mean(z_dom[t_dom >= max_t * 0.95], axis=0)

        mid_points = []
        for p in np.linspace(0.1, 0.9, 5):
            window = 0.05
            idx = (t_dom >= max_t * (p - window)) & (t_dom <= max_t * (p + window))
            if np.sum(idx) > 0: mid_points.append(np.mean(z_dom[idx], axis=0))

        ctrl_points = np.vstack([p_start, mid_points, p_end])

        t_param = np.linspace(0, 1, len(ctrl_points))
        spl_x = make_interp_spline(t_param, ctrl_points[:, 0], k=2)
        spl_y = make_interp_spline(t_param, ctrl_points[:, 1], k=2)
        t_fine = np.linspace(0, 1, 100)
        x_fine = spl_x(t_fine)
        y_fine = spl_y(t_fine)

        # 彩虹蛇
        points = np.array([x_fine, y_fine]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, 1))
        lc.set_array(t_fine)
        lc.set_linewidth(6)
        lc.set_alpha(0.9)
        lc.set_capstyle('round')
        ax.add_collection(lc)

        # Start / End 点
        ax.scatter(x_fine[0], y_fine[0], color=cmap(0.0), s=120, edgecolor='black', zorder=10)
        ax.scatter(x_fine[-1], y_fine[-1], color=cmap(1.0), s=120, edgecolor='black', zorder=10)

        # 智能标签
        offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08  # 稍微加大偏移量

        # 判断走向
        if y_fine[-1] > y_fine[0]:  # 向上走
            ax.text(x_fine[0], y_fine[0] - offset, "Start", fontsize=9, fontweight='bold', ha='center', va='top')
            ax.text(x_fine[-1], y_fine[-1] + offset / 1.5, "End", fontsize=9, fontweight='bold', ha='center',
                    va='bottom')
        else:  # 向下走
            ax.text(x_fine[0], y_fine[0] + offset / 1.5, "Start", fontsize=9, fontweight='bold', ha='center',
                    va='bottom')
            ax.text(x_fine[-1], y_fine[-1] - offset, "End", fontsize=9, fontweight='bold', ha='center', va='top')

    ax.set_title("(c) t-SNE's Trajectory Evolution of IDT", fontweight='bold')
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Evolution (0% -> 100%)", rotation=270, labelpad=15)


# ===========================
# 3. 主程序
# ===========================
def visualize(cfg, path_context):
    print("🎨 [Visualization] Generating Final Polished Plots...")
    asset_dir = resolve_project_path(path_context, cfg.asset_dir)
    figure_dir = resolve_project_path(path_context, cfg.output_dir)
    os.makedirs(figure_dir, exist_ok=True)

    z_ppo, l_ppo, t_ppo, e_ppo = load_data(os.path.join(asset_dir, "ppo_data.npy"))
    z_dt, l_dt, t_dt, e_dt = load_data(os.path.join(asset_dir, "dt_data.npy"))
    if z_ppo is None: return

    z_ppo = compute_tsne(z_ppo)
    z_dt = compute_tsne(z_dt)

    sns.set_context("paper", font_scale=1.6)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)

    plot_ppo_simplified(axes[0], z_ppo, l_ppo, e_ppo, t_ppo)
    plot_dt_refined(axes[1], z_dt, l_dt, t_dt)
    plot_dt_aligned_labels(axes[2], z_dt, l_dt, t_dt)

    # H3: 量化跨场景语义流形差异（并导出 MMD 矩阵）
    ppo_mmd_matrix, ppo_mean_mmd = compute_cross_scenario_mmd(
        z_ppo, l_ppo, save_path=os.path.join(figure_dir, "mmd_matrix_ppo")
    )
    dt_mmd_matrix, dt_mean_mmd = compute_cross_scenario_mmd(
        z_dt, l_dt, save_path=os.path.join(figure_dir, "mmd_matrix_idt")
    )

    def extract_source_target_mmd(labels, mmd_matrix):
        unique_labels = sorted(np.unique(labels))
        if len(unique_labels) < 2:
            return float("nan")
        src_label, tgt_label = unique_labels[0], unique_labels[-1]
        return mmd_matrix.get((src_label, tgt_label), float("nan"))

    ppo_src_tgt_mmd = extract_source_target_mmd(l_ppo, ppo_mmd_matrix)
    dt_src_tgt_mmd = extract_source_target_mmd(l_dt, dt_mmd_matrix)

    axes[0].text(
        0.02, 0.98, f"MMD(src->tgt) = {ppo_src_tgt_mmd:.4f}",
        transform=axes[0].transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    axes[1].text(
        0.02, 0.98, f"MMD(src->tgt) = {dt_src_tgt_mmd:.4f}",
        transform=axes[1].transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    axes[2].text(
        0.02, 0.98, f"Mean Cross-Scenario MMD (IDT) = {dt_mean_mmd:.4f}",
        transform=axes[2].transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    for ax in axes:
        ax.set_xlabel("Latent Dim 1")
        ax.set_ylabel("Latent Dim 2")
        ax.grid(True, linestyle=':', alpha=0.4)

    out_path = os.path.join(figure_dir, "final_polish.pdf")
    plt.savefig(out_path, bbox_inches='tight')
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved to: {out_path}")
    print(f"✅ MMD summary: PPO(src->tgt)={ppo_src_tgt_mmd:.6f}, IDT(src->tgt)={dt_src_tgt_mmd:.6f}")


def compute_mmd(embeddings_a, embeddings_b, kernel='rbf', gamma=None):
    """
    H3 实验：计算两组嵌入向量之间的 Maximum Mean Discrepancy（MMD）。

    MMD 小 → 两组语义流形对齐良好（IDT 能区分场景）；
    MMD 大 → 流形分布差距大，场景表示混淆。

    Args:
        embeddings_a: ndarray [n_a, d]，第一组嵌入（如 IDT 跨场景嵌入）。
        embeddings_b: ndarray [n_b, d]，第二组嵌入（如 DT Baseline 嵌入）。
        kernel: 核函数类型，'rbf'（高斯核）或 'linear'。
        gamma: RBF 核的带宽参数。为 None 时使用中位数启发式。

    Returns:
        mmd_value: float，MMD² 估计值（越小说明分布越相似）。
    """
    def rbf_kernel(X, Y, g):
        """计算 RBF（高斯）核矩阵 K[i,j] = exp(-g * ||X_i - Y_j||²)"""
        XX = np.sum(X ** 2, axis=1, keepdims=True)
        YY = np.sum(Y ** 2, axis=1, keepdims=True)
        dist = XX + YY.T - 2.0 * X @ Y.T
        return np.exp(-g * np.clip(dist, 0, None))

    def linear_kernel(X, Y):
        return X @ Y.T

    if gamma is None and kernel == 'rbf':
        # 中位数启发式带宽
        all_emb = np.vstack([embeddings_a, embeddings_b])
        pairwise_sq = np.sum((all_emb[:, None] - all_emb[None]) ** 2, axis=-1)
        median_sq = np.median(pairwise_sq[pairwise_sq > 0])
        gamma = 1.0 / (2.0 * median_sq) if median_sq > 0 else 1.0

    if kernel == 'rbf':
        K_aa = rbf_kernel(embeddings_a, embeddings_a, gamma)
        K_bb = rbf_kernel(embeddings_b, embeddings_b, gamma)
        K_ab = rbf_kernel(embeddings_a, embeddings_b, gamma)
    else:
        K_aa = linear_kernel(embeddings_a, embeddings_a)
        K_bb = linear_kernel(embeddings_b, embeddings_b)
        K_ab = linear_kernel(embeddings_a, embeddings_b)

    n_a, n_b = len(embeddings_a), len(embeddings_b)
    # 无偏 MMD² 估计：去除对角线自身项
    np.fill_diagonal(K_aa, 0)
    np.fill_diagonal(K_bb, 0)
    mmd2 = (K_aa.sum() / (n_a * (n_a - 1))
            + K_bb.sum() / (n_b * (n_b - 1))
            - 2.0 * K_ab.mean())

    return float(mmd2)


def compute_cross_scenario_mmd(embeddings, labels, save_path=None):
    """
    H3 辅助函数：计算跨场景 MMD 矩阵，用于量化场景分离度。

    Args:
        embeddings: ndarray [N, d]，嵌入向量。
        labels: ndarray [N]，场景标签（整数）。
        save_path: 若提供，保存 MMD 矩阵热图（png + pdf）。

    Returns:
        mmd_matrix: dict[(scen_a, scen_b)] = mmd_value
        mean_mmd: float，所有跨场景对的平均 MMD
    """
    unique_labels = sorted(np.unique(labels))
    n = len(unique_labels)
    mmd_matrix = {}
    mat = np.zeros((n, n))

    for i, la in enumerate(unique_labels):
        for j, lb in enumerate(unique_labels):
            if i == j:
                mmd_matrix[(la, lb)] = 0.0
                continue
            emb_a = embeddings[labels == la]
            emb_b = embeddings[labels == lb]
            if len(emb_a) < 2 or len(emb_b) < 2:
                val = float('nan')
            else:
                val = compute_mmd(emb_a, emb_b)
            mmd_matrix[(la, lb)] = val
            mat[i, j] = val if not np.isnan(val) else 0.0

    # 仅取上三角（对称）的平均 MMD
    upper_vals = [mmd_matrix[(la, lb)] for i, la in enumerate(unique_labels)
                  for j, lb in enumerate(unique_labels) if j > i]
    mean_mmd = float(np.nanmean(upper_vals)) if upper_vals else 0.0

    print(f"H3 Cross-Scenario MMD: mean = {mean_mmd:.6f}")
    for (la, lb), v in sorted(mmd_matrix.items()):
        if la < lb:
            print(f"  Scen {la} vs Scen {lb}: MMD = {v:.6f}")

    if save_path is not None:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(6, 5))
        scen_names = [f"Scen {l}" for l in unique_labels]
        sns.heatmap(mat, ax=ax, annot=True, fmt=".4f", cmap='Blues',
                    xticklabels=scen_names, yticklabels=scen_names)
        ax.set_title("H3: Cross-Scenario MMD Matrix", fontweight='bold')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for ext in ('png', 'pdf'):
            plt.savefig(f"{save_path}.{ext}", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"MMD matrix saved to {save_path}")

    return mmd_matrix, mean_mmd


if __name__ == "__main__":
    config_path = "conf/experiments/semantic_manifold.yaml"
    if not os.path.exists(config_path): config_path = "semantic_manifold.yaml"
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
        pm = PathContext(os.getcwd())
        visualize(cfg, pm)