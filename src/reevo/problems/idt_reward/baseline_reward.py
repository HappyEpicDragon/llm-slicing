"""
baseline_reward.py — 当前人工设计的 reward function（公式参考）

这是 _builtin_compute_reward 的独立版本，作为：
  1. seed_func.txt 的来源（进化初始种子）
  2. 人工 baseline 性能参考
  3. 快速手工测试 gpt.py 的替代
"""
import numpy as np


def compute_reward(
    slice_min_margins: list,
    slice_mean_margins: list,
    slice_metric_margins: list,
    is_hp: list,
    num_active_slices: int,
    prev_min_margins: list,
    prev_mean_margins: list,
    slice_mean_buffer_occ: list,
    slice_max_buffer_occ: list,
) -> float:
    """
    当前人工设计的 baseline reward。

    Args:
        slice_min_margins     : per-slice worst-user composite margin, [-1, 1]
        slice_mean_margins    : per-slice average composite margin, [-1, 1]
        slice_metric_margins  : list of {"thr": float, "rel": float, "lat": float} mean margins per slice
        is_hp                 : True if the slice is High-Priority
        num_active_slices     : number of active slices
        prev_min_margins      : previous TTI slice_min_margins (0.0 at episode start)
        prev_mean_margins     : previous TTI slice_mean_margins (0.0 at episode start)
        slice_mean_buffer_occ : per-slice mean buffer occupancy in [0, 1]
        slice_max_buffer_occ  : per-slice max buffer occupancy in [0, 1]

    Returns:
        scalar float reward
    """
    if num_active_slices == 0:
        return 0.0

    slice_scores = np.array(slice_min_margins, dtype=np.float64)
    slice_priorities = np.array([1 if hp else 0 for hp in is_hp], dtype=np.float64)

    # 1. NHP Capping: positive scores are down-weighted to discourage over-provisioning
    capped_scores = np.where(slice_scores > 0, slice_scores * 0.2, slice_scores)

    # 2. HP Exponential Penalty: severe punishment when any HP slice is violated
    hp_penalty = 0.0
    hp_violation_mask = (slice_scores < 0) & (slice_priorities > 0)
    if np.any(hp_violation_mask):
        hp_bad_scores = slice_scores[hp_violation_mask]
        hp_penalty = -10.0 - np.sum(np.exp(np.abs(hp_bad_scores) * 2.0))

    # 3. Base Reward
    min_score = np.min(capped_scores)
    if min_score < 0:
        base_reward = min_score * 2.0
    else:
        base_reward = float(np.mean(capped_scores))

    # 4. Risk Penalty: penalize high buffer occupancy to preempt packet drops
    risk_coef = 0.2
    max_buf = np.array(slice_max_buffer_occ, dtype=np.float64)
    danger_mask = max_buf > 0.8
    risk_loss = 0.0
    if np.any(danger_mask):
        risk_loss = np.sum(np.exp(max_buf[danger_mask] * 5.0))
    risk_penalty = -1.0 * risk_loss * risk_coef

    return float(base_reward + hp_penalty + risk_penalty)
