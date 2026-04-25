import numpy as np
import numpy as np

def compute_reward_v2(
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
    if num_active_slices == 0:
        return 0.0
    
    # Convert inputs to numpy arrays
    slice_min_margins_arr = np.array(slice_min_margins, dtype=np.float64)
    slice_mean_margins_arr = np.array(slice_mean_margins, dtype=np.float64)
    is_hp_arr = np.array(is_hp, dtype=bool)
    prev_min_margins_arr = np.array(prev_min_margins, dtype=np.float64)
    prev_mean_margins_arr = np.array(prev_mean_margins, dtype=np.float64)
    slice_mean_buffer_occ_arr = np.array(slice_mean_buffer_occ, dtype=np.float64)
    slice_max_buffer_occ_arr = np.array(slice_max_buffer_occ, dtype=np.float64)
    
    # Mask definitions
    hp_mask = is_hp_arr
    nhp_mask = ~hp_mask
    violation_mask = slice_min_margins_arr < 0
    hp_violation_mask = hp_mask & violation_mask
    nhp_violation_mask = nhp_mask & violation_mask
    safe_mask = slice_min_margins_arr >= 0
    hp_safe_mask = hp_mask & safe_mask
    nhp_safe_mask = nhp_mask & safe_mask
    
    # --- Core reward components ---
    
    # 1. High Priority Protection (Heavy penalty for HP violations)
    hp_violation_penalty = 0.0
    if np.any(hp_violation_mask):
        hp_min_margins = slice_min_margins_arr[hp_violation_mask]
        # Exponential penalty with severity scaling
        hp_violation_penalty = -20.0 * np.exp(np.mean(np.abs(hp_min_margins) * 3.0))
    
    # 2. HP Safe Reward (Reward for HP slices that are safe)
    hp_safe_reward = 0.0
    if np.any(hp_safe_mask):
        hp_min_margins_safe = slice_min_margins_arr[hp_safe_mask]
        # Encourage higher margins for HP slices
        hp_safe_reward = 3.0 * np.mean(hp_min_margins_safe)
    
    # 3. NHP Violation Penalty (Moderate penalty for NHP violations)
    nhp_violation_penalty = 0.0
    if np.any(nhp_violation_mask):
        nhp_min_margins = slice_min_margins_arr[nhp_violation_mask]
        nhp_violation_penalty = -5.0 * np.mean(np.abs(nhp_min_margins))
    
    # 4. NHP Safe Reward (Encourage NHP slices to meet SLA)
    nhp_safe_reward = 0.0
    if np.any(nhp_safe_mask):
        nhp_min_margins_safe = slice_min_margins_arr[nhp_safe_mask]
        nhp_safe_reward = np.mean(nhp_min_margins_safe)
    
    # 5. Margin Improvement Bonus (Reward for improving from previous TTI)
    improvement_bonus = 0.0
    min_margin_diff = slice_min_margins_arr - prev_min_margins_arr
    positive_improvements = min_margin_diff > 0
    if np.any(positive_improvements):
        improvement_values = min_margin_diff[positive_improvements]
        # Weight improvements higher for HP slices
        hp_improvement_mask = hp_mask & positive_improvements
        if np.any(hp_improvement_mask):
            hp_improvement = min_margin_diff[hp_improvement_mask]
            improvement_bonus += 2.0 * np.mean(hp_improvement)
        nhp_improvement_mask = nhp_mask & positive_improvements
        if np.any(nhp_improvement_mask):
            nhp_improvement = min_margin_diff[nhp_improvement_mask]
            improvement_bonus += 0.5 * np.mean(nhp_improvement)
    
    # 6. Buffer Risk Penalty (Penalize high buffer occupancy)
    buffer_risk_penalty = 0.0
    danger_mask = slice_max_buffer_occ_arr > 0.9
    if np.any(danger_mask):
        max_buffer_danger = slice_max_buffer_occ_arr[danger_mask]
        buffer_risk_penalty = -10.0 * np.exp(np.mean(max_buffer_danger * 4.0))
    
    # 7. Buffer Stability Bonus (Reward for managing buffers)
    buffer_stability_bonus = 0.0
    moderate_mask = slice_max_buffer_occ_arr <= 0.5
    if np.any(moderate_mask):
        # Encourage keeping buffers at moderate levels
        buffer_stability_bonus = 1.0 * np.sum(slice_max_buffer_occ_arr[moderate_mask] <= 0.3)
    
    # 8. Metric Balance Bonus (Reward for balanced metric fulfillment)
    metric_balance_bonus = -5.0
    if slice_metric_margins:
        for slice_margin_dict in slice_metric_margins:
            thr_margin = slice_margin_dict.get("thr", 0)
            rel_margin = slice_margin_dict.get("rel", 0)
            lat_margin = slice_margin_dict.get("lat", 0)
            metric_spread = abs(thr_margin - rel_margin) + abs(thr_margin - lat_margin) + abs(rel_margin - lat_margin)
            if metric_spread < 0.5:
                metric_balance_bonus += 0.3
    
    # --- Composite reward calculation ---
    reward = hp_safe_reward + nhp_safe_reward + improvement_bonus + buffer_stability_bonus + metric_balance_bonus
    reward += hp_violation_penalty + nhp_violation_penalty + buffer_risk_penalty
    
    # Clip reward to reasonable range (environment will further clamp to [-10,10])
    reward = np.clip(reward, -15.0, 15.0)
    
    return float(reward)
