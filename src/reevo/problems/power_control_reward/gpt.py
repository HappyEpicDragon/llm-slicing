"""
LLM-generated reward function for power control.
This file is replaced by ReEvo during evolution.

Signature: compute_reward(rates, qos_violations, jains_fairness) -> float
  - rates: np.ndarray of shape (K,), per-user achievable rate (bps/Hz)
  - qos_violations: float in [0, 1], fraction of users below R_min
  - jains_fairness: float in [0, 1], Jain's fairness index
"""
import numpy as np


def compute_reward(rates: np.ndarray, qos_violations: float, jains_fairness: float) -> float:
    """
    Hybrid reward: sum-rate backbone + QoS penalty + fairness penalty.

    This is the initial seed function. ReEvo will evolve better variants.
    """
    K = len(rates)
    sum_rate = float(np.sum(rates))

    # QoS penalty: penalize users below minimum rate
    qos_penalty = -10.0 * qos_violations

    # Fairness penalty: penalize when below adaptive threshold
    fairness_threshold = 0.5 + 0.5 / K
    fairness_penalty = -5.0 * max(0.0, fairness_threshold - jains_fairness)

    return sum_rate + qos_penalty + fairness_penalty
