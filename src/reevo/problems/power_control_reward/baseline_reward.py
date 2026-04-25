"""
Manual baseline reward for power control.
Nasir-Guo backbone + QoS term + Fairness term (hybrid extension).

Reference: Nasir & Guo, Asilomar 2020
"""
import numpy as np


def compute_reward(rates: np.ndarray, qos_violations: float, jains_fairness: float) -> float:
    """
    Hybrid baseline reward combining sum-rate, QoS, and fairness.

    Args:
        rates: per-user achievable rate (K,), bps/Hz
        qos_violations: fraction of users below R_min, in [0, 1]
        jains_fairness: Jain's fairness index, in [0, 1]

    Returns:
        scalar reward
    """
    K = len(rates)
    sum_rate = float(np.sum(rates))

    # QoS penalty (ours): hard penalize violations
    qos_penalty = -10.0 * qos_violations

    # Fairness penalty (ours): adaptive threshold τ(K) = 0.5 + 0.5/K
    fairness_threshold = 0.5 + 0.5 / K
    fairness_penalty = -5.0 * max(0.0, fairness_threshold - jains_fairness)

    return sum_rate + qos_penalty + fairness_penalty
