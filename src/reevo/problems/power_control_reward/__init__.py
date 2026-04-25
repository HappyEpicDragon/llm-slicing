"""
Power Control Reward Evolution Problem.

Multi-cell power control with QoS and fairness constraints.
K-user interference channel with discrete power levels.
"""

from .eval import main as eval_main
from .baseline_reward import compute_reward as baseline_reward

__all__ = ["eval_main", "baseline_reward"]
