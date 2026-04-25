"""
Gym environment for K-user interference channel power control.

Features:
- Global observation: full channel matrix H (K×K)
- MultiDiscrete action: discrete power levels per user
- Reward: injectable function (for ReEvo)
- Metrics: QoS violation rate, Jain's fairness index
"""
from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .channel import compute_rates, generate_channel_sequence


class PowerControlEnv(gym.Env):
    """
    K-user interference channel with discrete power control.

    Observation:
        - H_flat: (K*K,) flattened channel matrix
        - prev_rates: (K,) rates from previous step
        - prev_actions_onehot: (K*n_power_levels,) one-hot encoded previous actions

    Action:
        - MultiDiscrete([n_power_levels] * K): discrete power level per user

    Reward:
        - Computed by injectable reward_fn(rates, qos_violations, jains_fairness)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        K: int = 10,
        N: int = 20,
        R_defined: float = 400.0,
        min_dist: float = 35.0,
        P_max: float = 6.31,
        R_min: float = 1.0,
        n_power_levels: int = 10,
        noise_var: float = 3.98e-15,
        shadowing_dev: float = 10.0,
        dcor: float = 10.0,
        fd: float = 10.0,
        T: float = 0.02,
        episode_length: int = 1000,
        seed: Optional[int] = None,
        reward_fn: Optional[Callable] = None,
    ):
        """
        Initialize power control environment.

        Args:
            K: Number of base stations
            N: Number of users (must equal K for K-user IC)
            R_defined: Cell radius (m)
            min_dist: Minimum UE-BS distance (m)
            P_max: Maximum power per user (W)
            R_min: Minimum rate requirement (bps/Hz)
            n_power_levels: Number of discrete power levels
            noise_var: Noise power (W)
            shadowing_dev: Shadowing std dev (dB)
            dcor: Shadowing decorrelation distance (m)
            fd: Doppler frequency (Hz)
            T: Time step (s)
            episode_length: Steps per episode
            seed: RNG seed
            reward_fn: Reward function(rates, qos_viol, fairness) -> float
        """
        super().__init__()

        # Validate K-user IC constraint
        assert N == K, f"K-user IC requires N=K, got N={N}, K={K}"

        self.K = K
        self.N = N
        self.R_defined = R_defined
        self.min_dist = min_dist
        self.P_max = P_max
        self.R_min = R_min
        self.n_power_levels = n_power_levels
        self.noise_var = noise_var
        self.shadowing_dev = shadowing_dev
        self.dcor = dcor
        self.fd = fd
        self.T = T
        self.episode_length = episode_length
        self._seed = seed

        # Discrete power levels: [0, P_max]
        self.power_levels = np.linspace(0, P_max, n_power_levels)

        # Adaptive fairness threshold: τ(K) = 0.5 + 0.5/K
        self.fairness_threshold = 0.5 + 0.5 / K

        # Action space: MultiDiscrete([n_power_levels] * K)
        self.action_space = spaces.MultiDiscrete([n_power_levels] * K)

        # Observation space: H_flat + prev_rates + prev_actions_onehot
        obs_dim = K * K + K + K * n_power_levels
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Reward function (default: sum-rate)
        self.reward_fn = reward_fn or self._default_reward

        # Episode state
        self.H_all: Optional[np.ndarray] = None
        self.current_step = 0
        self.prev_rates = np.zeros(K)
        self.prev_actions = np.zeros(K, dtype=int)

        # Metrics tracking
        self.episode_metrics = {
            "sum_rates": [],
            "qos_violations": [],
            "jains_fairness": [],
        }

    def _default_reward(
        self, rates: np.ndarray, qos_viol: float, fairness: float
    ) -> float:
        """Default reward: sum-rate only."""
        return float(np.sum(rates))

    def set_reward_fn(self, reward_fn: Callable) -> None:
        """Inject custom reward function (for ReEvo)."""
        self.reward_fn = reward_fn

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment and generate new channel sequence."""
        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed

        # Generate channel sequence for this episode
        self.H_all, _ = generate_channel_sequence(
            N=self.N,
            K=self.K,
            R_defined=self.R_defined,
            min_dist=self.min_dist,
            total_samples=self.episode_length,
            shadowing_dev=self.shadowing_dev,
            dcor=self.dcor,
            equal_number_for_BS=True,
            fd=self.fd,
            T=self.T,
            rayleigh_var=1.0,
            seed=self._seed,
        )

        self.current_step = 0
        self.prev_rates = np.zeros(self.K)
        self.prev_actions = np.zeros(self.K, dtype=int)

        # Reset metrics
        self.episode_metrics = {
            "sum_rates": [],
            "qos_violations": [],
            "jains_fairness": [],
        }

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step."""
        # Map discrete actions to power levels
        powers = self.power_levels[action]

        # Get current channel
        H = self.H_all[self.current_step]

        # Compute rates
        rates = compute_rates(H, powers, self.noise_var)

        # Compute metrics
        qos_viol = float(np.mean(rates < self.R_min))
        jains_fairness = self._compute_jains_fairness(rates)

        # Compute reward
        reward = self.reward_fn(rates, qos_viol, jains_fairness)

        # Track metrics
        self.episode_metrics["sum_rates"].append(float(np.sum(rates)))
        self.episode_metrics["qos_violations"].append(qos_viol)
        self.episode_metrics["jains_fairness"].append(jains_fairness)

        # Update state
        self.prev_rates = rates
        self.prev_actions = action
        self.current_step += 1

        # Check termination
        terminated = self.current_step >= self.episode_length
        truncated = False

        # Get next observation
        obs = self._get_obs()

        # Info dict
        info = {
            "sum_rate": float(np.sum(rates)),
            "qos_violation": qos_viol,
            "jains_fairness": jains_fairness,
            "rates": rates.copy(),
            "powers": powers.copy(),
        }

        if terminated:
            info["episode"] = {
                "avg_sum_rate": float(np.mean(self.episode_metrics["sum_rates"])),
                "avg_qos_violation": float(
                    np.mean(self.episode_metrics["qos_violations"])
                ),
                "avg_jains_fairness": float(
                    np.mean(self.episode_metrics["jains_fairness"])
                ),
            }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        if self.current_step >= self.episode_length:
            # Episode ended, return zero obs
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        H = self.H_all[self.current_step]

        # Flatten channel matrix
        H_flat = H.flatten()

        # One-hot encode previous actions
        prev_actions_onehot = np.zeros(self.K * self.n_power_levels)
        for k in range(self.K):
            idx = k * self.n_power_levels + self.prev_actions[k]
            prev_actions_onehot[idx] = 1.0

        # Concatenate
        obs = np.concatenate([H_flat, self.prev_rates, prev_actions_onehot])

        return obs.astype(np.float32)

    def _compute_jains_fairness(self, rates: np.ndarray) -> float:
        """
        Compute Jain's fairness index.

        J = (Σr_k)² / (K·Σr_k²)
        """
        sum_rates = np.sum(rates)
        sum_rates_sq = np.sum(rates**2)
        return float(sum_rates**2 / (self.K * sum_rates_sq + 1e-12))


def make_env(env_config: Dict[str, Any]) -> PowerControlEnv:
    """Factory function for creating environment (for SB3 VecEnv)."""
    return PowerControlEnv(**env_config)
