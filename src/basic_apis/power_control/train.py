"""
PPO training script for power control environment.

Supports:
- Stable-Baselines3 PPO with MultiDiscrete action space
- Hydra configuration
- Custom reward function injection (for ReEvo)
- Metrics logging and evaluation
"""
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from .env import PowerControlEnv


class MetricsCallback(BaseCallback):
    """Callback to log episode metrics during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_sum_rates = []
        self.episode_qos_violations = []
        self.episode_fairness = []
        # Track per-episode metrics
        self.current_ep_sum_rates = []
        self.current_ep_qos_viols = []
        self.current_ep_fairness = []

    def _on_step(self) -> bool:
        # Collect step-level metrics
        for idx, done in enumerate(self.locals["dones"]):
            info = self.locals["infos"][idx]

            # Collect step metrics
            if "sum_rate" in info:
                self.current_ep_sum_rates.append(info["sum_rate"])
            if "qos_violation" in info:
                self.current_ep_qos_viols.append(info["qos_violation"])
            if "jains_fairness" in info:
                self.current_ep_fairness.append(info["jains_fairness"])

            # When episode ends, aggregate
            if done:
                if len(self.current_ep_sum_rates) > 0:
                    self.episode_sum_rates.append(np.mean(self.current_ep_sum_rates))
                    self.episode_qos_violations.append(np.mean(self.current_ep_qos_viols))
                    self.episode_fairness.append(np.mean(self.current_ep_fairness))

                    # Reset for next episode
                    self.current_ep_sum_rates = []
                    self.current_ep_qos_viols = []
                    self.current_ep_fairness = []

        return True


def make_vec_env(env_config: Dict, n_envs: int = 1) -> DummyVecEnv:
    """Create vectorized environment with Monitor wrapper."""

    def _make_env():
        env = PowerControlEnv(**env_config)
        env = Monitor(env)
        return env

    return DummyVecEnv([_make_env for _ in range(n_envs)])


def train(
    env_config: Dict,
    ppo_config: Dict,
    total_timesteps: int,
    save_path: Optional[Path] = None,
    reward_fn: Optional[Callable] = None,
    seed: int = 42,
    verbose: int = 1,
) -> Dict[str, float]:
    """
    Train PPO agent on power control environment.

    Args:
        env_config: Environment configuration
        ppo_config: PPO hyperparameters
        total_timesteps: Total training steps
        save_path: Path to save trained model
        reward_fn: Custom reward function (for ReEvo)
        seed: Random seed
        verbose: Verbosity level

    Returns:
        metrics: Training and evaluation metrics
    """
    # Create environment
    env = make_vec_env(env_config, n_envs=1)

    # Inject custom reward function if provided
    if reward_fn is not None:
        env.envs[0].set_reward_fn(reward_fn)

    # Create PPO agent
    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        verbose=verbose,
        **ppo_config,
    )

    # Create callback
    callback = MetricsCallback(verbose=verbose)

    # Train
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save model
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        if verbose:
            print(f"Model saved to {save_path}")

    # Evaluate
    eval_metrics = evaluate(model, env_config, n_episodes=10, seed=seed + 1000)

    # Aggregate training metrics
    train_metrics = {
        "train_avg_sum_rate": float(np.mean(callback.episode_sum_rates[-100:])),
        "train_avg_qos_violation": float(
            np.mean(callback.episode_qos_violations[-100:])
        ),
        "train_avg_fairness": float(np.mean(callback.episode_fairness[-100:])),
    }

    # Combine
    metrics = {**train_metrics, **eval_metrics}

    return metrics


def evaluate(
    model: PPO,
    env_config: Dict,
    n_episodes: int = 10,
    seed: int = 42,
    verbose: int = 0,
) -> Dict[str, float]:
    """
    Evaluate trained PPO agent.

    Args:
        model: Trained PPO model
        env_config: Environment configuration
        n_episodes: Number of evaluation episodes
        seed: Random seed
        verbose: Verbosity level

    Returns:
        metrics: Evaluation metrics
    """
    # Create evaluation environment
    env = PowerControlEnv(**env_config)

    episode_sum_rates = []
    episode_qos_violations = []
    episode_fairness = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_sum_rates = []
        ep_qos_viols = []
        ep_fairness = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_sum_rates.append(info["sum_rate"])
            ep_qos_viols.append(info["qos_violation"])
            ep_fairness.append(info["jains_fairness"])

        episode_sum_rates.append(np.mean(ep_sum_rates))
        episode_qos_violations.append(np.mean(ep_qos_viols))
        episode_fairness.append(np.mean(ep_fairness))

        if verbose:
            print(
                f"Episode {ep+1}/{n_episodes}: "
                f"sum_rate={episode_sum_rates[-1]:.2f}, "
                f"qos_viol={episode_qos_violations[-1]:.3f}, "
                f"fairness={episode_fairness[-1]:.3f}"
            )

    metrics = {
        "eval_avg_sum_rate": float(np.mean(episode_sum_rates)),
        "eval_std_sum_rate": float(np.std(episode_sum_rates)),
        "eval_avg_qos_violation": float(np.mean(episode_qos_violations)),
        "eval_std_qos_violation": float(np.std(episode_qos_violations)),
        "eval_avg_fairness": float(np.mean(episode_fairness)),
        "eval_std_fairness": float(np.std(episode_fairness)),
    }

    return metrics


def load_and_evaluate(
    model_path: Path,
    env_config: Dict,
    n_episodes: int = 10,
    seed: int = 42,
) -> Dict[str, float]:
    """Load trained model and evaluate."""
    model = PPO.load(model_path)
    return evaluate(model, env_config, n_episodes, seed)
