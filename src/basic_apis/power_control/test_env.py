"""
Test Gym environment implementation.

Verifies:
1. Environment can be created and reset
2. Step function works correctly
3. Observation/action spaces are valid
4. Metrics are computed correctly
"""
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.basic_apis.power_control.env import PowerControlEnv


def test_env_creation():
    """Test environment creation."""
    print("\n[Test 1/5] Environment Creation")
    print("-" * 50)

    env = PowerControlEnv(
        K=5,
        N=5,
        episode_length=100,
        seed=42,
    )

    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Power levels: {env.power_levels[:3]}...{env.power_levels[-1]:.2f}")
    print(f"  Fairness threshold: {env.fairness_threshold:.3f}")
    print("  ✓ PASS")
    return True


def test_reset():
    """Test reset function."""
    print("\n[Test 2/5] Reset Function")
    print("-" * 50)

    env = PowerControlEnv(K=5, N=5, episode_length=100, seed=42)
    obs, info = env.reset()

    print(f"  Observation shape: {obs.shape}")
    print(f"  Expected shape: {env.observation_space.shape}")
    print(f"  Observation range: [{obs.min():.2e}, {obs.max():.2e}]")

    assert obs.shape == env.observation_space.shape, "Observation shape mismatch"
    assert env.observation_space.contains(obs), "Observation out of bounds"

    print("  ✓ PASS")
    return True


def test_step():
    """Test step function."""
    print("\n[Test 3/5] Step Function")
    print("-" * 50)

    env = PowerControlEnv(K=5, N=5, episode_length=100, seed=42)
    obs, _ = env.reset()

    # Take random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"  Action: {action}")
    print(f"  Reward: {reward:.4f}")
    print(f"  Sum-rate: {info['sum_rate']:.4f}")
    print(f"  QoS violation: {info['qos_violation']:.3f}")
    print(f"  Jain's fairness: {info['jains_fairness']:.3f}")
    print(f"  Terminated: {terminated}")

    assert obs.shape == env.observation_space.shape, "Observation shape mismatch"
    assert isinstance(reward, (float, np.floating)), "Reward must be float"
    assert "sum_rate" in info, "Missing sum_rate in info"
    assert "qos_violation" in info, "Missing qos_violation in info"
    assert "jains_fairness" in info, "Missing jains_fairness in info"

    print("  ✓ PASS")
    return True


def test_episode():
    """Test full episode."""
    print("\n[Test 4/5] Full Episode")
    print("-" * 50)

    env = PowerControlEnv(K=5, N=5, episode_length=50, seed=42)
    obs, _ = env.reset()

    total_reward = 0
    steps = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Episode metrics:")
    print(f"    Avg sum-rate: {info['episode']['avg_sum_rate']:.2f}")
    print(f"    Avg QoS viol: {info['episode']['avg_qos_violation']:.3f}")
    print(f"    Avg fairness: {info['episode']['avg_jains_fairness']:.3f}")

    assert steps == 50, f"Expected 50 steps, got {steps}"
    assert "episode" in info, "Missing episode metrics"

    print("  ✓ PASS")
    return True


def test_custom_reward():
    """Test custom reward function injection."""
    print("\n[Test 5/5] Custom Reward Function")
    print("-" * 50)

    def custom_reward(rates, qos_viol, fairness):
        """Hybrid reward: sum-rate - QoS penalty - fairness penalty."""
        return float(np.sum(rates) - 10.0 * qos_viol - 5.0 * max(0, 0.6 - fairness))

    env = PowerControlEnv(K=5, N=5, episode_length=10, seed=42)
    env.set_reward_fn(custom_reward)

    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # Manually compute expected reward
    expected_reward = custom_reward(
        info["rates"], info["qos_violation"], info["jains_fairness"]
    )

    print(f"  Reward: {reward:.4f}")
    print(f"  Expected: {expected_reward:.4f}")
    print(f"  Difference: {abs(reward - expected_reward):.2e}")

    assert abs(reward - expected_reward) < 1e-6, "Custom reward mismatch"

    print("  ✓ PASS")
    return True


def main():
    print("=" * 70)
    print("Gym Environment Test Suite")
    print("=" * 70)

    results = []
    results.append(test_env_creation())
    results.append(test_reset())
    results.append(test_step())
    results.append(test_episode())
    results.append(test_custom_reward())

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    if all(results):
        print("✓ All tests passed - Gym environment is correct")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
