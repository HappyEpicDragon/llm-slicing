"""
Full validation test for Power Control integration.

Tests:
1. Complete PPO training (K=10, 1M timesteps)
2. Multi-scenario testing (K=4, 8, 16, each 1M timesteps)
3. Benchmark algorithms comparison (WMMSE, FP, Random, MaxPower)
"""
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.basic_apis.power_control.train import train, load_and_evaluate
from src.basic_apis.power_control.benchmarks import run_all_benchmarks
from src.basic_apis.power_control.channel import generate_channel_sequence


def test_ppo_training(K: int, total_timesteps: int, seed: int = 42) -> Dict:
    """Test PPO training for given K."""
    print(f"\n{'='*60}")
    print(f"Testing PPO Training: K={K}, timesteps={total_timesteps}")
    print(f"{'='*60}")

    # Load config
    cfg_path = project_root / "conf" / "environment" / "power_control_env.yaml"
    cfg = OmegaConf.load(cfg_path)

    # Override K
    cfg.environment.K = K
    cfg.environment.N = K

    # Train
    start_time = time.time()
    metrics = train(
        env_config=dict(cfg.environment),
        ppo_config=dict(cfg.ppo),
        total_timesteps=total_timesteps,
        save_path=project_root / f"data/power_control/models/ppo_K{K}.zip",
        reward_fn=None,
        seed=seed,
        verbose=1
    )
    elapsed = time.time() - start_time

    print(f"\n[RESULTS K={K}]")
    print(f"  Training time: {elapsed/60:.1f} minutes")
    print(f"  Sum Rate: {metrics['eval_avg_sum_rate']:.2f} ± {metrics['eval_std_sum_rate']:.2f} bps/Hz")
    print(f"  QoS Violation: {metrics['eval_avg_qos_violation']:.4f} ± {metrics['eval_std_qos_violation']:.4f}")
    print(f"  Jain's Fairness: {metrics['eval_avg_fairness']:.4f} ± {metrics['eval_std_fairness']:.4f}")

    return metrics


def test_benchmarks(K: int, n_episodes: int = 100, seed: int = 42) -> Dict:
    """Test benchmark algorithms."""
    print(f"\n{'='*60}")
    print(f"Testing Benchmarks: K={K}, episodes={n_episodes}")
    print(f"{'='*60}")

    # Load config
    cfg_path = project_root / "conf" / "environment" / "power_control_env.yaml"
    cfg = OmegaConf.load(cfg_path)

    env_config = dict(cfg.environment)
    env_config['K'] = K
    env_config['N'] = K

    # Generate test channels (batch generation for efficiency)
    np.random.seed(seed)
    H_all, _ = generate_channel_sequence(
        K=K,
        N=K,
        R_defined=env_config.get('R_defined', 400.0),
        min_dist=env_config.get('min_dist', 35.0),
        total_samples=n_episodes,
        shadowing_dev=env_config.get('shadowing_dev', 10.0),
        dcor=env_config.get('dcor', 50.0),
        fd=env_config.get('fd', 10.0),
        T=env_config.get('T', 0.02),
        seed=seed
    )

    # Equal weights for all users
    weights = np.ones(K)

    start_time = time.time()

    # Run all benchmarks
    results = run_all_benchmarks(
        H_all=H_all,
        Pmax=env_config.get('P_max', 6.31),
        noise_var=env_config.get('noise_var', 3.98e-15),
        weights=weights
    )

    elapsed = time.time() - start_time

    print(f"\n[BENCHMARK RESULTS K={K}]")
    print(f"  Test time: {elapsed:.1f} seconds")

    # Extract and print results
    output = {}
    for algo_name, metrics in results.items():
        avg_sumrate = metrics['avg_sumrate']
        print(f"  {algo_name:10s}: {avg_sumrate:6.2f} bps/Hz")
        output[algo_name.lower()] = {
            'avg_sumrate': avg_sumrate,
            'avg_time': metrics.get('avg_time', 0.0),
            'avg_iters': metrics.get('avg_iters', 0.0)
        }

    return output


def main():
    """Run full validation suite."""
    print("\n" + "="*60)
    print("POWER CONTROL INTEGRATION - FULL VALIDATION")
    print("="*60)

    # Create output directory
    output_dir = project_root / "data" / "power_control" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    total_start = time.time()

    # Test 1: Complete PPO training for K=10
    print("\n\n### TEST 1: Complete PPO Training (K=10) ###")
    results['ppo_K10'] = test_ppo_training(K=10, total_timesteps=1_000_000, seed=42)

    # Test 2: Multi-scenario PPO training
    print("\n\n### TEST 2: Multi-Scenario PPO Training ###")
    for K in [4, 8, 16]:
        results[f'ppo_K{K}'] = test_ppo_training(K=K, total_timesteps=1_000_000, seed=42)

    # Test 3: Benchmark algorithms
    print("\n\n### TEST 3: Benchmark Algorithms ###")
    for K in [4, 8, 10, 16]:
        results[f'benchmarks_K{K}'] = test_benchmarks(K=K, n_episodes=100, seed=42)

    total_elapsed = time.time() - total_start

    # Summary
    print("\n\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total time: {total_elapsed/3600:.2f} hours")

    print("\n### PPO Performance ###")
    for K in [4, 8, 10, 16]:
        key = f'ppo_K{K}'
        if key in results:
            m = results[key]
            print(f"K={K:2d}: Sum Rate={m['eval_avg_sum_rate']:6.2f} bps/Hz, "
                  f"QoS Viol={m['eval_avg_qos_violation']:.4f}, "
                  f"Fairness={m['eval_avg_fairness']:.4f}")

    print("\n### Benchmark Comparison (WMMSE) ###")
    for K in [4, 8, 10, 16]:
        key = f'benchmarks_K{K}'
        if key in results:
            wmmse = results[key]['wmmse']
            print(f"K={K:2d}: {wmmse['mean']:6.2f} ± {wmmse['std']:5.2f} bps/Hz")

    # Save results
    import json
    output_file = output_dir / "validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
