"""Quick smoke test for benchmark functions."""
import sys
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.basic_apis.power_control.benchmarks import run_all_benchmarks
from src.basic_apis.power_control.channel import generate_channel_sequence

# Load config
cfg_path = project_root / "conf" / "environment" / "power_control_env.yaml"
cfg = OmegaConf.load(cfg_path)

# Test with K=4, 5 episodes
K = 4
n_episodes = 5

print(f"Testing benchmarks with K={K}, episodes={n_episodes}")

# Generate channels
H_all, meta = generate_channel_sequence(
    K=K,
    N=K,
    R_defined=400.0,
    min_dist=35.0,
    total_samples=n_episodes,
    shadowing_dev=10.0,
    dcor=50.0,
    fd=10.0,
    T=0.02,
    seed=42
)

print(f"Generated H_all shape: {H_all.shape}")

# Equal weights
weights = np.ones(K)

# Run benchmarks
results = run_all_benchmarks(
    H_all=H_all,
    Pmax=6.31,
    noise_var=3.98e-15,
    weights=weights
)

print("\nResults:")
for algo_name, metrics in results.items():
    print(f"  {algo_name}: {metrics}")

print("\n✓ Benchmark test passed!")
