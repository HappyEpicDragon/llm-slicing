"""Test WMMSE/FP with realistic channel from generate_channel_sequence."""
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.basic_apis.power_control.channel import (
    compute_rates,
    fp,
    generate_channel_sequence,
    wmmse,
)

print("=" * 70)
print("Test with Realistic Channel")
print("=" * 70)

# Generate realistic channel
H_all, meta = generate_channel_sequence(
    N=10,
    K=5,
    R_defined=400.0,
    min_dist=35.0,
    total_samples=10,
    shadowing_dev=10.0,
    dcor=10.0,
    equal_number_for_BS=True,
    fd=10.0,
    T=0.02,
    rayleigh_var=1.0,
    seed=42,
)

# Use first channel snapshot
H = H_all[0]
N = H.shape[0]

print(f"\nChannel properties:")
print(f"  Shape: {H.shape}")
print(f"  Range: [{H.min():.2e}, {H.max():.2e}]")
print(f"  Diagonal: {np.diag(H)[:3]}")

# Physical params
Pmax_dB = 38.0 - 30
Pmax = np.power(10.0, Pmax_dB / 10)
n0_dB = -114.0 - 30
noise_var = np.power(10.0, n0_dB / 10)
weights = np.ones(N)

print(f"\nPhysical params:")
print(f"  Pmax: {Pmax:.2f} W ({Pmax_dB:.1f} dBW)")
print(f"  Noise: {noise_var:.2e} W ({n0_dB:.1f} dBm)")

# Run algorithms
print("\n[WMMSE]")
p_wmmse, stats_wmmse = wmmse(N, H, Pmax, noise_var, weights)
r_wmmse = compute_rates(H, p_wmmse, noise_var)
print(f"  Iterations: {stats_wmmse[1]}")
print(f"  Power range: [{p_wmmse.min():.2e}, {p_wmmse.max():.2e}]")
print(f"  Sum-rate: {np.sum(r_wmmse):.4f} bps/Hz")

print("\n[FP]")
p_fp, stats_fp = fp(N, H, Pmax, noise_var, weights)
r_fp = compute_rates(H, p_fp, noise_var)
print(f"  Iterations: {stats_fp[1]}")
print(f"  Power range: [{p_fp.min():.2e}, {p_fp.max():.2e}]")
print(f"  Sum-rate: {np.sum(r_fp):.4f} bps/Hz")

print("\n[Random]")
p_random = Pmax * np.random.rand(N)
r_random = compute_rates(H, p_random, noise_var)
print(f"  Sum-rate: {np.sum(r_random):.4f} bps/Hz")

print("\n[MaxPower]")
p_max = Pmax * np.ones(N)
r_max = compute_rates(H, p_max, noise_var)
print(f"  Sum-rate: {np.sum(r_max):.4f} bps/Hz")

print("\n" + "=" * 70)
print("Monotonicity check:")
print("=" * 70)
sr_wmmse = np.sum(r_wmmse)
sr_fp = np.sum(r_fp)
sr_random = np.sum(r_random)
sr_max = np.sum(r_max)

print(f"WMMSE:    {sr_wmmse:.4f}")
print(f"FP:       {sr_fp:.4f}")
print(f"Random:   {sr_random:.4f}")
print(f"MaxPower: {sr_max:.4f}")

if sr_wmmse >= sr_fp - 0.01 and sr_fp >= sr_random - 0.01:
    print("\n✓ PASS: WMMSE ≥ FP ≥ Random (correct)")
else:
    print(f"\n✗ FAIL: Monotonicity violated")
