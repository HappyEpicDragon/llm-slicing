"""Compare WMMSE and FP convergence behavior."""
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.basic_apis.power_control.channel import compute_rates, fp, wmmse

np.random.seed(42)
N = 10
H = np.abs(np.random.randn(N, N) + 1j * np.random.randn(N, N))
Pmax = 6.31
noise_var = 3.98e-15
weights = np.ones(N)

print("Convergence Comparison")
print("=" * 70)

# Run WMMSE
print("\n[WMMSE]")
p_wmmse, stats_wmmse = wmmse(N, H, Pmax, noise_var, weights)
r_wmmse = compute_rates(H, p_wmmse, noise_var)
print(f"  Iterations: {stats_wmmse[1]}")
print(f"  Time: {stats_wmmse[0]:.4f}s")
print(f"  Power: {p_wmmse[:5]}")
print(f"  Rates: {r_wmmse[:5]}")
print(f"  Sum-rate: {np.sum(r_wmmse):.4f} bps/Hz")

# Run FP
print("\n[FP]")
p_fp, stats_fp = fp(N, H, Pmax, noise_var, weights)
r_fp = compute_rates(H, p_fp, noise_var)
print(f"  Iterations: {stats_fp[1]}")
print(f"  Time: {stats_fp[0]:.4f}s")
print(f"  Power: {p_fp[:5]}")
print(f"  Rates: {r_fp[:5]}")
print(f"  Sum-rate: {np.sum(r_fp):.4f} bps/Hz")

print("\n" + "=" * 70)
print(f"WMMSE vs FP: {np.sum(r_wmmse):.4f} vs {np.sum(r_fp):.4f}")
if np.sum(r_wmmse) >= np.sum(r_fp) - 0.01:
    print("✓ WMMSE ≥ FP (correct)")
else:
    print("✗ WMMSE < FP (BUG!)")
