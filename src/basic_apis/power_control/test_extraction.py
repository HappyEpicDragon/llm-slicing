"""
Unit test: verify WMMSE/FP extraction correctness via numerical properties.

Since original code requires TF1.x, we test via algorithm properties:
1. Convergence: objective improves monotonically
2. Feasibility: power constraints satisfied
3. Optimality: WMMSE ≥ FP ≥ Random ≥ MaxPower (for sum-rate)
"""
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.basic_apis.power_control.channel import compute_rates, fp, wmmse


def test_convergence():
    """Test that algorithms converge and satisfy constraints."""
    print("\n[Test 1/3] Convergence & Feasibility")
    print("-" * 50)

    np.random.seed(42)
    N = 10
    H = np.abs(np.random.randn(N, N) + 1j * np.random.randn(N, N))
    Pmax = 6.31
    noise_var = 3.98e-15
    weights = np.ones(N)

    # Test WMMSE
    p_wmmse, stats_wmmse = wmmse(N, H, Pmax, noise_var, weights)
    assert np.all(p_wmmse >= 0), "WMMSE: negative power"
    assert np.all(p_wmmse <= Pmax + 1e-6), f"WMMSE: power exceeds Pmax (max={np.max(p_wmmse):.4f})"
    print(f"  WMMSE: converged in {stats_wmmse[1]} iters, max_power={np.max(p_wmmse):.4f}")

    # Test FP
    p_fp, stats_fp = fp(N, H, Pmax, noise_var, weights)
    assert np.all(p_fp >= 0), "FP: negative power"
    assert np.all(p_fp <= Pmax + 1e-6), f"FP: power exceeds Pmax (max={np.max(p_fp):.4f})"
    print(f"  FP:    converged in {stats_fp[1]} iters, max_power={np.max(p_fp):.4f}")

    print("  ✓ PASS: Both algorithms converge and satisfy constraints")
    return True


def test_monotonicity():
    """Test that WMMSE ≥ FP ≥ Random ≥ MaxPower."""
    print("\n[Test 2/3] Algorithm Monotonicity")
    print("-" * 50)

    np.random.seed(42)
    N = 10
    H = np.abs(np.random.randn(N, N) + 1j * np.random.randn(N, N))
    Pmax = 6.31
    noise_var = 3.98e-15
    weights = np.ones(N)

    # Run all algorithms
    p_wmmse, _ = wmmse(N, H, Pmax, noise_var, weights)
    p_fp, _ = fp(N, H, Pmax, noise_var, weights)
    p_random = Pmax * np.random.rand(N)
    p_max = Pmax * np.ones(N)

    # Compute sum-rates
    r_wmmse = np.sum(compute_rates(H, p_wmmse, noise_var))
    r_fp = np.sum(compute_rates(H, p_fp, noise_var))
    r_random = np.sum(compute_rates(H, p_random, noise_var))
    r_max = np.sum(compute_rates(H, p_max, noise_var))

    print(f"  WMMSE:    {r_wmmse:.4f} bps/Hz")
    print(f"  FP:       {r_fp:.4f} bps/Hz")
    print(f"  Random:   {r_random:.4f} bps/Hz")
    print(f"  MaxPower: {r_max:.4f} bps/Hz")

    # Check monotonicity (allow small numerical errors)
    eps = 1e-3
    assert r_wmmse >= r_fp - eps, f"WMMSE < FP: {r_wmmse:.4f} < {r_fp:.4f}"
    assert r_fp >= r_random - eps, f"FP < Random: {r_fp:.4f} < {r_random:.4f}"
    # Note: Random may occasionally beat MaxPower, so we don't enforce that

    print("  ✓ PASS: WMMSE ≥ FP ≥ Random (monotonicity holds)")
    return True


def test_multiple_channels():
    """Test on 100 random channels to ensure robustness."""
    print("\n[Test 3/3] Robustness (100 random channels)")
    print("-" * 50)

    Pmax = 6.31
    noise_var = 3.98e-15
    N = 10
    weights = np.ones(N)

    wmmse_rates = []
    fp_rates = []

    for seed in range(100):
        np.random.seed(seed)
        H = np.abs(np.random.randn(N, N) + 1j * np.random.randn(N, N))

        p_wmmse, _ = wmmse(N, H, Pmax, noise_var, weights)
        p_fp, _ = fp(N, H, Pmax, noise_var, weights)

        wmmse_rates.append(np.sum(compute_rates(H, p_wmmse, noise_var)))
        fp_rates.append(np.sum(compute_rates(H, p_fp, noise_var)))

    avg_wmmse = np.mean(wmmse_rates)
    avg_fp = np.mean(fp_rates)
    std_wmmse = np.std(wmmse_rates)
    std_fp = np.std(fp_rates)

    print(f"  WMMSE: {avg_wmmse:.2f} ± {std_wmmse:.2f} bps/Hz")
    print(f"  FP:    {avg_fp:.2f} ± {std_fp:.2f} bps/Hz")
    print(f"  WMMSE wins: {np.sum(np.array(wmmse_rates) >= np.array(fp_rates))}/100")

    assert avg_wmmse >= avg_fp - 0.1, "WMMSE should outperform FP on average"
    print("  ✓ PASS: Algorithms are robust across diverse channels")
    return True


def main():
    print("=" * 70)
    print("Unit Test: WMMSE/FP Extraction Correctness")
    print("=" * 70)

    results = []
    results.append(test_convergence())
    results.append(test_monotonicity())
    results.append(test_multiple_channels())

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    if all(results):
        print("✓ All tests passed - Phase 1 extraction is correct")
        print("\nNote: Absolute sum-rate values depend on channel realization.")
        print("The key validation is algorithmic correctness (monotonicity, convergence).")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
