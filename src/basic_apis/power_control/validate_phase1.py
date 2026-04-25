"""
Phase 1 validation: verify extracted physical layer matches Nasir 2020.

Reproduces WMMSE/FP benchmarks on K=10, N=20, shadow=10dB, fd=10Hz.
Compares against expected sum-rate from original paper (±5% tolerance).
"""
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.basic_apis.power_control.benchmarks import run_all_benchmarks
from src.basic_apis.power_control.deployment import generate_and_save, load_channel


def main():
    print("=" * 70)
    print("Phase 1 Validation: Physical Layer Extraction")
    print("=" * 70)

    # Config matching train_K10_N20_shadow10_episode10-5000_travel0_fd10.json
    config = {
        "N": 20,
        "K": 10,
        "R_defined": 400.0,
        "min_dist": 35.0,
        "total_samples": 1000,  # Reduced for quick validation
        "shadowing_dev": 10.0,
        "dcor": 10.0,
        "equal_number_for_BS": True,
        "fd": 10.0,
        "T": 0.02,
        "rayleigh_var": 1.0,
        "seed": 42,
    }

    # Physical layer params (matching Nasir 2020 get_benchmarks.py)
    Pmax_dB = 38.0 - 30  # 8 dBW
    Pmax = np.power(10.0, Pmax_dB / 10)  # ~6.31 W
    n0_dB = -114.0 - 30  # -144 dBm
    noise_var = np.power(10.0, n0_dB / 10)  # ~3.98e-15 W
    weights = np.ones(config["N"])  # Uniform weights

    print("\n[1/3] Generating channel sequence...")
    print(f"  N={config['N']}, K={config['K']}, T={config['total_samples']}")
    print(f"  fd={config['fd']} Hz, shadowing_dev={config['shadowing_dev']} dB")

    save_path = Path("/tmp/power_control_validation.npz")
    generate_and_save(save_path, **config)
    print(f"  ✓ Saved to {save_path}")

    print("\n[2/3] Loading channel data...")
    H_all, cell_mapping = load_channel(save_path)
    print(f"  ✓ Loaded H_all: {H_all.shape}")

    print("\n[3/3] Running benchmarks...")
    results = run_all_benchmarks(H_all, Pmax, noise_var, weights)

    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    for algo, metrics in results.items():
        print(f"\n{algo}:")
        for k, v in metrics.items():
            print(f"  {k:20s}: {v:.4f}")

    # Expected values from Nasir 2020 paper (approximate, K=10 scenario)
    # Paper reports ~35-40 bps/Hz sum-rate for WMMSE at SNR=20dB
    # Our noise_var=-90dBm, Pmax=1W → SNR ≈ 20dB
    expected_wmmse = 38.0  # bps/Hz (approximate from paper Fig. 3)
    tolerance = 0.05  # ±5%

    wmmse_sumrate = results["WMMSE"]["avg_sumrate"]
    error = abs(wmmse_sumrate - expected_wmmse) / expected_wmmse

    print("\n" + "=" * 70)
    print("Validation:")
    print("=" * 70)
    print(f"Expected WMMSE sum-rate: {expected_wmmse:.2f} bps/Hz (Nasir 2020)")
    print(f"Achieved WMMSE sum-rate: {wmmse_sumrate:.2f} bps/Hz")
    print(f"Relative error:          {error*100:.2f}%")

    if error <= tolerance:
        print(f"\n✓ PASS: Error within {tolerance*100:.0f}% tolerance")
        return 0
    else:
        print(f"\n✗ FAIL: Error exceeds {tolerance*100:.0f}% tolerance")
        print("\nNote: If this is first run, expected values may need calibration.")
        print("Check that WMMSE > FP > Random > MaxPower (monotonicity).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
