"""
Classical power control benchmarks: WMMSE, FP, Random, Max Power.

Extracted from get_benchmarks.py in Nasir & Guo (Asilomar 2020).
"""
from typing import Dict, List, Tuple

import numpy as np

from .channel import compute_rates, fp, wmmse


def run_wmmse_benchmark(
    H_all: np.ndarray,
    Pmax: float,
    noise_var: float,
    weights: np.ndarray,
) -> Dict[str, float]:
    """
    Run WMMSE on channel sequence.

    Args:
        H_all:     (T, N, N) channel gain matrices
        Pmax:      Max power per user
        noise_var: Noise power
        weights:   (N,) user weights

    Returns:
        metrics: avg_sumrate, avg_time, avg_iters
    """
    T, N, _ = H_all.shape
    sumrates = []
    times = []
    iters = []

    for t in range(T):
        p_opt, stats = wmmse(N, H_all[t], Pmax, noise_var, weights)
        rates = compute_rates(H_all[t], p_opt, noise_var)
        sumrates.append(np.sum(rates))
        times.append(stats[0])
        iters.append(stats[1])

    return {
        "avg_sumrate": float(np.mean(sumrates)),
        "avg_time": float(np.mean(times)),
        "avg_iters": float(np.mean(iters)),
    }


def run_fp_benchmark(
    H_all: np.ndarray,
    Pmax: float,
    noise_var: float,
    weights: np.ndarray,
) -> Dict[str, float]:
    """Run FP on channel sequence."""
    T, N, _ = H_all.shape
    sumrates = []
    times = []
    iters = []

    for t in range(T):
        p_opt, stats = fp(N, H_all[t], Pmax, noise_var, weights)
        rates = compute_rates(H_all[t], p_opt, noise_var)
        sumrates.append(np.sum(rates))
        times.append(stats[0])
        iters.append(stats[1])

    return {
        "avg_sumrate": float(np.mean(sumrates)),
        "avg_time": float(np.mean(times)),
        "avg_iters": float(np.mean(iters)),
    }


def run_random_benchmark(
    H_all: np.ndarray,
    Pmax: float,
    noise_var: float,
    seed: int = 42,
) -> Dict[str, float]:
    """Random power allocation baseline."""
    np.random.seed(seed)
    T, N, _ = H_all.shape
    sumrates = []

    for t in range(T):
        p = Pmax * np.random.rand(N)
        rates = compute_rates(H_all[t], p, noise_var)
        sumrates.append(np.sum(rates))

    return {"avg_sumrate": float(np.mean(sumrates))}


def run_maxpower_benchmark(
    H_all: np.ndarray,
    Pmax: float,
    noise_var: float,
) -> Dict[str, float]:
    """Max power baseline (all users transmit at Pmax)."""
    T, N, _ = H_all.shape
    sumrates = []
    p = Pmax * np.ones(N)

    for t in range(T):
        rates = compute_rates(H_all[t], p, noise_var)
        sumrates.append(np.sum(rates))

    return {"avg_sumrate": float(np.mean(sumrates))}


def run_all_benchmarks(
    H_all: np.ndarray,
    Pmax: float,
    noise_var: float,
    weights: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Run all four benchmarks and return results."""
    return {
        "WMMSE": run_wmmse_benchmark(H_all, Pmax, noise_var, weights),
        "FP": run_fp_benchmark(H_all, Pmax, noise_var, weights),
        "Random": run_random_benchmark(H_all, Pmax, noise_var),
        "MaxPower": run_maxpower_benchmark(H_all, Pmax, noise_var),
    }
