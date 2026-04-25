"""
Physical layer for K-user interference channel.

Extracted from Nasir & Guo (Asilomar 2020) with TF removed.
Includes: Jakes fading, shadowing, path loss, SINR/rate, WMMSE, FP.
"""
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import special

# 30 dB SINR cap to prevent log from blowing up
_THRESHOLD_SINR = 10.0 ** (30.0 / 10.0)


# ---------------------------------------------------------------------------
# Fast-fading: Rayleigh / Jakes model
# ---------------------------------------------------------------------------


def get_random_rayleigh_variable(rayleigh_var: float, N: int, K: Optional[int] = None) -> np.ndarray:
    """Initial i.i.d. complex Rayleigh channel (K×N) or (N×N)."""
    shape = (K, N) if K is not None else (N, N)
    return np.sqrt(2.0 / np.pi) * rayleigh_var * (
        np.random.randn(*shape) + 1j * np.random.randn(*shape)
    )


def get_markov_rayleigh_variable(
    state: np.ndarray,
    correlation: float,
    rayleigh_var: float,
    N: int,
    K: Optional[int] = None,
) -> np.ndarray:
    """
    Advance Rayleigh channel by one step via Markov (Jakes) model.

    correlation = J0(2π·fd·T)  (Bessel function of the first kind)
    """
    shape = (K, N) if K is not None else (N, N)
    noise = np.sqrt(2.0 / np.pi) * rayleigh_var * (
        np.random.randn(*shape) + 1j * np.random.randn(*shape)
    )
    return correlation * state + np.sqrt(1.0 - correlation ** 2) * noise


# ---------------------------------------------------------------------------
# Topology: hexagonal grid + UE placement
# ---------------------------------------------------------------------------


def _inside_hexagon(x: float, y: float, xhex: np.ndarray, yhex: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test for a hexagonal cell."""
    n = len(xhex) - 1
    inside = False
    p1x, p1y = xhex[0], yhex[0]
    for i in range(n + 1):
        p2x, p2y = xhex[i % n], yhex[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y):
            if x <= max(p1x, p2x):
                if p1y != p2y:
                    xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xints:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _build_hexagonal_grid(K: int, R: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Place K BSs on a hexagonal grid.

    Returns:
        TX_loc  (2, K)
        TX_xhex (7, K) - hexagon vertex x-coords
        TX_yhex (7, K) - hexagon vertex y-coords
    """
    x_hex = R * np.array([0, -np.sqrt(3) / 2, -np.sqrt(3) / 2, 0, np.sqrt(3) / 2, np.sqrt(3) / 2, 0])
    y_hex = R * np.array([-1, -0.5, 0.5, 1, 0.5, -0.5, -1])

    TX_loc = np.zeros((2, K))
    TX_xhex = np.zeros((7, K))
    TX_yhex = np.zeros((7, K))

    TX_xhex[:, 0] = x_hex
    TX_yhex[:, 0] = y_hex
    gen = 1
    i = 0
    while gen < K:
        for j in range(6):
            cx = TX_loc[0, i] + np.sqrt(3) * R * np.cos(j * np.pi / 3)
            cy = TX_loc[1, i] + np.sqrt(3) * R * np.sin(j * np.pi / 3)
            # deduplicate
            dup = any(
                abs(cx - TX_loc[0, k]) < R * 1e-2 and abs(cy - TX_loc[1, k]) < R * 1e-2
                for k in range(gen)
            )
            if not dup:
                TX_loc[0, gen] = cx
                TX_loc[1, gen] = cy
                TX_xhex[:, gen] = cx + x_hex
                TX_yhex[:, gen] = cy + y_hex
                gen += 1
            if gen >= K:
                break
        i += 1
    return TX_loc, TX_xhex, TX_yhex


def _find_neighbors(K: int, R: float, TX_loc: np.ndarray) -> List[List[int]]:
    neighbors: List[List[int]] = []
    for i in range(K):
        row: List[int] = []
        for j in range(6):
            cx = TX_loc[0, i] + np.sqrt(3) * R * np.cos(j * np.pi / 3)
            cy = TX_loc[1, i] + np.sqrt(3) * R * np.sin(j * np.pi / 3)
            for m in range(K):
                if m != i and abs(cx - TX_loc[0, m]) < R * 1e-2 and abs(cy - TX_loc[1, m]) < R * 1e-2:
                    row.append(m)
        neighbors.append(row)
    return neighbors


def _place_ues(
    N: int,
    K: int,
    R: float,
    min_dist: float,
    TX_loc: np.ndarray,
    TX_xhex: np.ndarray,
    TX_yhex: np.ndarray,
    equal_number_for_BS: bool,
    v_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Place N UEs inside hexagonal cells. No mobility (v_max handled elsewhere).

    Returns:
        RX_loc_0   (2, N) - initial UE positions
        cell_map_0 (N,)   - initial cell assignments (int)
    """
    RX_loc_0 = np.zeros((2, N))
    cell_map_0 = np.zeros(N, dtype=int)

    for i in range(N):
        if equal_number_for_BS:
            assert N % K == 0
            cell_map_0[i] = int(i / (N // K))
        else:
            cell_map_0[i] = np.random.randint(K)

        c = cell_map_0[i]
        xmin, xmax = TX_xhex[:, c].min(), TX_xhex[:, c].max()
        ymin, ymax = TX_yhex[:, c].min(), TX_yhex[:, c].max()

        while True:
            px = np.random.uniform(xmin, xmax)
            py = np.random.uniform(ymin, ymax)
            d = np.sqrt((px - TX_loc[0, c]) ** 2 + (py - TX_loc[1, c]) ** 2)
            if _inside_hexagon(px, py, TX_xhex[:, c], TX_yhex[:, c]) and min_dist < d < R:
                RX_loc_0[0, i] = px
                RX_loc_0[1, i] = py
                break

    return RX_loc_0, cell_map_0


# ---------------------------------------------------------------------------
# Distance & path-loss
# ---------------------------------------------------------------------------


def _compute_distances(
    N: int,
    TX_loc: np.ndarray,
    RX_loc: np.ndarray,
    cell_mapping: np.ndarray,
    total_samples: int,
) -> np.ndarray:
    """
    Compute N×N distance matrix for each sample.

    distance_vector[i, j, t] = distance from BS_i to UE_j at time t.
    TX of UE j at time t is cell_mapping[j, t].
    """
    distance_vector = np.zeros((N, N, total_samples))
    for t in range(total_samples):
        tmp_TX = TX_loc[:, cell_mapping[:, t]]  # (2, N)
        for j in range(N):
            dx = tmp_TX[0, j] - RX_loc[0, :, t]
            dy = tmp_TX[1, j] - RX_loc[1, :, t]
            distance_vector[:, j, t] = np.sqrt(dx ** 2 + dy ** 2)
    return distance_vector


# ---------------------------------------------------------------------------
# Channel generation: gains + Jakes fading
# ---------------------------------------------------------------------------


def generate_channel_sequence(
    N: int,
    K: int,
    R_defined: float,
    min_dist: float,
    total_samples: int,
    shadowing_dev: float = 10.0,
    dcor: float = 10.0,
    equal_number_for_BS: bool = True,
    fd: float = 10.0,
    T: float = 0.02,
    rayleigh_var: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Generate a time series of channel matrices H_t (N×N) for total_samples steps.

    Uses hexagonal topology, spatially-correlated shadowing, and Jakes fading.
    UE positions are static (v_max=0, pure Jakes fading with fd Hz).

    Args:
        N: Number of UE-BS pairs (users)
        K: Number of BSs
        R_defined: Cell radius in metres
        min_dist: Minimum UE-BS distance in metres
        total_samples: Number of time steps
        shadowing_dev: Shadowing standard deviation in dB
        dcor: Shadowing decorrelation distance in metres
        equal_number_for_BS: If True, N/K UEs per BS
        fd: Doppler frequency in Hz
        T: Time step in seconds
        rayleigh_var: Rayleigh fading variance
        seed: RNG seed

    Returns:
        H_all: (total_samples, N, N) channel gain matrix |h|
        meta:  dict with TX_loc, RX_loc, TX_xhex, TX_yhex, neighbors, cell_mapping
    """
    if seed is not None:
        np.random.seed(seed)

    R = (2.0 / np.sqrt(3)) * R_defined

    # --- Build topology ---
    TX_loc, TX_xhex, TX_yhex = _build_hexagonal_grid(K, R)
    neighbors = _find_neighbors(K, R, TX_loc)

    # Static UE placement (no physical mobility)
    RX_loc_0, cell_map_0 = _place_ues(
        N, K, R, min_dist, TX_loc, TX_xhex, TX_yhex, equal_number_for_BS, v_max=0.0
    )

    # Tile positions across time (static UEs)
    RX_loc = np.tile(RX_loc_0[:, :, np.newaxis], (1, 1, total_samples))  # (2, N, T)
    cell_mapping = np.tile(cell_map_0[:, np.newaxis], (1, total_samples))  # (N, T)

    # Zero displacement for static UEs (needed for shadowing decorrelation)
    RX_displacement = np.zeros((4, N, total_samples))

    # --- Path-loss (3GPP UMa) ---
    distance_vector = _compute_distances(N, TX_loc, RX_loc, cell_mapping, total_samples)
    g_dB = -(128.1 + 37.6 * np.log10(1e-3 * np.maximum(distance_vector, 1.0)))

    # --- Spatially correlated shadowing ---
    shadowing = np.zeros((K, N, total_samples))
    shadowing[:, :, 0] = np.random.randn(K, N)
    for t in range(1, total_samples):
        disp_norm = np.sqrt(
            RX_displacement[0, :, t] ** 2 + RX_displacement[1, :, t] ** 2
        )  # (N,) — all zeros for static UEs → correlation=1
        corr = np.exp(-disp_norm / dcor)  # (N,)
        for bs in range(K):
            shadowing[bs, :, t] = (
                corr * shadowing[bs, :, t - 1]
                + np.sqrt(1.0 - corr ** 2) * np.random.randn(N)
            )

    # Combine path-loss + shadowing into large-scale gains (N, N, T)
    tmp_g_dB = np.zeros((N, N, total_samples))
    for t in range(total_samples):
        for j in range(N):
            tmp_g_dB[j, :, t] = (
                g_dB[j, :, t] + shadowing_dev * shadowing[cell_mapping[:, t], j, t]
            )
    gains = np.power(10.0, tmp_g_dB / 10.0)  # (N, N, T)

    # --- Jakes fading ---
    correlation = special.j0(2.0 * np.pi * fd * T)
    channel_b = get_random_rayleigh_variable(rayleigh_var, N, K)  # (K, N)

    # Map BS-indexed fading to N×N using cell_mapping
    def _bs_to_ue_channel(channel_b: np.ndarray, cm: np.ndarray) -> np.ndarray:
        # channel_b: (K, N) → channel: (N, N) indexed by UE's serving BS
        ch = np.zeros((N, N), dtype=complex)
        for j in range(N):
            ch[j, :] = channel_b[cm[:, 0], j]
        return ch

    channel = _bs_to_ue_channel(channel_b, cell_mapping)
    H_all = np.zeros((total_samples, N, N))
    H_all[0] = np.sqrt(gains[:, :, 0]) * np.abs(channel)

    for t in range(1, total_samples):
        channel_b = get_markov_rayleigh_variable(channel_b, correlation, rayleigh_var, N, K)
        ch = np.zeros((N, N), dtype=complex)
        for j in range(N):
            ch[j, :] = channel_b[cell_mapping[:, t], j]
        H_all[t] = np.sqrt(gains[:, :, t]) * np.abs(ch)

    meta = {
        "TX_loc": TX_loc,
        "RX_loc": RX_loc,
        "TX_xhex": TX_xhex,
        "TX_yhex": TX_yhex,
        "neighbors": neighbors,
        "cell_mapping": cell_mapping,
    }
    return H_all, meta


# ---------------------------------------------------------------------------
# SINR / rate utilities
# ---------------------------------------------------------------------------


def compute_rates(H: np.ndarray, p: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Compute per-user rates (bits/s/Hz).

    Args:
        H:         (N, N) channel gain matrix (|h|, not |h|^2)
        p:         (N,) power allocation in Watts
        noise_var: Noise power in Watts

    Returns:
        rates: (N,) per-user rates, SINR capped at THRESHOLD_SINR
    """
    H2 = H ** 2
    rates = np.zeros(H.shape[0])
    N = H.shape[0]
    for k in range(N):
        signal = H2[k, k] * p[k]
        total = H2[k, :] @ p + noise_var
        interference = total - signal
        sinr = min(signal / (interference + 1e-15), _THRESHOLD_SINR)
        rates[k] = np.log2(1.0 + sinr)
    return rates


def compute_weighted_sumrate(H: np.ndarray, p: np.ndarray, noise_var: float, weights: np.ndarray) -> float:
    """Weighted sum-rate (scalar)."""
    return float(np.sum(weights * compute_rates(H, p, noise_var)))


def compute_sinr_db(H: np.ndarray, p: np.ndarray, noise_var: float) -> np.ndarray:
    """Per-user SINR in dB."""
    H2 = H ** 2
    N = H.shape[0]
    sinr = np.zeros(N)
    for k in range(N):
        signal = H2[k, k] * p[k]
        interference = H2[k, :] @ p + noise_var - signal
        sinr[k] = 10.0 * np.log10(signal / (interference + 1e-15))
    return sinr


# ---------------------------------------------------------------------------
# WMMSE algorithm (Shi et al. 2011, weighted variant)
# ---------------------------------------------------------------------------


def wmmse(
    N: int,
    H: np.ndarray,
    Pmax: float,
    noise_var: float,
    weights: np.ndarray,
    max_iter: int = 100,
    tol: float = 0.01,
) -> Tuple[np.ndarray, List]:
    """
    Weighted WMMSE power allocation.

    Args:
        N:         Number of users
        H:         (N, N) channel gain matrix (|h|)
        Pmax:      Maximum power per user (Watts)
        noise_var: Noise power
        weights:   (N,) user weights
        max_iter:  Maximum iterations
        tol:       Convergence tolerance

    Returns:
        p_opt:      (N,) optimal power allocation
        stats:      [solve_time, iterations]
    """
    t0 = time.time()
    b = np.sqrt(Pmax) * np.ones(N)
    f = np.zeros(N)
    w = np.zeros(N)

    for i in range(N):
        f[i] = H[i, i] * b[i] / (np.square(H[i, :]) @ np.square(b) + noise_var)
        w[i] = 1.0 / (1.0 - f[i] * b[i] * H[i, i])

    vnew = np.sum(np.log2(w))

    itr = 0
    for itr in range(max_iter):
        vold = vnew
        for i in range(N):
            denom = np.sum(weights * w * np.square(f) * np.square(H[:, i]))
            btmp = weights[i] * w[i] * f[i] * H[i, i] / (denom + 1e-15)
            b[i] = np.clip(btmp, 0.0, np.sqrt(Pmax))

        vnew = 0.0
        for i in range(N):
            f[i] = H[i, i] * b[i] / (np.square(H[i, :]) @ np.square(b) + noise_var)
            w[i] = 1.0 / max(1.0 - f[i] * b[i] * H[i, i], 1e-15)
            vnew += np.log2(w[i])

        if vnew - vold <= tol:
            break

    return np.square(b), [time.time() - t0, itr]


# ---------------------------------------------------------------------------
# FP algorithm (Shen et al., weighted variant)
# ---------------------------------------------------------------------------


def fp(
    N: int,
    H: np.ndarray,
    Pmax: float,
    noise_var: float,
    weights: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, List]:
    """
    Weighted Fractional Programming power allocation.

    Args:
        N:         Number of users
        H:         (N, N) channel gain matrix (|h|)
        Pmax:      Maximum power per user (Watts)
        noise_var: Noise power
        weights:   (N,) user weights
        max_iter:  Maximum iterations
        tol:       Convergence tolerance

    Returns:
        p_opt:  (N,) optimal power allocation
        stats:  [solve_time, iterations]
    """
    t0 = time.time()
    H2 = H ** 2
    p = Pmax * np.ones(N)
    gamma = np.zeros(N)
    y = np.zeros(N)

    for i in range(N):
        sig = H2[i, i] * p[i]
        tot = H2[i, :] @ p + noise_var
        gamma[i] = sig / (tot - sig)

    f_new = 0.0
    itr = 0
    for itr in range(max_iter):
        f_old = f_new
        for i in range(N):
            sig = H2[i, i] * p[i]
            tot = H2[i, :] @ p + noise_var
            y[i] = np.sqrt(weights[i] * (1 + gamma[i]) * sig) / tot
            gamma[i] = sig / (tot - sig)

        for i in range(N):
            denom = (np.square(y) @ H2[:, i]) ** 2
            p[i] = min(Pmax, (y[i] ** 2) * weights[i] * (1 + gamma[i]) * H2[i, i] / (denom + 1e-15))

        f_new = 0.0
        for i in range(N):
            f_new += (
                2 * y[i] * np.sqrt(weights[i] * (1 + gamma[i]) * H2[i, i] * p[i])
                - (y[i] ** 2) * (H2[i, :] @ p + noise_var)
            )

        if f_new - f_old <= tol:
            break

    return p, [time.time() - t0, itr]
