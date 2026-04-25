"""
Channel dataset generation for K-user interference channel.

Wraps channel.py to generate and persist channel time series,
replacing the hardcoded-path logic in random_deployment.py.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .channel import generate_channel_sequence


def generate_and_save(
    save_path: Path,
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
) -> Path:
    """
    Generate channel sequence and save to .npz.

    Returns path to saved file.
    """
    H_all, meta = generate_channel_sequence(
        N=N,
        K=K,
        R_defined=R_defined,
        min_dist=min_dist,
        total_samples=total_samples,
        shadowing_dev=shadowing_dev,
        dcor=dcor,
        equal_number_for_BS=equal_number_for_BS,
        fd=fd,
        T=T,
        rayleigh_var=rayleigh_var,
        seed=seed,
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, H_all=H_all, cell_mapping=meta["cell_mapping"])
    return save_path


def load_channel(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load channel sequence from .npz.

    Returns:
        H_all:        (T, N, N) channel gain matrices
        cell_mapping: (N, T) cell assignments
    """
    data = np.load(path, allow_pickle=True)
    return data["H_all"], data["cell_mapping"]
