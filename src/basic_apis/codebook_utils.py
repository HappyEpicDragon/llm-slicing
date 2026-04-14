import numpy as np
from typing import List, Set, Tuple


def _gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient for non-negative allocations."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = x.size
    total = np.sum(x)
    if n == 0 or total <= 0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * x) / (n * total)) - ((n + 1.0) / n))


def build_dirichlet_inter_quota_codebook(
    num_slices: int = 5,
    num_prbs: int = 135,
    c_min: int = 10,
    num_concentration_levels: int = 10,
    num_mc_samples: int = 10_000,
    beta_min: float = 0.1,
    beta_max: float = 50.0,
    seed: int = 2025,
) -> List[np.ndarray]:
    """
    Build sorted inter-slice quota templates using Dirichlet order statistics.

    The generated templates satisfy:
    - each template is descending-sorted
    - each slice gets at least c_min PRBs
    - total equals num_prbs
    - duplicates after rounding are removed
    """
    if num_slices <= 0:
        raise ValueError("num_slices must be positive")
    if num_prbs <= 0:
        raise ValueError("num_prbs must be positive")
    if c_min * num_slices > num_prbs:
        raise ValueError("c_min * num_slices cannot exceed num_prbs")
    if num_concentration_levels <= 0 or num_mc_samples <= 0:
        raise ValueError("num_concentration_levels and num_mc_samples must be positive")

    rng = np.random.default_rng(seed)
    surplus = num_prbs - num_slices * c_min
    betas = np.logspace(np.log10(beta_min), np.log10(beta_max), num_concentration_levels)

    templates: Set[Tuple[int, ...]] = set()

    # Explicitly include the egalitarian template (Gini = 0).
    uniform = np.full(num_slices, num_prbs // num_slices, dtype=np.int32)
    uniform[: num_prbs - int(uniform.sum())] += 1
    templates.add(tuple(sorted(uniform.tolist(), reverse=True)))

    for beta in betas:
        samples = rng.dirichlet(np.full(num_slices, beta, dtype=np.float64), size=num_mc_samples)
        expected_sorted_weights = np.sort(samples, axis=1)[:, ::-1].mean(axis=0)

        alloc = c_min + np.rint(surplus * expected_sorted_weights).astype(np.int32)
        diff = int(num_prbs - int(alloc.sum()))

        if diff > 0:
            grow_order = np.argsort(-expected_sorted_weights)
            for idx in grow_order[:diff]:
                alloc[idx] += 1
        elif diff < 0:
            shrink_order = np.argsort(expected_sorted_weights)
            for idx in shrink_order[: -diff]:
                if alloc[idx] > c_min:
                    alloc[idx] -= 1
                    continue
                # Guard: in rare edge cases, pick next feasible dimension.
                for alt in shrink_order:
                    if alloc[alt] > c_min:
                        alloc[alt] -= 1
                        break

        alloc = np.sort(alloc)[::-1]
        templates.add(tuple(int(v) for v in alloc))

    sorted_templates = sorted(
        templates,
        key=lambda tpl: (_gini_coefficient(np.array(tpl, dtype=np.float64)), tpl),
    )
    return [np.array(tpl, dtype=np.int32) for tpl in sorted_templates]
