from src.basic_apis.codebook_utils import build_dirichlet_inter_quota_codebook


def resolve_action_dims_from_env_cfg(env_cfg, intra_mode_count=3):
    """
    Resolve MultiDiscrete action dimensions from environment config.

    Returns:
        action_dims: [num_inter_modes] + [intra_mode_count] * num_slices
    """
    try:
        env_settings = env_cfg.env_settings if hasattr(env_cfg, "env_settings") else env_cfg
        num_slices = int(env_settings.components.slices.max_number_slices)
        num_prbs = int(env_settings.components.basestations.num_available_rbs[0])
        inter_modes = len(
            build_dirichlet_inter_quota_codebook(
                num_slices=num_slices,
                num_prbs=num_prbs,
                c_min=10,
                num_concentration_levels=10,
                num_mc_samples=10_000,
                beta_min=0.1,
                beta_max=50.0,
                seed=2025,
            )
        )
        return [inter_modes] + [int(intra_mode_count)] * num_slices
    except Exception:
        # Conservative fallback for legacy configs.
        return [11] + [int(intra_mode_count)] * 5
