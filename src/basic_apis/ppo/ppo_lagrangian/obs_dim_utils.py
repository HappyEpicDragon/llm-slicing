"""CostNetwork 输入维：与 HierarchicalSlicingEnvV2 / ppo-baseline 观测展平一致。"""
from __future__ import annotations

from typing import Any, Union

from omegaconf import DictConfig, OmegaConf

from src.basic_apis.dt_v2.env_v2 import INTER_DIM, INTRA_DIM
from src.basic_apis.network_slicing_business.network_slicing_business_executor import (
    ComponentConfig,
)

# ppo-baseline (IBSched) 扁平观测维度：5*10 + 5*19 = 145
LAGRANGIAN_BASELINE_OBS_DIM = 145


def lagrangian_flat_obs_dim_v2(env_settings: Union[dict, DictConfig, Any]) -> int:
    """CostNetwork 输入维（V2 env）：num_slices * INTER_DIM + max_users * INTRA_DIM + 2."""
    if isinstance(env_settings, dict):
        comp = env_settings["components"]
    else:
        comp = env_settings.components
    if isinstance(comp, dict):
        comp_d = comp
    else:
        comp_d = OmegaConf.to_container(comp, resolve=True)
    cc = ComponentConfig(comp_d)
    ns = cc.slice_config.max_number_slices
    nu = cc.ue_config.max_number_ues
    return int(ns * INTER_DIM + nu * INTRA_DIM + 2)
