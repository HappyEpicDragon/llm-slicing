"""
XAI 公共工具模块

集中管理 XAI 三个子任务共享的：
- ACTION_DIMS: 动作空间维度默认值（优先动态解析）
- make_env: 按 scenario_id 创建测试环境
- make_env_for_scenario: 按 scenario_conf 对象创建环境
- load_models: 加载 PPO + DT 模型对
"""

import os
import numpy as np
import torch
from stable_baselines3 import PPO

from src.basic_apis.dt_utils.model_ha_dt import HierarchicalStateEncoder, DecisionTransformer
from src.basic_apis.dt_utils.action_dims import resolve_action_dims_from_env_cfg
from src.basic_apis.ppo.ppo_ha_weighted.hierarchical_slicing_env import HierarchicalSlicingEnv
from src.basic_apis.ppo.ppo_ha_weighted.agent_hierarchical import HierarchicalSmartPolicy

# Fallback only: Inter modes (11) + 5 × Intra schedulers (3)
ACTION_DIMS = [11, 3, 3, 3, 3, 3]


def resolve_project_path(pm, cfg_path):
    """
    将配置中的相对路径解析为项目根目录下的绝对路径。
    若 cfg_path 已是绝对路径，则原样返回。
    """
    if os.path.isabs(cfg_path):
        return cfg_path
    return os.path.join(pm.root_path, cfg_path)


def make_env(env_cfg, pm, scenario_id, seed=42):
    """按 scenario_id 创建单一测试环境（用于 attention / rtg_sweeping）"""
    env_config = env_cfg.env_settings.copy()
    env_config.mode = 'testing'
    scenario_mode = env_config.scenario_mode

    if 'testing' not in env_config[scenario_mode]:
        env_config[scenario_mode]['testing'] = env_config[scenario_mode]['evaluating'].copy()

    env_config[scenario_mode]['testing'].active_scenario_list = [scenario_id]
    return HierarchicalSlicingEnv(env_config, np.random.default_rng(seed), pm)


def make_env_for_scenario(base_env_cfg, scenario_conf, pm, seed=42):
    """按 scenario_conf 对象创建环境（用于 semantic_manifold）"""
    local_cfg = base_env_cfg.copy()
    mode = scenario_conf.mode
    scen_id = scenario_conf.scenario_id

    local_cfg.mode = mode
    scenario_mode = local_cfg.scenario_mode

    if mode not in local_cfg[scenario_mode]:
        target_cfg = local_cfg[scenario_mode]['evaluating']
    else:
        target_cfg = local_cfg[scenario_mode][mode]

    target_cfg.active_scenario_list = [scen_id]
    return HierarchicalSlicingEnv(local_cfg, np.random.default_rng(seed), pm)


def load_models(cfg_xai, env_cfg=None):
    """
    加载 PPO（教师）和 DT（学生）模型对。

    先尝试 embed_dim=256，失败时自动回退到 embed_dim=512。
    返回 (ppo_model, dt_model)，失败时返回 (None, None)。
    """
    print("Loading Models...")
    device = cfg_xai.device

    # 1. PPO
    try:
        ppo_model = PPO.load(cfg_xai.model_paths.ppo, device=device, custom_objects={
            "HierarchicalSmartPolicy": HierarchicalSmartPolicy,
            "learning_rate": 0.0,
            "clip_range": 0.1,
        })
    except Exception as e:
        print(f"PPO load failed: {e}")
        return None, None

    # 2. DT
    # 尽量从配置/环境动态解析动作维度，避免硬编码 M=11。
    if env_cfg is not None:
        action_dims = resolve_action_dims_from_env_cfg(env_cfg, intra_mode_count=3)
    elif hasattr(cfg_xai, "environment"):
        action_dims = resolve_action_dims_from_env_cfg(cfg_xai.environment, intra_mode_count=3)
    else:
        action_dims = ACTION_DIMS

    context_len = int(getattr(cfg_xai, "context_len", 20))

    def _build_dt(embed_dim, n_layer=12, n_head=16):
        enc = HierarchicalStateEncoder(embed_dim=embed_dim)
        return DecisionTransformer(
            state_encoder=enc,
            action_dims=action_dims,
            hidden_size=embed_dim,
            max_length=context_len,
            n_layer=n_layer,
            n_head=n_head,
            dropout=0.1,
            activation="relu",
        ).to(device)

    candidates = [
        (512, 12, 16),  # current training config
        (512, 3, 4),    # legacy shallow DT
        (256, 12, 16),
        (256, 3, 4),
    ]
    dt_model = None
    last_err = None
    for embed_dim, n_layer, n_head in candidates:
        try:
            cand = _build_dt(embed_dim, n_layer=n_layer, n_head=n_head)
            cand.load_state_dict(torch.load(cfg_xai.model_paths.dt, map_location=device))
            dt_model = cand
            print(f"✅ DT loaded with embed_dim={embed_dim}, layers={n_layer}, heads={n_head}, action_dims={action_dims}")
            break
        except Exception as e:
            last_err = e
            continue

    if dt_model is None:
        print(f"DT load failed after trying all architecture candidates: {last_err}")
        return None, None

    dt_model.eval()
    return ppo_model, dt_model
