import os
import numpy as np
import torch
from tqdm import tqdm
from stable_baselines3 import PPO

from src.basic_apis.ppo.ppo_ha_weighted.agent_hierarchical import HierarchicalSmartPolicy
from src.basic_apis.explainable_ai_utils.xai_utils import make_env_for_scenario, resolve_project_path


def extract_ppo_features(model, obs):
    """
    核心手术刀函数：从 PPO Policy 中提取 Latent Features (z)
    对应 agent_hierarchical.py -> HierarchicalAttentionExtractor -> fusion_layer 输出
    """
    with torch.no_grad():
        # 1. 将 Dict Observation 转为 Tensor
        # obs_tensor 是一个 Dict[str, Tensor]
        obs_tensor, _ = model.policy.obs_to_tensor(obs)

        # 2. 前向传播通过 Features Extractor
        # 输出应该是 [1, 256] 的向量
        features = model.policy.features_extractor(obs_tensor)

    return features.cpu().numpy().flatten()


def run_ppo_collection(cfg, path_manager):
    """
    语义流形对齐实验 - PPO 数据采集入口

    Args:
        cfg: Hydra 配置对象 (包含 model_paths, scenarios 等)
        path_manager: 用于处理路径的工具类
    """
    print(f"🚀 [PPO] Starting Data Collection")
    print(f"    Model Path: {cfg.model_paths.ppo}")

    # 1. 准备工作
    # 确保输出目录存在
    save_dir = resolve_project_path(path_manager, cfg.asset_dir)
    os.makedirs(save_dir, exist_ok=True)

    # 加载基础环境配置 (hierarchical_env.yaml)
    # 注意：这里假设 cfg.env_config_path 是相对于工作目录的路径
    base_env_conf = cfg.environment

    # 2. 加载 PPO 模型
    # custom_objects 必须包含策略类，否则 SB3 会报错
    try:
        model = PPO.load(
            cfg.model_paths.ppo,
            device=cfg.device,
            custom_objects={
                "HierarchicalSmartPolicy": HierarchicalSmartPolicy,
                "learning_rate": 0.0,
                "clip_range": 0.1
            }
        )
        print("✅ PPO Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading PPO Model: {e}")
        return

    # 准备数据容器
    collected_data = []

    # 3. 遍历定义的场景 (S2 和 S9)
    # cfg.target_scenarios 是一个 DictConfig
    for key, scen_conf in cfg.target_scenarios.items():
        print(f"👉 Processing {scen_conf.name} (Scenario {scen_conf.scenario_id})...")

        # 创建环境
        env = make_env_for_scenario(base_env_conf.env_settings, scen_conf, path_manager)
        obs, _ = env.reset()

        # 采集循环
        for _ in tqdm(range(cfg.samples_per_scenario), desc=f"Collecting {key}"):
            # A. 提取特征 (The Brain Scan) [256-dim]
            latent_vec = extract_ppo_features(model, obs)

            # B. 预测动作 (Deterministic) - 获取 Semantic Action
            action, _ = model.predict(obs, deterministic=True)

            # C. 记录数据
            # action[0] 是 Inter-slice Pattern ID (我们关心的语义动作)
            record = {
                "embedding": latent_vec,
                "scenario_label": scen_conf.label,  # 0 for S2, 1 for S9
                "semantic_action": int(action[0]),
                "model_type": "PPO",
            }
            collected_data.append(record)

            # D. 环境步进
            obs, _, done, _, _ = env.step(action)
            if done:
                obs, _ = env.reset()

        env.close()

    # 4. 保存数据
    save_path = os.path.join(save_dir, "ppo_data.npy")
    np.save(save_path, collected_data)
    print(f"💾 [PPO] Data collection complete. Saved {len(collected_data)} samples to:")
    print(f"   -> {save_path}")