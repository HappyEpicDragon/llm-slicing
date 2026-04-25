import os
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from omegaconf import OmegaConf

# Local Imports (保持与你现有项目结构一致)
from src.basic_apis.ppo.ppo_ha.hierarchical_slicing_env import HierarchicalSlicingEnv
from src.basic_apis.ppo.ppo_ha.agent_hierarchical import HierarchicalSmartPolicy
from src.basic_apis.network_slicing_business.path_context import PathContext


def make_env(cfg, path_context, rank=0, seed=0):
    def _init():
        env_config = cfg.env_settings if hasattr(cfg, 'env_settings') else cfg
        if 'env_settings' in cfg:
            env_config = cfg.env_settings
        env = HierarchicalSlicingEnv(env_config, np.random.default_rng(seed + rank), path_context)
        return env

    return _init


def diagnose_agent(cfg, path_context, model_path, output_csv="diagnosis_report.csv"):
    # 1. 配置环境为 Testing 模式，覆盖所有场景
    scenario_mode = cfg.env_settings.scenario_mode
    mode = 'testing'
    cfg.env_settings.mode = mode

    # 强制覆盖配置以确保测试所有场景
    test_cfg = cfg.env_settings[scenario_mode][mode]
    print(f"🔬 Starting Diagnosis on {test_cfg.max_scenario_episodes - test_cfg.init_scenario_episode} Episodes...")

    env = DummyVecEnv([make_env(cfg, path_context, rank=0, seed=42)])

    # 2. 加载模型
    print(f"📥 Loading Model: {model_path}")
    try:
        model = PPO.load(model_path, env=env, device='cpu',
                         custom_objects={"learning_rate": 0.0, "clip_range": 0.1})
    except:
        model = PPO.load(model_path, env=env, device='cpu',
                         custom_objects={"HierarchicalSmartPolicy": HierarchicalSmartPolicy})

    # 3. 数据容器
    records = []

    obs = env.reset()

    # 获取总测试 Episode 数
    total_episodes = test_cfg.max_scenario_episodes - test_cfg.init_scenario_episode

    # 临时变量用于累积单个 Episode 的数据
    ep_data = {
        "reward_sum": 0,
        "csi_sum": 0,
        "traffic_sum": 0,
        "steps": 0,
        "hp_viol": 0,
        "nhp_viol": 0
    }

    pbar = tqdm(total=total_episodes, unit="ep")

    try:
        while len(records) < total_episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            info = infos[0]

            # === 核心：从 Observation 中提取环境特征 ===
            # 根据 hierarchical_slicing_env.py:
            # intra_feat[:, 3] 是 csi_norm (信道质量)
            # inter_feat[:, 2] 是 traffic_load (归一化流量)

            # obs 是一个 DictWrapper (DummyVecEnv)，需要取 [0] 或对应的 key
            # SB3 的 VecEnv 返回的 obs 是堆叠的，这里 batch=1
            current_intra = obs['intra_feat'][0]  # Shape: (25, 5)
            current_inter = obs['inter_feat'][0]  # Shape: (5, 4)

            # 计算当前 Step 的平均信道质量 (越高越好)
            step_avg_csi = np.mean(current_intra[:, 3])

            # 计算当前 Step 的平均流量负载 (越高越难)
            step_avg_traffic = np.mean(current_inter[:, 2])

            # 累积数据
            ep_data["reward_sum"] += rewards[0]
            ep_data["csi_sum"] += step_avg_csi
            ep_data["traffic_sum"] += step_avg_traffic
            ep_data["steps"] += 1

            # 累积违约 (从 Info 中获取)
            # 需要解析 info 里的 drift/violation
            # 假设 info 中包含 'violation/slice_X_thr' 等
            for k, v in info.items():
                if k.startswith("violation/slice_"):
                    # 解析 slice index
                    parts = k.split('_')
                    s_idx = int(parts[1])
                    prio = info.get(f"meta/slice_{s_idx}_priority", 0)
                    if v > 0:  # 发生违约
                        if prio > 0:
                            ep_data["hp_viol"] += 1
                        else:
                            ep_data["nhp_viol"] += 1

            if dones[0]:
                # Episode 结束，计算统计量
                avg_csi = ep_data["csi_sum"] / ep_data["steps"]
                avg_traffic = ep_data["traffic_sum"] / ep_data["steps"]

                # 记录这一局的完整画像
                records.append({
                    "episode_idx": len(records),
                    "reward": ep_data["reward_sum"],
                    "hp_violations": ep_data["hp_viol"],
                    "nhp_violations": ep_data["nhp_viol"],
                    "avg_csi": avg_csi,  # 环境难度指标 1 (大=简单)
                    "avg_traffic": avg_traffic,  # 环境难度指标 2 (大=困难)
                    "scenario_difficulty": avg_traffic / (avg_csi + 1e-6)  # 综合难度 (高=难)
                })

                # Reset 临时变量
                ep_data = {
                    "reward_sum": 0, "csi_sum": 0, "traffic_sum": 0,
                    "steps": 0, "hp_viol": 0, "nhp_viol": 0
                }
                pbar.update(1)

    except KeyboardInterrupt:
        pass
    finally:
        pbar.close()
        env.close()

    # 4. 分析与报告
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Diagnosis saved to {output_csv}")

    analyze_diagnosis(df)


def analyze_diagnosis(df):
    print("\n" + "=" * 40)
    print("📊 DIAGNOSIS ANALYSIS REPORT")
    print("=" * 40)

    # 1. 相关性分析
    corr_matrix = df[['reward', 'hp_violations', 'nhp_violations', 'avg_csi', 'avg_traffic']].corr()
    print("\nCorrelation Matrix (Pearson):")
    print(corr_matrix[['reward', 'nhp_violations']].round(3))

    print("\n🔍 Key Insights:")

    # 分析 Reward 与 CSI (运气) 的关系
    reward_csi_corr = corr_matrix.loc['reward', 'avg_csi']
    print(f"1. Reward vs. Channel Quality (Luck): {reward_csi_corr:.3f}")
    if abs(reward_csi_corr) > 0.5:
        print("   ⚠️ High correlation! High rewards are heavily biased by good channel conditions.")
    else:
        print("   ✅ Low correlation. Reward function is relatively robust to channel noise.")

    # 分析 NHP Violation 与 Traffic (负载) 的关系
    nhp_traf_corr = corr_matrix.loc['nhp_violations', 'avg_traffic']
    print(f"2. NHP Violations vs. Traffic Load: {nhp_traf_corr:.3f}")

    # 2. 困难场景下的表现
    # 定义困难场景：Traffic 高 且 CSI 低
    hard_episodes = df[df['scenario_difficulty'] > df['scenario_difficulty'].quantile(0.75)]
    print(f"\n💀 Hard Mode Analysis (Top 25% Difficulty):")
    print(f"   Avg Reward: {hard_episodes['reward'].mean():.2f}")
    print(f"   Avg NHP Violations: {hard_episodes['nhp_violations'].mean():.2f}")

    # 3. 黄金样本筛选建议
    # 如果我们在 Hard Mode 下也能找到 reward 较高的样本，那才是真金
    good_in_hard = hard_episodes[hard_episodes['reward'] > df['reward'].median()]
    print(f"   ✨ Found {len(good_in_hard)} 'Golden' episodes in Hard Mode.")

    print("=" * 40 + "\n")


if __name__ == "__main__":
    config_path = "/root/decision_transformer_slicing/conf/environment/env_ha.yaml"
    # 请根据实际情况修改模型路径
    model_path = "/root/decision_transformer_slicing/data/channel_generality/ppo_ha/models/scenario_0/final_model_ha_best.zip"
    # 备选路径
    if not os.path.exists(model_path):
        model_path = "models_hierarchical/final_model_ha_best.zip"  # 示例

    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
        cfg.env_settings.inside.testing.active_scenario_list = [0]
        cfg.env_settings.model_name = 'diagnostic'
        work_dir = './diagnostic'
        pm = PathContext(work_dir, project_root='/root/decision_transformer_slicing')
        diagnose_agent(cfg, pm, model_path)
    else:
        print("Config not found.")