import os
import numpy as np
import torch
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from omegaconf import OmegaConf

# === Local Imports ===
from src.basic_apis.ppo.ppo_ha.hierarchical_slicing_env import HierarchicalSlicingEnv
from src.basic_apis.ppo.ppo_ha.agent_hierarchical import HierarchicalSmartPolicy
from src.basic_apis.network_slicing_business.path_manager import PathManager
from src.basic_apis.physics_probe import PhysicsProbe, SLAProbe


def make_env(cfg, path_manager, rank=0, seed=0):
    """H+A 环境工厂函数"""

    def _init():
        env_config = cfg.env_settings if hasattr(cfg, 'env_settings') else cfg
        if 'env_settings' in cfg:
            env_config = cfg.env_settings

        env = HierarchicalSlicingEnv(env_config, np.random.default_rng(seed + rank), path_manager)
        return env

    return _init


def test_ppo_ha(cfg, path_manager):
    """H+A 测试入口"""
    environment_cfg = cfg.environment
    env_updates = cfg.env_updates

    # 更新环境配置
    environment_cfg.env_settings.mode = env_updates.mode
    environment_cfg.env_settings.scenario_mode = env_updates.scenario_mode
    environment_cfg.env_settings.model_name = env_updates.model_name
    environment_cfg.env_settings.inside.training.update(env_updates.inside.training)
    environment_cfg.env_settings.inside.evaluating.update(env_updates.inside.evaluating)
    environment_cfg.env_settings.inside.testing.update(env_updates.inside.testing)

    # 2. 模型路径
    # 这里请根据你的实际路径修改，或者使用传入的参数
    # model_path = "/root/decision_transformer_slicing/data/channel_generality/ppo_ha/models/scenario_0/best_model_hierarchical/best_model.zip"
    model_path = cfg.model_path
    if not os.path.exists(model_path):
        # 尝试 final model
        model_path = "/root/decision_transformer_slicing/final_model_ha_cont.zip"
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found at default paths. Please check: {model_path}")
            # 你可以在这里硬编码一个你刚刚训练好的模型路径，例如：
            # model_path = "/root/decision_transformer_slicing/models_hierarchical/ppo_ha_cont_19200_steps.zip"

    print(f"Loading Model from: {model_path}")
    test(model_path, environment_cfg, path_manager)


def test(model_path, cfg, path_manager):
    # 1. 环境设置
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # 获取测试配置
    scenario_mode = cfg.env_settings.scenario_mode
    mode = cfg.env_settings.mode
    test_cfg = cfg.env_settings[scenario_mode][mode]

    n_scenarios = test_cfg.max_scenario_episodes - test_cfg.init_scenario_episode
    print(f"Test Config: Mode={mode}, Scenarios={n_scenarios}")

    # 创建环境
    seed = 42
    env = DummyVecEnv([make_env(cfg, path_manager, rank=0, seed=seed)])

    # 2. 加载模型
    try:
        model = PPO.load(
            model_path,
            env=env,
            device='cpu',
            custom_objects={
                "learning_rate": 0.0,
                "clip_range": 0.1
            }
        )
    except Exception as e:
        print(f"Standard load failed, trying custom injection: {e}")
        model = PPO.load(
            model_path,
            env=env,
            device='cpu',
            custom_objects={
                "HierarchicalSmartPolicy": HierarchicalSmartPolicy
            }
        )

    # 3. [关键修正] 先 Reset 环境以初始化 Physics Components
    print("Initializing environment components...")
    obs = env.reset()

    # 4. 初始化 Probe (现在 env.components 不再是 None)
    probe = PhysicsProbe()
    sla_probe = SLAProbe(env)

    # 5. 测试循环
    episodes_completed = 0
    pbar = tqdm(total=n_scenarios, unit="ep")

    ep_rewards = []

    try:
        step = 0
        current_ep_reward = 0

        while episodes_completed < n_scenarios:
            # A. 预测
            action, _ = model.predict(obs, deterministic=True)

            # B. 执行
            obs, rewards, dones, infos = env.step(action)
            info = infos[0]

            # C. 记录数据
            raw_action = action[0]  # Continuous Logits (30,)

            # SLA Probe 捕获
            # 这里传入 Logits，SLAProbe 内部如果只做统计没问题
            # 如果需要具体分配，需要从 info['final_executed_action'] 获取
            # 建议修改 Probe 调用以使用 info 中的物理动作
            executed_action = info.get('final_executed_action', raw_action)
            sla_probe.capture(executed_action, step, info)

            # Physics Probe
            probe.capture(env, executed_action, step, info)

            current_ep_reward += rewards[0]
            step += 1

            if dones[0]:
                episodes_completed += 1
                ep_rewards.append(current_ep_reward)

                print(f"\n[Episode {episodes_completed}] Reward: {current_ep_reward:.2f}")

                # 报告 Probe
                # 注意：report 可能会重置内部统计
                sla_probe.report(episodes_completed - 1)

                # 重置 Probe
                # DummyVecEnv 会自动 reset 环境，components 会被重建
                # 我们需要重新绑定 Probe 到新的 components (通过重新初始化)
                probe = PhysicsProbe()
                sla_probe = SLAProbe(env)

                current_ep_reward = 0
                step = 0
                pbar.update(1)

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        env.close()
        pbar.close()

    print(f"\n=== H+A Test Results ===")
    print(f"Avg Reward: {np.mean(ep_rewards):.2f}")


if __name__ == "__main__":
    config_path = "hierarchical_env.yaml"
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
        work_dir = os.getcwd()
        pm = PathManager(work_dir)

        cfg.env_settings.mode = 'testing'
        if 'testing' not in cfg.env_settings.inside:
            cfg.env_settings.inside.testing = cfg.env_settings.inside.evaluating

        test_ppo_ha(cfg, pm)
    else:
        print("Config not found.")