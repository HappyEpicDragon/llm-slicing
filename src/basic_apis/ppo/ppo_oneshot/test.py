import os
import numpy as np
import torch
from tqdm import tqdm
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

# === 关键修改 1: 引用 One-Shot 的环境和 Agent ===
from src.basic_apis.ppo.ppo_oneshot.global_slicing_env import GlobalSlicingEnv
from src.basic_apis.ppo.ppo_oneshot.agent_oneshot import CustomFeatureExtractor

from src.basic_apis.physics_probe import PhysicsProbe, SLAProbe


def make_env(cfg, path_manager, rank=0, seed=None):
    """
    One-Shot 环境工厂函数
    """

    def _init():
        # 确保使用的是 GlobalSlicingEnv
        env = GlobalSlicingEnv(cfg.env_settings, np.random.default_rng(seed + rank), path_manager)
        return env

    return _init


def test_ppo_oneshot(cfg, path_manager):
    environment_cfg = cfg.environment
    env_updates = cfg.env_updates

    # 更新环境配置
    environment_cfg.env_settings.mode = env_updates.mode
    environment_cfg.env_settings.scenario_mode = env_updates.scenario_mode
    environment_cfg.env_settings.model_name = env_updates.model_name
    environment_cfg.env_settings.inside.training.update(env_updates.inside.training)
    environment_cfg.env_settings.inside.evaluating.update(env_updates.inside.evaluating)
    environment_cfg.env_settings.inside.testing.update(env_updates.inside.testing)

    # 模型路径 (请根据实际情况修改或保持动态生成)
    # model_path = os.path.join(cfg.model_base_dir, env_updates.model_name, 'final_model.zip')
    # 或者使用最佳模型
    # model_path = os.path.join(cfg.model_base_dir, env_updates.model_name, 'best_model', 'best_model.zip')

    model_path = '/root/decision_transformer_slicing/data/channel_generality/ppo_oneshot/models/scenario_0/best_model/best_model.zip'

    test(model_path, environment_cfg, path_manager)
    # prove_blindness(model_path, environment_cfg, path_manager)


def test(model_path, cfg, path_manager):
    """
    One-Shot 模型评估脚本
    """
    # 1. 强制单线程 (避免评估时 CPU 竞争)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # 2. 创建环境
    seed = cfg.train_rl.evaluation.eval_env.seed
    env = DummyVecEnv([make_env(cfg, path_manager, rank=0, seed=seed)])

    # 计算需要评估的总 Scenario 数量
    scenario_mode = cfg.env_settings.scenario_mode
    n_scenarios = (
            cfg.env_settings[scenario_mode].testing.max_scenario_episodes -
            cfg.env_settings[scenario_mode].testing.init_scenario_episode
    )

    # One-Shot 架构下:
    # 如果 1 RL Step = 1 TTI, 且 1 Episode = 1000 Steps (1秒):
    # 总评估 Episodes = 场景数 * 每个场景包含的 Episodes (通常是1个长Episode或多个)
    # 这里假设我们要跑完所有场景的所有时间步
    total_test_episodes = n_scenarios  # 如果每个场景只跑 1 个由 1000 steps 组成的 Episode


    print(f"评估配置: 场景模式={scenario_mode}, 场景数={n_scenarios}")

    # 3. 加载模型
    # === 关键修改 2: 传入 custom_objects 以防 pickle 找不到类 ===
    print(f"加载模型: {model_path}")
    try:
        model = MaskablePPO.load(
            model_path,
            env=env,
            device='cpu',
            custom_objects={
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.1,
                # 确保反序列化时能找到 CustomFeatureExtractor
                "CustomFeatureExtractor": CustomFeatureExtractor
            }
        )
    except Exception as e:
        print(f"模型加载失败，尝试不带 custom_objects 加载: {e}")
        model = MaskablePPO.load(model_path, env=env, device='cpu')

    # 4. 初始化统计变量
    episode_rewards = []
    episode_lengths = []
    power_scales = []  # One-Shot 特有指标
    throughputs = []  # One-Shot 特有指标

    obs = env.reset()

    probe = PhysicsProbe()
    sla_probe = SLAProbe(env)

    # === 关键修改 3: One-Shot 不需要 LSTM States ===
    # lstm_states = None
    # episode_starts = ... (PPO 内部处理，predict 接口不需要显式传)

    current_ep_reward = 0.0
    current_ep_length = 0

    print(f"开始测试...")
    # 使用 total_test_episodes 控制进度条，或者手动控制
    pbar = tqdm(total=n_scenarios, unit="scenario")

    episodes_completed = 0

    try:
        step = 0
        while episodes_completed < n_scenarios:


            # A. 获取 Mask
            action_masks = env.env_method("action_masks")[0]

            # B. 预测 (移除 state 和 episode_start 参数)
            action, _ = model.predict(
                obs,
                action_masks=action_masks,
                deterministic=True  # 评估通常使用确定性策略
            )

            # C. 执行动作
            obs, rewards, dones, infos = env.step(action)
            # [✅ 修正] 取列表的第一个元素
            if len(infos) > 0:
                # 确保 info 里有这个键 (防止 Expert Override 没生效时报错)
                executed_action = infos[0].get('final_executed_action', action[0])
                probe.capture(env, executed_action, step, infos[0])
            else:
                # Fallback (不应该发生)
                probe.capture(env, action[0], step, {})

            executed_action = infos[0].get('final_executed_action', action[0])
            sla_probe.capture(executed_action, step, infos[0])

            # 累积奖励
            current_ep_reward += rewards[0]
            current_ep_length += 1

            # 收集 One-Shot 特有指标 (从 info 中)
            if len(infos) > 0:
                info = infos[0]
                if "power_scale_factor" in info:
                    power_scales.append(info["power_scale_factor"])
                if "throughput" in info:
                    throughputs.append(info["throughput"])
            step += 1

            # D. 处理结束
            if dones[0]:
                episode_rewards.append(current_ep_reward)
                episode_lengths.append(current_ep_length)
                episodes_completed += 1
                step = 0
                probe.report()
                sla_probe.report(episodes_completed - 1)

                probe = PhysicsProbe()
                sla_probe = SLAProbe(env)

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    'Rew': f"{current_ep_reward:.2f}",
                    'Scale': f"{infos[0].get('power_scale_factor', 0):.2f}"
                })

                # 重置累积变量
                current_ep_reward = 0.0
                current_ep_length = 0

                # DummyVecEnv 会自动 reset，obs 已经是新的了

    except KeyboardInterrupt:
        print("\n用户中断测试")
    finally:
        env.close()
        pbar.close()

    if len(episode_rewards) > 0:
        print(f"\n=== 测试结果 (One-Shot) ===")
        print(f"平均奖励: {np.mean(episode_rewards):.4f} (std: {np.std(episode_rewards):.4f})")
        print(f"平均步数: {np.mean(episode_lengths):.2f}")

        if len(power_scales) > 0:
            print(f"平均功率缩放因子: {np.mean(power_scales):.4f}")
        if len(throughputs) > 0:
            # 假设 throughput 单位是 bits，转换为 Mbps (假设 step 是 1ms)
            # 注意：这里的 throughput 是累积的还是平均的取决于怎么统计
            avg_thr_mbps = np.mean(throughputs) / 1e6 * 1000
            # 如果 info['throughput'] 已经是 step 内的总量，且我们关注的是平均速率
            print(f"平均吞吐量指标: {np.mean(throughputs):.2e} (Raw Units)")

    return