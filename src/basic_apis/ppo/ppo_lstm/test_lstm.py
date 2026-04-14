import os
import numpy as np
import torch
from tqdm import tqdm
from sb3_contrib import MaskablePPO
# 引入 DummyVecEnv 以保持与 SB3 接口的最佳兼容性
from stable_baselines3.common.vec_env import DummyVecEnv

# 引入之前定义的 make_env
from src.basic_apis.ppo.ppo_lstm.train_recurrent_ppo import make_env


def test_ppo_lstm(cfg, path_manager):
    environment_cfg = cfg.environment
    env_updates = cfg.env_updates
    environment_cfg.env_settings.mode = env_updates.mode
    environment_cfg.env_settings.scenario_mode = env_updates.scenario_mode
    environment_cfg.env_settings.model_name = env_updates.model_name
    environment_cfg.env_settings.inside.training.update(env_updates.inside.training)
    environment_cfg.env_settings.inside.evaluating.update(env_updates.inside.evaluating)
    environment_cfg.env_settings.inside.testing.update(env_updates.inside.testing)
    # model_path = os.path.join(cfg.model_base_dir, env_updates.model_name, 'best_model', 'best_model.zip')
    model_path = '/root/decision_transformer_slicing/data/channel_generality/ppo_lstm/models/sla_rbg/best_model.zip'
    test(model_path, environment_cfg, path_manager)


def test(model_path, cfg, path_manager):
    """
    单环境串行评估（手动累积奖励修复版）
    """
    # 1. 强制单线程
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # 2. 创建环境
    seed = cfg.train_rl.evaluation.eval_env.seed
    # 注意：即使这里没有 Monitor 也没关系，我们下面手动算
    env = DummyVecEnv([make_env(cfg, path_manager, rank=0, seed=seed)])

    scenario_mode = cfg.env_settings.scenario_mode
    n_eval_episodes = (
                              cfg.env_settings[scenario_mode].testing.max_scenario_episodes -
                              cfg.env_settings[scenario_mode].testing.init_scenario_episode
                      ) * 1000

    # 3. 加载模型 (CPU)
    print(f"加载模型: {model_path}")
    model = MaskablePPO.load(model_path, env=env, device='cpu')

    # 4. 初始化变量
    episode_rewards = []
    episode_lengths = []

    obs = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    # === 新增：临时累积变量 ===
    current_ep_reward = 0.0
    current_ep_length = 0

    print(f"开始测试 (共 {n_eval_episodes} 个 Episodes)...")
    pbar = tqdm(total=n_eval_episodes, unit="ep")

    try:
        while len(episode_rewards) < n_eval_episodes:

            # A. 获取 Mask
            action_masks = env.env_method("action_masks")[0]

            # [新增调试]
            if len(episode_rewards) == 0 and current_ep_length % 50 == 0:
                print(f"Test Script Mask Check: Action 7 Allowed? {action_masks[7]}")

            # B. 预测
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                action_masks=action_masks,
                deterministic=cfg.train_rl.evaluation.deterministic
            )

            # C. 执行动作
            obs, rewards, dones, infos = env.step(action)

            # === 修复核心：手动累积奖励 ===
            # rewards[0] 是当前步的奖励（float）
            current_ep_reward += rewards[0]
            current_ep_length += 1

            # 更新 episode_starts
            episode_starts = dones

            # D. 处理结束
            if dones[0]:
                # 当 done=True 时，current_ep_reward 已经包含了最后一步的奖励
                # 记录结果
                episode_rewards.append(current_ep_reward)
                episode_lengths.append(current_ep_length)

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    'Rew': f"{current_ep_reward:.2f}",
                    'Len': f"{current_ep_length}"
                })

                # === 重置累积变量 ===
                current_ep_reward = 0.0
                current_ep_length = 0

                # 注意：不需要手动 env.reset()，DummyVecEnv 已经自动重置了
                # obs 已经是新 Episode 的第一帧了

    except KeyboardInterrupt:
        print("\n用户中断测试")
    finally:
        env.close()
        pbar.close()

    if len(episode_rewards) > 0:
        print(f"\n=== 测试结果 ===")
        print(f"平均奖励: {np.mean(episode_rewards):.4f} (std: {np.std(episode_rewards):.4f})")
        print(f"平均步数: {np.mean(episode_lengths):.2f}")

    return
