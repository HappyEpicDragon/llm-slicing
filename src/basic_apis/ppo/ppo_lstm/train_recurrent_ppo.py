# train_with_config.py
import os

# === 关键性能优化 ===
# 强制 NumPy 和 PyTorch 只使用 1 个 CPU 线程
# 这可以避免多进程训练时的 CPU 竞争和上下文切换开销
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from omegaconf import DictConfig, OmegaConf
import numpy as np
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback  # ✅ 添加BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from src.basic_apis.ppo.ppo_lstm.environment_psra import PSJRAEnv
from src.basic_apis.ppo.ppo_lstm.agent_psra import MaskableLSTMPolicy, LSTMStateResetCallback

# ==================== ✅ 1.2 第一处修改：导入诊断工具（在文件开头） ====================
from src.basic_apis.ppo.ppo_lstm.diagnosis_tools import RecurrentPPODiagnostics
import torch


# ==================== 修改结束 ====================


def make_env(cfg, path_manager, rank=0, seed=None):
    """
    创建单个环境的工厂函数
    """

    def _init():
        if seed is None:
            env_seed = np.random.randint(0, 2 ** 31 - 1) + rank
        else:
            env_seed = seed + rank

        np_random = np.random.default_rng(env_seed)
        env = PSJRAEnv(cfg.env_settings, np_random, path_manager)
        env._np_random_seed = env_seed
        if cfg.env_settings.mode != 'testing':
            env = Monitor(env)
        return env

    return _init


# ==================== ✅ 1.3 第二处修改：添加详细日志回调类（在train函数之前） ====================
class DetailedLoggingCallback(BaseCallback):
    """详细日志回调 - 监控训练细节"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # 每100步记录一次梯度和价值函数信息
        if self.num_timesteps % 100 == 0:
            # 1. 记录梯度范数
            if hasattr(self.model, 'policy'):
                total_norm = 0
                param_count = 0
                for p in self.model.policy.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1

                if param_count > 0:
                    total_norm = total_norm ** 0.5
                    self.logger.record("train/gradient_norm", total_norm)

            # 2. 记录价值函数预测
            if 'values' in self.locals:
                values = self.locals['values']

                # ✅ 修复：转换 PyTorch tensor 为 NumPy array
                if isinstance(values, torch.Tensor):
                    values = values.detach().cpu().numpy()

                self.logger.record("train/value_mean", float(np.mean(values)))
                self.logger.record("train/value_std", float(np.std(values)))
                self.logger.record("train/value_max", float(np.max(values)))
                self.logger.record("train/value_min", float(np.min(values)))

            # 3. 记录动作分布（如果可用）
            if 'actions' in self.locals:
                actions = self.locals['actions']

                # ✅ 修复：处理 PyTorch tensor
                if isinstance(actions, torch.Tensor):
                    actions = actions.detach().cpu().numpy()

                if len(actions.shape) > 1:  # MultiDiscrete
                    user_ids = actions[:, 0]
                    power_levels = actions[:, 1]

                    self.logger.record("train/action_user_mean", float(np.mean(user_ids)))
                    self.logger.record("train/action_user_std", float(np.std(user_ids)))
                    self.logger.record("train/action_power_mean", float(np.mean(power_levels)))
                    self.logger.record("train/action_power_std", float(np.std(power_levels)))

        # 记录episode信息
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])

                    # 计算最近100个episode的统计
                    if len(self.episode_rewards) >= 100:
                        recent_rewards = self.episode_rewards[-100:]
                        self.logger.record("rollout/ep_rew_mean_100", float(np.mean(recent_rewards)))
                        self.logger.record("rollout/ep_rew_std_100", float(np.std(recent_rewards)))

        return True


# class DiagnosticCallback(BaseCallback):
#     """诊断回调 - 收集动作分布和LSTM状态信息"""
#
#     def __init__(self, diagnostics, verbose=0):
#         super().__init__(verbose)
#         self.diagnostics = diagnostics
#         self.episode_count = 0
#
#     def _on_step(self) -> bool:
#         # 获取当前步的信息
#         if 'infos' in self.locals:
#             actions = self.locals.get('actions', None)
#             rewards = self.locals.get('rewards', None)
#
#             if actions is not None and rewards is not None:
#                 # ✅ 修复：转换 PyTorch tensor
#                 if isinstance(actions, torch.Tensor):
#                     actions = actions.detach().cpu().numpy()
#                 if isinstance(rewards, torch.Tensor):
#                     rewards = rewards.detach().cpu().numpy()
#
#                 # 处理每个环境的动作和奖励
#                 for i, (action, reward) in enumerate(zip(actions, rewards)):
#                     # 获取LSTM状态（如果可用）
#                     lstm_states = None
#                     if hasattr(self.model.policy, 'features_extractor'):
#                         extractor = self.model.policy.features_extractor
#                         if hasattr(extractor, '_lstm_h'):
#                             try:
#                                 # 提取该环境的LSTM状态
#                                 lstm_states = {
#                                     i: (
#                                         extractor._lstm_h[:, i:i + 1, :].clone(),
#                                         extractor._lstm_c[:, i:i + 1, :].clone()
#                                     )
#                                 }
#                             except (IndexError, RuntimeError):
#                                 # 状态维度不匹配时跳过
#                                 lstm_states = None
#
#                     # 收集数据
#                     self.diagnostics.collect_step_data(
#                         timestep=self.num_timesteps + i,
#                         action=action,
#                         reward=float(reward),
#                         observation={},
#                         lstm_states=lstm_states
#                     )
#
#         # 每10000步保存一次中间报告
#         if self.num_timesteps % 10000 == 0 and self.num_timesteps > 0:
#             print(f"\n{'=' * 80}")
#             print(f"[诊断报告] 训练步数: {self.num_timesteps:,}")
#             print(f"{'=' * 80}")
#             self.diagnostics.analyze_action_distribution()
#             self.diagnostics.analyze_exploration()
#             print(f"{'=' * 80}\n")
#
#         return True
#
#     def _on_training_end(self):
#         """训练结束时保存完整报告"""
#         print("\n" + "=" * 80)
#         print("生成最终诊断报告...")
#         print("=" * 80)
#         self.diagnostics.save_report()
#         print("=" * 80 + "\n")


# ==================== 修改结束 ====================


def train(cfg: DictConfig, path_manager):
    """使用Hydra配置进行训练"""

    # ==================== 原有代码：创建输出目录 ====================
    env_config = cfg.environment
    env_updates = cfg.env_updates
    env_config.env_settings.mode = env_updates.mode
    env_config.env_settings.scenario_mode = env_updates.scenario_mode
    env_config.env_settings.model_name = env_updates.model_name
    env_config.env_settings.inside.training.update(env_updates.inside.training)
    env_config.env_settings.inside.evaluating.update(env_updates.inside.evaluating)
    env_config.env_settings.inside.testing.update(env_updates.inside.testing)

    save_path = Path(env_config.train_rl.saving.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    log_path = Path(env_config.train_rl.logging.log_path)
    log_path.mkdir(parents=True, exist_ok=True)

    best_model_path = Path(env_config.train_rl.saving.best_model.save_path)
    best_model_path.mkdir(parents=True, exist_ok=True)

    seed = env_config.train_rl.algorithm.seed
    # ==================== 原有代码结束 ====================

    # ==================== ✅ 1.2 第三处修改：创建诊断工具实例 ====================
    print("\n" + "=" * 80)
    print("初始化诊断工具...")
    print("=" * 80)

    diagnostics = RecurrentPPODiagnostics(
        save_dir=str(save_path / "diagnostics")
    )

    print(f"✓ 诊断工具已初始化")
    print(f"  保存路径: {save_path / 'diagnostics'}")
    print("=" * 80 + "\n")
    # ==================== 修改结束 ====================

    # ==================== 原有代码：创建环境 ====================
    n_envs = env_config.train_rl.environment.n_envs
    print(f"\n创建 {n_envs} 个并行训练环境...")

    if n_envs > 1:
        env = DummyVecEnv([
            make_env(env_config, path_manager, rank=i, seed=seed)
            for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(env_config, path_manager, rank=0, seed=seed)
        ])

    if env_config.train_rl.debug.check_env:
        print("检查环境有效性...")
        try:
            test_env = PSJRAEnv(
                env_config.env_settings,
                np.random.default_rng(seed),
                path_manager
            )
            check_env(test_env, warn=True)
            test_env.close()
            print("✓ 环境检查通过")
        except Exception as e:
            print(f"⚠ 环境检查失败: {e}")
            if env_config.train_rl.debug.verbose_errors:
                import traceback
                traceback.print_exc()

    # ==================== 创建评估环境 ====================
    eval_env = None
    if env_config.train_rl.callbacks.eval.enabled and env_config.train_rl.evaluation.eval_freq > 0:
        print("创建评估环境...")
        eval_seed = env_config.train_rl.evaluation.eval_env.seed
        env_config.env_settings.mode = 'evaluating'
        eval_env = DummyVecEnv([
            make_env(env_config, path_manager, rank=999, seed=eval_seed)
        ])
    # ==================== 原有代码结束 ====================

    # ==================== 原有代码：创建模型 ====================
    print("\n初始化MaskablePPO模型...")

    if env_config.train_rl.ppo.lr_schedule.enabled:
        print("⚠ 学习率调度功能尚未实现，使用固定学习率")
        learning_rate = env_config.train_rl.ppo.learning_rate
    else:
        learning_rate = env_config.train_rl.ppo.learning_rate

    model = MaskablePPO(
        policy=MaskableLSTMPolicy,
        env=env,
        learning_rate=learning_rate,
        n_steps=env_config.train_rl.ppo.n_steps,
        batch_size=env_config.train_rl.ppo.batch_size,
        n_epochs=env_config.train_rl.ppo.n_epochs,
        gamma=env_config.train_rl.ppo.gamma,
        gae_lambda=env_config.train_rl.ppo.gae_lambda,
        clip_range=env_config.train_rl.ppo.clip_range,
        clip_range_vf=env_config.train_rl.ppo.clip_range_vf,
        ent_coef=env_config.train_rl.ppo.ent_coef,
        vf_coef=env_config.train_rl.ppo.vf_coef,
        max_grad_norm=env_config.train_rl.ppo.max_grad_norm,
        target_kl=env_config.train_rl.ppo.target_kl,
        verbose=env_config.train_rl.logging.verbose,
        tensorboard_log=env_config.train_rl.logging.tensorboard_log,
        device=env_config.train_rl.algorithm.device,
        seed=seed,
    )

    print(f"✓ 模型初始化完成")
    print(f"  - 设备: {env_config.train_rl.algorithm.device}")
    print(f"  - 学习率: {learning_rate}")
    print(f"  - 并行环境数: {n_envs}")
    # ==================== 原有代码结束 ====================

    # ==================== 原有代码：创建回调（开始） ====================
    callbacks = []

    # 1. LSTM状态重置回调
    if env_config.train_rl.callbacks.lstm_reset.enabled:
        print("添加LSTM状态重置回调")
        lstm_callback = LSTMStateResetCallback(
            verbose=env_config.train_rl.callbacks.lstm_reset.verbose
        )
        callbacks.append(lstm_callback)

    # 2. 检查点保存回调
    if env_config.train_rl.callbacks.checkpoint.enabled:
        print(f"添加检查点保存回调 (每 {env_config.train_rl.callbacks.checkpoint.save_freq} 步)")
        checkpoint_callback = CheckpointCallback(
            save_freq=env_config.train_rl.callbacks.checkpoint.save_freq,
            save_path=env_config.train_rl.callbacks.checkpoint.save_path,
            name_prefix=env_config.train_rl.callbacks.checkpoint.name_prefix,
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        callbacks.append(checkpoint_callback)

    # 3. 评估回调
    if env_config.train_rl.callbacks.eval.enabled and eval_env is not None:
        print(f"添加评估回调 (每 {env_config.train_rl.callbacks.eval.eval_freq} 步)")
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=env_config.train_rl.callbacks.eval.best_model_save_path,
            log_path=env_config.train_rl.logging.log_path,
            eval_freq=env_config.train_rl.callbacks.eval.eval_freq,
            n_eval_episodes=env_config.train_rl.callbacks.eval.n_eval_episodes,
            deterministic=env_config.train_rl.callbacks.eval.deterministic,
            render=False,
        )
        callbacks.append(eval_callback)
    # ==================== 原有代码结束 ====================

    # ==================== ✅ 1.2 & 1.3 第四处修改：添加诊断回调 ====================
    # 添加诊断回调
    # print("添加诊断回调")
    # diag_callback = DiagnosticCallback(
    #     diagnostics=diagnostics,
    #     verbose=1
    # )
    # callbacks.append(diag_callback)

    # 添加详细日志回调
    # print("添加详细日志回调 (梯度、价值函数等)")
    # detailed_logging_callback = DetailedLoggingCallback(verbose=1)
    # callbacks.append(detailed_logging_callback)
    # ==================== 修改结束 ====================

    print(f"\n总共启用了 {len(callbacks)} 个回调")

    # ==================== 原有代码：开始训练 ====================
    print("\n" + "=" * 80)
    print("开始训练...")
    print("=" * 80)
    print(f"总训练步数: {env_config.train_rl.training.total_timesteps:,}")
    print(f"日志间隔: {env_config.train_rl.training.log_interval} episodes")
    print("=" * 80 + "\n")

    try:
        scenario_mode = env_config.env_settings.scenario_mode
        training_settings = env_config.env_settings[scenario_mode].training
        channel_episode_num = training_settings.max_scenario_episodes - training_settings.init_scenario_episode
        effective_channel_timestep = 1000 # 1000
        # env_config.train_rl.training.total_timesteps = 27 * effective_channel_timestep * channel_episode_num * len(
        #     training_settings.active_scenario_list)
        model.learn(
            total_timesteps=env_config.train_rl.training.total_timesteps,
            callback=callbacks if callbacks else None,
            log_interval=env_config.train_rl.training.log_interval,
            progress_bar=env_config.train_rl.training.progress_bar,
        )
    except KeyboardInterrupt:
        print("\n⚠ 训练被用户中断!")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        if env_config.train_rl.debug.verbose_errors:
            import traceback
            traceback.print_exc()
        raise
    # ==================== 原有代码结束 ====================

    # ==================== 原有代码：保存模型 ====================
    print("\n" + "=" * 80)
    print("保存最终模型...")

    final_model_path = save_path / "final_model"
    model.save(str(final_model_path))
    print(f"✓ 模型已保存至: {final_model_path}")

    config_path = save_path / "training_config.yaml"
    with open(config_path, 'w') as f:
        OmegaConf.save(env_config, f)
    print(f"✓ 配置已保存至: {config_path}")

    stats_path = save_path / "training_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"实验名称: {env_config.train_rl.experiment.name}\n")
        f.write(f"实验组: {env_config.train_rl.experiment.group}\n")
        f.write(f"标签: {', '.join(env_config.train_rl.experiment.tags)}\n")
        f.write(f"总训练步数: {env_config.train_rl.training.total_timesteps:,}\n")
        f.write(f"并行环境数: {n_envs}\n")
        f.write(f"学习率: {learning_rate}\n")
        f.write(f"设备: {env_config.train_rl.algorithm.device}\n")
    print(f"✓ 训练统计已保存至: {stats_path}")

    print("\n清理资源...")
    env.close()
    if eval_env is not None:
        eval_env.close()

    print("\n" + "=" * 80)
    print("✓ 训练完成!")
    print("=" * 80)
    print(f"模型保存位置: {final_model_path}")
    print(f"日志保存位置: {log_path}")
    if eval_env is not None:
        print(f"最佳模型位置: {best_model_path}")

    # ==================== ✅ 第五处修改：添加诊断报告路径 ====================
    print(f"诊断报告位置: {save_path / 'diagnostics'}")
    # ==================== 修改结束 ====================

    print("=" * 80)

    return model


if __name__ == "__main__":
    print("请从main.py运行训练脚本")