import os
import numpy as np
import torch
import pickle
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Callable

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback
)

# Local Imports
from src.basic_apis.ppo.ppo_ha.hierarchical_slicing_env import HierarchicalSlicingEnv
from src.basic_apis.ppo.ppo_ha.agent_hierarchical import HierarchicalSmartPolicy
from src.basic_apis.network_slicing_business.path_context import PathContext


# =========================================================================
# 0. 黄金缓冲池 (Golden Buffer Callback) - DT 的弹药库
# =========================================================================
class GoldenBufferCallback(BaseCallback):
    def __init__(self, save_dir, reward_threshold=-5.0, verbose=0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.reward_threshold = reward_threshold
        os.makedirs(save_dir, exist_ok=True)
        self.current_ep_data = {"obs": [], "actions": [], "rewards": []}

    def _on_step(self) -> bool:
        # 收集原始数据
        obs = self.locals['new_obs']
        action = self.locals['actions']
        reward = self.locals['rewards']
        done = self.locals['dones']

        self.current_ep_data["obs"].append(obs)
        self.current_ep_data["actions"].append(action)
        self.current_ep_data["rewards"].append(reward)

        if done[0]:
            ep_reward = np.sum(self.current_ep_data["rewards"])

            # 如果是表现好的 Episode，保存下来
            if ep_reward > self.reward_threshold:
                self._save_trajectory(ep_reward)

            # Reset
            self.current_ep_data = {"obs": [], "actions": [], "rewards": []}
        return True

    def _save_trajectory(self, score):
        # 简单序列化
        filename = os.path.join(self.save_dir, f"golden_ep_{self.num_timesteps}_score_{score:.2f}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(self.current_ep_data, f)
        if self.verbose > 0:
            print(f"✨ Golden Buffer: Saved trajectory with score {score:.2f}")



# ... (ConsoleLogCallback, PeriodicMetricsCallback, HierarchicalDiagnosisCallback 保持不变，省略以节省空间) ...
# 请务必保留这些监控回调！复制之前的代码即可。
class ConsoleLogCallback(BaseCallback):
    def __init__(self, verbose=0, print_freq=200):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.ep_stats = {"HP_viol": 0, "NHP_viol": 0}
        self.RED = "\033[91m";
        self.GREEN = "\033[92m";
        self.YELLOW = "\033[93m";
        self.GRAY = "\033[90m";
        self.CYAN = "\033[96m";
        self.RESET = "\033[0m"

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}]);
        info = infos[0] if infos else {}
        if "reward/base" in info:
            total_rew = info.get("reward/base", 0.0) + info.get("reward/risk", 0.0)
            self.logger.record("reward/total_step", total_rew)
        if self.n_calls % 1000 == 1: self.ep_stats = {"HP_viol": 0, "NHP_viol": 0}
        self._quick_accumulate(info)
        if self.n_calls % self.print_freq == 0: self._print_console_report(info)
        return True

    def _quick_accumulate(self, info):
        for k in info.keys():
            if k.startswith("drift/slice_"):
                try:
                    parts = k.split('_');
                    s_idx = int(parts[1]);
                    metric = parts[2]
                    if info.get(f"meta/slice_{s_idx}_active", 0) == 0: continue
                    if info.get(f"meta/slice_{s_idx}_{metric}_req", 0) == 0: continue
                    if info[k] < 0:
                        prio = info.get(f"meta/slice_{s_idx}_priority", 0)
                        if prio > 0:
                            self.ep_stats['HP_viol'] += 1
                        else:
                            self.ep_stats['NHP_viol'] += 1
                except:
                    continue

    def _print_console_report(self, info):
        total_steps = self.num_timesteps
        base = info.get("reward/base", 0.0);
        risk = info.get("reward/risk", 0.0);
        total = base + risk
        print(f"\n{'=' * 95}")
        print(f"🚀 [Step {total_steps}] H+A Monitor (Continuous) | Ep Progress: {total_steps % 1000}/1000")
        print(f"{'=' * 95}")
        hp_v = self.ep_stats['HP_viol'];
        nhp_v = self.ep_stats['NHP_viol']
        c_hp = self.RED if hp_v > 0 else self.GREEN;
        c_nhp = self.YELLOW if nhp_v > 0 else self.GREEN
        print(f"📊 Violations: HP {c_hp}{hp_v}{self.RESET} | NHP {c_nhp}{nhp_v}{self.RESET}")
        print(f"💰 Rewards   : Base {base:+.2f} | Risk {risk:+.2f} | {self.CYAN}Sum {total:+.2f}{self.RESET}")
        print(f"{'-' * 95}")
        print(f"{'Slice':<6} | {'Thr (Drift)':<22} | {'Lat (Drift)':<22} | {'Rel (Drift)':<22}")
        print(f"{'-' * 95}")
        slice_indices = set()
        for key in info.keys():
            if key.startswith("drift/slice_"):
                parts = key.split('_');
                if len(parts) >= 2 and parts[1].isdigit(): slice_indices.add(int(parts[1]))
        for s_idx in sorted(list(slice_indices)):
            if info.get(f"meta/slice_{s_idx}_active", 0) == 0: continue
            prio = info.get(f"meta/slice_{s_idx}_priority", 0)
            prio_mark = f"{self.RED}*{self.RESET}" if prio > 0 else " "
            row_str = f"S-{s_idx:<4}{prio_mark}| "
            for metric in ['thr', 'lat', 'rel']: row_str += self._format_metric_cell(info, s_idx, metric)
            print(row_str)
        print(f"{'=' * 95}\n")

    def _format_metric_cell(self, info, s_idx, metric):
        is_req = info.get(f"meta/slice_{s_idx}_{metric}_req", 0)
        if not is_req: return f"{self.GRAY}{'---':<22}{self.RESET} | "
        val = info.get(f"drift/slice_{s_idx}_{metric}", 0.0)
        color = self.RED if val < 0 else self.GREEN;
        status = "Viol" if val < 0 else "OK"
        return f"{color}{f'{val:+.4f} ({status})':<22}{self.RESET} | "


class PeriodicMetricsCallback(BaseCallback):
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose);
        self.check_freq = check_freq;
        self._reset_stats()
        self.CYAN = "\033[96m";
        self.RESET = "\033[0m"

    def _reset_stats(self):
        self.stats = {"HP": {"viol": 0, "count": 0}, "NHP": {"viol": 0, "count": 0}};
        self.current_step_in_period = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}]);
        info = infos[0] if infos else {}
        self._accumulate(info);
        self.current_step_in_period += 1
        if self.current_step_in_period >= self.check_freq: self._print_summary(); self._reset_stats()
        return True

    def _accumulate(self, info):
        slice_indices = set()
        for k in info.keys():
            if k.startswith("meta/slice_") and k.endswith("_priority"): slice_indices.add(int(k.split('_')[1]))
        for s_idx in slice_indices:
            if info.get(f"meta/slice_{s_idx}_active", 0) == 0: continue
            prio = info.get(f"meta/slice_{s_idx}_priority", 0)
            cat = "HP" if prio > 0 else "NHP"
            for metric in ['thr', 'rel', 'lat']:
                if info.get(f"meta/slice_{s_idx}_{metric}_req", 0) == 0: continue
                drift = info.get(f"drift/slice_{s_idx}_{metric}", 0.0)
                self.stats[cat]["count"] += 1
                if drift < 0: self.stats[cat]["viol"] += 1

    def _print_summary(self):
        hp = self.stats["HP"];
        nhp = self.stats["NHP"]
        hp_rate = hp["viol"] / hp["count"] if hp["count"] > 0 else 0.0
        nhp_rate = nhp["viol"] / nhp["count"] if nhp["count"] > 0 else 0.0
        print(f"\n{self.CYAN}=== EPISODE SUMMARY (Last {self.current_step_in_period} Steps) ==={self.RESET}")
        print(f"HP Violations : {hp['viol']}/{hp['count']} ({hp_rate:.2%})")
        print(f"NHP Violations: {nhp['viol']}/{nhp['count']} ({nhp_rate:.2%})")
        print(f"{self.CYAN}{'=' * 60}{self.RESET}\n")


class HierarchicalDiagnosisCallback(BaseCallback):
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose);
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            actions = self.locals.get("actions", None)
            if actions is not None:
                action = actions[0];
                inter_logits = action[:5]
                print(f"[Internal Diag] Action Logits Stats:")
                print(f"                Min: {action.min():.3f} | Max: {action.max():.3f} | Mean: {action.mean():.3f}")
                print(f"                Inter-Slice Logits: {inter_logits}")
                exps = np.exp(inter_logits - np.max(inter_logits));
                probs = exps / np.sum(exps)
                print(f"                Inter Probabilities: {np.round(probs, 3)}")
        return True


# =========================================================================
# 2. 核心训练接口
# =========================================================================
def make_env(cfg, path_context, rank=0, seed=0):
    def _init():
        env_config = cfg.env_settings if hasattr(cfg, 'env_settings') else cfg
        if 'env_settings' in cfg: env_config = cfg.env_settings
        env = HierarchicalSlicingEnv(env_config, np.random.default_rng(seed + rank), path_context)
        if hasattr(cfg, 'train_rl') and hasattr(cfg.train_rl, 'reward_weights'):
            env.reward_weights = cfg.train_rl.reward_weights
        env = Monitor(env)
        return env

    return _init


# =========================================================================
# 1. 学习率与Clip Range衰减函数生成器
# =========================================================================
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    线性学习率衰减: progress 1.0 -> 0.0
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def clip_range_schedule(initial_value: float, final_value: float = 0.05) -> Callable[[float], float]:
    """
    Clip Range 线性衰减
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining 从 1.0 降到 0.0
        # 我们希望 clip_range 从 initial_value 降到 final_value
        return final_value + progress_remaining * (initial_value - final_value)
    return func

def entropy_schedule(initial_value: float) -> Callable[[float], float]:
    """
    线性熵衰减
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining: 1.0 -> 0.0
        # 衰减到初始值的 1/100 或固定下限
        min_ent = 0.0001
        return min_ent + progress_remaining * (initial_value - min_ent)
    return func

class EntropyScheduleCallback(BaseCallback):
    """
    专门用于动态调整 PPO ent_coef 的回调函数
    """

    def __init__(self, schedule_func: Callable[[float], float], total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.schedule_func = schedule_func
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # 计算当前的进度 (1.0 -> 0.0)
        # num_timesteps 是当前步数
        progress_remaining = 1.0 - (self.num_timesteps / self.total_timesteps)

        # 限制范围，防止溢出
        progress_remaining = max(0.0, min(1.0, progress_remaining))

        # 计算新的熵系数
        new_ent_coef = self.schedule_func(progress_remaining)

        # [核心] 动态修改模型的 ent_coef
        self.model.ent_coef = new_ent_coef

        # 记录到 TensorBoard 方便监控
        self.logger.record("train/ent_coef", new_ent_coef)

        return True


def train(cfg: DictConfig, path_context: PathContext):
    env_config = cfg.environment
    seed = env_config.train_rl.seed

    mode = cfg.env_updates.mode
    env_config.env_settings.mode = mode
    scenario_mode = cfg.env_updates.scenario_mode
    env_config.env_settings.scenario_mode = scenario_mode

    env_config.env_settings.model_name = cfg.env_updates.model_name
    env_config.env_settings[scenario_mode][mode].active_scenario_list = cfg.env_updates[scenario_mode][
        mode].active_scenario_list

    # 创建环境
    env = DummyVecEnv([make_env(env_config, path_context, rank=0, seed=seed)])

    eval_env_config = env_config.copy()
    eval_env_config.env_settings.mode = 'evaluating'
    eval_env_config.env_settings[scenario_mode]['evaluating'].active_scenario_list = cfg.env_updates[scenario_mode][
        'evaluating'].active_scenario_list

    eval_env = DummyVecEnv([make_env(eval_env_config, path_context, rank=0, seed=seed + 1000)])
    rl_cfg = cfg.environment.train_rl

    net_arch = dict(pi=OmegaConf.to_container(rl_cfg.network.pi_head),
                    vf=OmegaConf.to_container(rl_cfg.network.vf_head))
    policy_kwargs = dict(features_extractor_class=None,
                         features_extractor_kwargs=dict(features_dim=rl_cfg.network.features_dim), net_arch=net_arch,
                         activation_fn=torch.nn.ReLU)

    # [关键修改] 使用 Linear Schedule
    base_lr = rl_cfg.ppo.learning_rate
    lr_schedule = linear_schedule(base_lr)
    base_clip = rl_cfg.ppo.clip_range
    clip_schedule_fn = clip_range_schedule(base_clip, final_value=0.05)
    base_ent_coef = rl_cfg.ppo.ent_coef
    ent_schedule = entropy_schedule(base_ent_coef)

    model = PPO(
        policy=HierarchicalSmartPolicy,
        env=env,
        learning_rate=lr_schedule,  # [应用衰减]
        n_steps=rl_cfg.ppo.n_steps,
        batch_size=rl_cfg.ppo.batch_size,
        n_epochs=rl_cfg.ppo.n_epochs,
        gamma=rl_cfg.ppo.gamma,
        gae_lambda=rl_cfg.ppo.gae_lambda,
        # clip_range=rl_cfg.ppo.clip_range,
        clip_range=clip_schedule_fn,
        ent_coef=rl_cfg.ppo.ent_coef,
        # ent_coef=ent_schedule,
        vf_coef=rl_cfg.ppo.vf_coef,
        max_grad_norm=rl_cfg.ppo.max_grad_norm,
        tensorboard_log=rl_cfg.logging.tensorboard_log,
        device=rl_cfg.device,
        verbose=1,
        policy_kwargs=policy_kwargs,
        seed=seed
    )

    print(f"✅ Model initialized with Hierarchical Attention Policy and Linear LR Schedule")

    callbacks = []
    if hasattr(rl_cfg, 'saving'):
        callbacks.append(CheckpointCallback(save_freq=rl_cfg.saving.save_freq, save_path=rl_cfg.saving.save_path,
                                            name_prefix="ppo_ha_best"))
    if hasattr(rl_cfg, 'evaluation'):
        callbacks.append(EvalCallback(eval_env, best_model_save_path=rl_cfg.evaluation.best_model_save_path,
                                      log_path=rl_cfg.logging.tensorboard_log, eval_freq=rl_cfg.evaluation.eval_freq,
                                      deterministic=True))

    callbacks.append(ConsoleLogCallback(print_freq=200))
    callbacks.append(PeriodicMetricsCallback(check_freq=1000))
    callbacks.append(HierarchicalDiagnosisCallback(check_freq=1000))

    # [新增] 黄金缓冲池 (Threshold 可根据实际情况调整，例如 -5.0)
    golden_dir = str(Path(rl_cfg.saving.save_path) / "golden_buffer")
    callbacks.append(GoldenBufferCallback(save_dir=golden_dir, reward_threshold=-5.0))

    # callbacks.append(EntropyScheduleCallback(
    #     schedule_func=ent_schedule,
    #     total_timesteps=rl_cfg.total_timesteps
    # ))

    model.learn(total_timesteps=rl_cfg.total_timesteps, callback=callbacks, progress_bar=True)

    if hasattr(rl_cfg, 'saving'):
        final_path = str(Path(rl_cfg.saving.save_path) / "final_model_ha_best")
        model.save(final_path)
        print(f"✅ Training finished. Model saved to {final_path}")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    config_path = "hierarchical_env.yaml"
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path);
        work_dir = os.getcwd();
        pm = PathContext(work_dir)
        train(cfg, pm)
    else:
        print(f"❌ Config file {config_path} not found.")