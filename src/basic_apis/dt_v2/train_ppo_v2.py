"""
PPO teacher training on EnvV2 (no urgency_rank, richer obs, violation reward).

Standalone script — does not modify any original code.

Usage:
    cd /root/decision_transformer_slicing
    .pixi/envs/default/bin/python -m src.basic_apis.dt_v2.train_ppo_v2
"""
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from src.basic_apis.dt_v2.env_v2 import HierarchicalSlicingEnvV2
from src.basic_apis.ppo.ppo_ha_weighted.agent_hierarchical import (
    HierarchicalSmartPolicy,
    HierarchicalMLPPolicy,
    HierarchicalPooledAttentionPolicy,
    HierarchicalSliceAttnPolicy,
)
from src.basic_apis.network_slicing_business.path_manager import PathManager


def make_env_v2(env_settings, path_manager, rank=0, seed=0, episode_offset=0):
    """Create a single EnvV2 factory.

    episode_offset staggers each subprocess's starting episode so that within
    one rollout buffer different envs observe different channel files.
    The offset is applied before the first reset(); the env still wraps
    cyclically over [init_ep, max_ep) as usual.
    """
    def _init():
        env = HierarchicalSlicingEnvV2(env_settings, np.random.default_rng(seed + rank), path_manager)
        if episode_offset > 0:
            n_eps = env.max_ep - env.init_ep
            env.internal_episode_ptr = env.init_ep + (episode_offset % n_eps)
        env = Monitor(env)
        return env
    return _init


def train_v2(
    scenarios=(0, 1, 2, 3, 4),
    total_timesteps=600_000,
    save_dir="data/channel_generality/dt_v2/ppo_teacher",
    seed=42,
    device="cuda",
    extractor="mlp",
    ent_coef=0.01,
    n_steps=512,
    n_envs=1,
):
    env_cfg = OmegaConf.load("conf/environment/env_ha.yaml")
    pm = PathManager(os.getcwd())

    # Training env
    train_settings = env_cfg.env_settings.copy()
    train_settings.mode = "training"
    train_settings.scenario_mode = "inside"
    train_settings.model_name = "v2_teacher"
    train_settings.inside.training.active_scenario_list = list(scenarios)

    # Eval env  (uses "evaluating" mode → ep 80-99, held-out from training)
    eval_settings = env_cfg.env_settings.copy()
    eval_settings.mode = "evaluating"
    eval_settings.scenario_mode = "inside"
    eval_settings.model_name = "v2_teacher"
    eval_settings.inside.evaluating.active_scenario_list = list(scenarios)

    # Stagger episode start across envs so each rollout buffer sees diverse
    # channel files.  env_i starts at ep (init + i * total_eps // n_envs).
    n_train_eps = (train_settings.inside.training.max_scenario_episodes
                   - train_settings.inside.training.init_scenario_episode)
    train_fns = [
        make_env_v2(train_settings, pm, rank=i, seed=seed + i,
                    episode_offset=(i * n_train_eps // n_envs) if n_envs > 1 else 0)
        for i in range(n_envs)
    ]
    env = SubprocVecEnv(train_fns) if n_envs > 1 else DummyVecEnv(train_fns)
    eval_env = DummyVecEnv([make_env_v2(eval_settings, pm, rank=0, seed=seed + 1000)])

    os.makedirs(save_dir, exist_ok=True)
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    best_dir = os.path.join(save_dir, "best_model")
    log_dir = os.path.join(save_dir, "logs")

    policy_map = {
        "mlp": HierarchicalMLPPolicy,
        "attention": HierarchicalSmartPolicy,
        "pooled_attention": HierarchicalPooledAttentionPolicy,
        "slice_attn": HierarchicalSliceAttnPolicy,
    }
    policy_cls = policy_map[extractor]

    features_dim = 256
    net_arch = dict(pi=[256, 128], vf=[256, 128])
    policy_kwargs = dict(
        features_extractor_class=None,
        features_extractor_kwargs=dict(features_dim=features_dim),
        net_arch=net_arch,
        activation_fn=torch.nn.ReLU,
    )

    model = PPO(
        policy=policy_cls,
        env=env,
        learning_rate=3e-5,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        device=device,
        verbose=1,
        policy_kwargs=policy_kwargs,
        seed=seed,
    )

    callbacks = [
        CheckpointCallback(save_freq=10000, save_path=ckpt_dir, name_prefix="ppo_v2"),
        EvalCallback(
            eval_env,
            best_model_save_path=best_dir,
            log_path=log_dir,
            eval_freq=10000,
            deterministic=True,
        ),
    ]

    print(f"Training PPO teacher on EnvV2 (extractor={extractor})")
    print(f"  Scenarios: {scenarios}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  n_envs: {n_envs}, ent_coef: {ent_coef}, n_steps: {n_steps}")
    print(f"  Save dir: {save_dir}")
    print(f"  Action space: {env.action_space}")
    print(f"  Obs space: inter={env.observation_space['inter_feat'].shape}, "
          f"intra={env.observation_space['intra_feat'].shape}")

    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"Training done. Model saved to {final_path}")

    env.close()
    eval_env.close()


def train(cfg, path_manager):
    """Hydra entry point: called by channel_generality.py → train_ppo_v2 mode.

    Reads cfg.train_ppo_v2 and delegates to train_v2().
    path_manager is accepted for API consistency but unused (train_v2 creates
    its own from os.getcwd()).
    """
    tc = cfg.train_ppo_v2
    train_v2(
        scenarios=tuple(tc.scenarios),
        total_timesteps=int(tc.ppo.total_timesteps),
        save_dir=str(tc.save_dir),
        seed=int(tc.ppo.seed),
        device=str(tc.ppo.device),
        extractor=str(tc.extractor),
        ent_coef=float(tc.ppo.ent_coef),
        n_steps=int(tc.ppo.n_steps),
        n_envs=int(tc.ppo.n_envs),
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--scenarios", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="data/channel_generality/dt_v2/ppo_teacher")
    parser.add_argument(
        "--extractor",
        type=str,
        choices=["mlp", "attention", "pooled_attention", "slice_attn"],
        default="mlp",
    )
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--n_envs", type=int, default=1,
                        help="Parallel envs via SubprocVecEnv (requires npz channel files)")
    args = parser.parse_args()

    train_v2(
        scenarios=tuple(args.scenarios),
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        seed=args.seed,
        device=args.device,
        extractor=args.extractor,
        ent_coef=args.ent_coef,
        n_steps=args.n_steps,
        n_envs=args.n_envs,
    )
