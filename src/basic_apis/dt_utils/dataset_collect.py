import os
import numpy as np
import torch
import pickle
import json
import shutil
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from omegaconf import OmegaConf

# === Local Imports ===
from src.basic_apis.ppo.ppo_ha.hierarchical_slicing_env import HierarchicalSlicingEnv
from src.basic_apis.ppo.ppo_ha.agent_hierarchical import HierarchicalSmartPolicy
from src.basic_apis.network_slicing_business.path_manager import PathManager


class RobustCollector:
    def __init__(self, cfg, path_manager, model_path, output_dir):
        self.cfg = cfg
        self.path_manager = path_manager
        self.model_path = model_path
        self.output_dir = output_dir

        # 创建输出目录结构
        os.makedirs(os.path.join(output_dir, "training"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "evaluating"), exist_ok=True)

        print(f"🔄 Loading Teacher from: {model_path}")
        # 临时环境用于加载模型
        temp_env = DummyVecEnv([self._make_env_fn(mode='training', seed=0)])
        try:
            self.model = PPO.load(
                model_path,
                env=temp_env,
                device='cpu',
                custom_objects={
                    "learning_rate": 0.0,
                    "clip_range": 0.1,
                    "HierarchicalSmartPolicy": HierarchicalSmartPolicy
                }
            )
        except Exception as e:
            print(f"⚠️ Custom load failed, trying standard: {e}")
            self.model = PPO.load(model_path, env=temp_env, device='cpu')
        temp_env.close()

    def _make_env_fn(self, mode, seed=0):
        def _init():
            env_config = self.cfg.env_settings.copy()
            env_config.mode = mode

            # 确保 scenario list 正确
            scenario_mode = env_config.scenario_mode
            target_cfg = env_config[scenario_mode][mode]
            if target_cfg.active_scenario_list is None:
                target_cfg.active_scenario_list = [0]

            env = HierarchicalSlicingEnv(env_config, np.random.default_rng(seed), self.path_manager)
            return env

        return _init

    def collect(self, mode, target_episodes, min_return_threshold=-300000.0):
        """
        收集数据主循环 (混合策略 + 单文件存储)
        """
        print(f"\n🚀 Collecting {target_episodes} episodes for [{mode}]...")
        save_dir = os.path.join(self.output_dir, mode)

        # 使用不同的种子保证流量多样性
        # training 用随机种子，evaluating 用固定种子
        base_seed = 1000 if mode == 'training' else 2024
        env = DummyVecEnv([self._make_env_fn(mode=mode, seed=base_seed)])

        obs = env.reset()

        curr_obs, curr_act, curr_rew, curr_done = [], [], [], []
        curr_info = []  # 用于统计元数据

        collected_count = 0
        pbar = tqdm(total=target_episodes, desc=f"Collecting {mode}")

        while collected_count < target_episodes:
            # === [核心改进 1] 混合策略 ===
            # 80% 概率使用确定性 (专家)
            # 20% 概率使用随机 (增加多样性/纠错)
            use_deterministic = (np.random.rand() < 0.8)

            action, _ = self.model.predict(obs, deterministic=use_deterministic)

            # 执行
            next_obs, rewards, dones, infos = env.step(action)
            info = infos[0]

            # === [核心改进 2] 数据解包与存储 ===
            # 解包 Obs: [1, F] -> [F]
            step_obs = {k: v[0].copy() for k, v in obs.items()}

            # 存储 Raw Action Logits (30-dim)
            curr_act.append(action[0].copy())
            curr_obs.append(step_obs)
            curr_rew.append(rewards[0])
            curr_done.append(dones[0])
            curr_info.append(info)

            obs = next_obs

            if dones[0]:
                ep_return = sum(curr_rew)

                # === [核心改进 3] 筛选与保存 ===
                # 统计 HP 违约
                hp_viols = 0
                for inf in curr_info:
                    for k, v in inf.items():
                        if k.startswith("violation/slice_") and v > 0:
                            s_idx = int(k.split('_')[1])
                            prio = inf.get(f"meta/slice_{s_idx}_priority", 0)
                            if prio > 0: hp_viols += 1

                # 筛选条件：回报不能太离谱 (过滤掉 NaN/Inf 导致的 -inf)
                # 即使有 HP 违约也保留 (作为负面教材)，除非你只想要完美数据
                # 这里我们放宽条件，只过滤彻底崩盘的局
                if ep_return > min_return_threshold:
                    traj_data = {
                        "observations": curr_obs,  # List of Dicts
                        "actions": np.array(curr_act, dtype=np.float32),
                        "rewards": np.array(curr_rew, dtype=np.float32),
                        "dones": np.array(curr_done, dtype=bool),
                        "episode_returns": ep_return,
                        "episode_length": len(curr_act),
                        "hp_viols": hp_viols,
                        "is_stochastic": not use_deterministic
                    }

                    # 单文件保存
                    pkl_name = f"ep_{collected_count}_ret_{ep_return:.0f}.pkl"
                    with open(os.path.join(save_dir, pkl_name), "wb") as f:
                        pickle.dump(traj_data, f)

                    collected_count += 1
                    pbar.update(1)
                    pbar.set_postfix({"Ret": f"{ep_return:.0f}", "HP": hp_viols})

                # 重置缓存
                curr_obs, curr_act, curr_rew, curr_done, curr_info = [], [], [], [], []
                # obs 自动 reset

        env.close()
        pbar.close()

    def compute_metadata(self):
        """
        [核心改进 4] 计算全局元数据 (Mean/Std/Percentiles)
        """
        print("\n📊 Computing Global Metadata...")
        train_dir = os.path.join(self.output_dir, "training")

        stats = {
            "inter": {"sum": 0, "sq_sum": 0, "count": 0},
            "intra": {"sum": 0, "sq_sum": 0, "count": 0},
            "global": {"sum": 0, "sq_sum": 0, "count": 0}
        }
        all_returns = []

        files = [f for f in os.listdir(train_dir) if f.endswith('.pkl')]
        print(f"Scanning {len(files)} trajectories...")

        for f_name in tqdm(files):
            with open(os.path.join(train_dir, f_name), "rb") as f:
                traj = pickle.load(f)

            all_returns.append(traj["episode_returns"])

            # 聚合 Obs 统计
            # Obs is List[Dict]
            # Convert list of dicts to dict of arrays for faster processing
            inter = np.array([o['inter_feat'] for o in traj['observations']])  # [T, 5, 4]
            intra = np.array([o['intra_feat'] for o in traj['observations']])  # [T, 25, 5]
            glob = np.array([o['global_feat'] for o in traj['observations']])  # [T, 2]

            # Flatten & Accumulate
            # Inter
            flat_inter = inter.reshape(-1, 4)
            stats["inter"]["sum"] += flat_inter.sum(axis=0)
            stats["inter"]["sq_sum"] += (flat_inter ** 2).sum(axis=0)
            stats["inter"]["count"] += flat_inter.shape[0]

            # Intra
            flat_intra = intra.reshape(-1, 5)
            stats["intra"]["sum"] += flat_intra.sum(axis=0)
            stats["intra"]["sq_sum"] += (flat_intra ** 2).sum(axis=0)
            stats["intra"]["count"] += flat_intra.shape[0]

            # Global
            flat_glob = glob.reshape(-1, 2)
            stats["global"]["sum"] += flat_glob.sum(axis=0)
            stats["global"]["sq_sum"] += (flat_glob ** 2).sum(axis=0)
            stats["global"]["count"] += flat_glob.shape[0]

        # Calculate Mean/Std
        metadata = {"obs_stats": {}, "return_stats": {}}

        for k in stats:
            N = stats[k]["count"]
            mean = stats[k]["sum"] / N
            var = (stats[k]["sq_sum"] / N) - (mean ** 2)
            std = np.sqrt(np.maximum(var, 1e-6))

            metadata["obs_stats"][k] = {
                "mean": mean.tolist(),
                "std": std.tolist()
            }

        # Return Percentiles
        metadata["return_stats"] = {
            "p50": float(np.percentile(all_returns, 50)),
            "p90": float(np.percentile(all_returns, 90)),
            "p95": float(np.percentile(all_returns, 95)),
            "max": float(np.max(all_returns)),
            "min": float(np.min(all_returns))
        }

        save_path = os.path.join(self.output_dir, "metadata.json")
        with open(save_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"✅ Metadata saved to {save_path}")
        print(f"   Target Suggestion (P90): {metadata['return_stats']['p90']:.2f}")



def collect(cfg, path_manager):
    env_cfg = cfg.environment
    env_update = cfg.env_updates
    env_cfg.env_settings.model_name = env_update.model_name
    scenario_mode = env_update.scenario_mode
    env_cfg.env_settings.scenario_mode = scenario_mode
    env_cfg.env_settings[scenario_mode].training.active_scenario_list = env_update[scenario_mode].training.active_scenario_list


    eval_env_cfg = env_cfg.copy()
    eval_env_cfg.env_settings.mode = 'evaluating'
    eval_env_cfg.env_settings[scenario_mode].evaluating.active_scenario_list = env_update[scenario_mode].evaluating.active_scenario_list

    model_path = cfg.model_path
    output_dir = cfg.output_path

    collector = RobustCollector(env_cfg, path_manager, model_path, output_dir)

    # 1. 收集训练集 (扩充到 2000 条以覆盖更多随机性)
    collector.collect(mode='training', target_episodes=2000)

    # 2. 收集验证集
    collector.collect(mode='evaluating', target_episodes=200)

    # 3. 计算元数据
    collector.compute_metadata()



if __name__ == "__main__":
    # config_path = "hierarchical_env.yaml"
    # model_path = "/root/decision_transformer_slicing/data/channel_generality/ppo_ha/models/scenario_0/final_model_ha_best.zip"
    # output_dir = "./data/dt_dataset_robust_v1"
    #
    # if os.path.exists(config_path):
    #     cfg = OmegaConf.load(config_path)
    #     pm = PathManager(os.getcwd())
    #
    #     collector = RobustCollector(cfg, pm, model_path, output_dir)
    #
    #     # 1. 收集训练集 (扩充到 2000 条以覆盖更多随机性)
    #     collector.collect(mode='training', target_episodes=2000)
    #
    #     # 2. 收集验证集
    #     collector.collect(mode='evaluating', target_episodes=200)
    #
    #     # 3. 计算元数据
    #     collector.compute_metadata()
    #
    # else:
    #     print("Config not found.")
    pass