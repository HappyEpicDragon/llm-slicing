import os
import numpy as np
import pickle
import json
import shutil
import ray
from tqdm import tqdm
from omegaconf import OmegaConf
from collections import deque
import pandas as pd
import traceback

# Ray RLLib
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune import ExperimentAnalysis
from ray.rllib.utils.checkpoints import get_checkpoint_info

# Local Imports
# 注意：请确保这些路径在你的 PYTHONPATH 下
from src.basic_apis.network_slicing_business.path_context import PathContext
from src.basic_apis.ppo.ppo_baseline.env_ray import env_creator
from src.basic_apis.ppo.ppo_baseline.test_ppo_baseline import init_ray_and_register_env, register_only


# ==============================================================================
# 1. 辅助函数
# ==============================================================================

def get_best_checkpoint_path(experiment_path: str, metric_name: str = "evaluation/env_runners/policy_reward_mean/inter_slice_sched", mode: str = "max"):
    try:
        # [增强] 打印路径以便调试
        # print(f"🔍 Searching checkpoint in: {experiment_path}")
        analysis = ExperimentAnalysis(experiment_path)
        if not analysis.trials:
            print(f"⚠️ No trials found in {experiment_path}")
            return None

        trial = analysis.trials[0]
        trial_dataframe = analysis.trial_dataframes.get(trial.trial_id)
        if trial_dataframe is None or trial_dataframe.empty:
            print(f"⚠️ No dataframe for trial {trial.trial_id}")
            return None

        trial_dir = trial.path if hasattr(trial, 'path') else trial.local_path

        def _candidate_iterations(checkpoint_number):
            # 兼容旧/新两种保存策略：
            # - 旧口径：checkpoint_idx -> training_iteration * 3
            # - 新口径：checkpoint_idx -> training_iteration * 1
            return [(checkpoint_number + 1), (checkpoint_number + 1) * 3]

        available_checkpoints = {}
        if os.path.exists(trial_dir):
            for item in os.listdir(trial_dir):
                if item.startswith('checkpoint_'):
                    try:
                        cp_number = int(item.split('_')[-1])
                        available_checkpoints[cp_number] = item
                    except: continue

        best_checkpoint = None
        best_value = float('-inf') if mode == "max" else float('inf')

        for cp_number, cp_name in available_checkpoints.items():
            matching_rows = pd.DataFrame()
            for it in _candidate_iterations(cp_number):
                rows = trial_dataframe[trial_dataframe['training_iteration'] == it]
                if not rows.empty:
                    matching_rows = rows
                    break

            if matching_rows.empty:
                continue

            row = matching_rows.iloc[0]
            # 兼容当前仓库中的不同 metric 命名
            val = None
            possible_keys = [
                metric_name,
                "evaluation/env_runners/episode_return_mean",
                "env_runners/episode_return_mean",
                "evaluation/env_runners/episode_reward_mean",
                "env_runners/episode_reward_mean",
                "episode_reward_mean",
                "env_runners/policy_reward_mean/inter_slice_sched",
            ]

            for k in possible_keys:
                if k in row:
                    val = row[k]
                    break

            if val is not None and not pd.isna(val):
                if (mode == "max" and val > best_value) or (mode == "min" and val < best_value):
                    best_value = val
                    best_checkpoint = cp_name

        if best_checkpoint:
            return os.path.join(trial_dir, best_checkpoint)
        else:
            # 若指标列缺失或 trial 中断，回退到最新 checkpoint，避免数据采集全量空跑。
            if available_checkpoints:
                latest_cp_number = max(available_checkpoints.keys())
                latest_cp_name = available_checkpoints[latest_cp_number]
                print(f"⚠️ No valid checkpoint by metric. Fallback to latest: {latest_cp_name}")
                return os.path.join(trial_dir, latest_cp_name)
            print(f"⚠️ No valid checkpoint found after filtering.")

    except Exception as e:
        print(f"Error finding checkpoint: {e}")
        return None
    return None


def load_algorithm_cpu_safe(ckpt_path: str):
    """
    尝试从 checkpoint 恢复 RLlib Algorithm。
    若命中“Found 0 GPUs”错误，则回退到 CPU-safe 恢复：
    读取 algorithm_state.pkl 并将所有 GPU 相关配置清零后再 from_state。
    """
    def _patch_config_for_cpu_collect(config):
        patch_dict = {
            "num_gpus": 0,
            "num_gpus_per_worker": 0,
            "num_gpus_per_learner": 0,
            "num_gpus_per_env_runner": 0,
            "num_workers": 0,
            "num_rollout_workers": 0,
            "num_env_runners": 0,
            "evaluation_num_workers": 0,
            "evaluation_num_env_runners": 0,
            "evaluation_parallel_to_training": False,
        }

        # 字典配置（旧式）
        if isinstance(config, dict):
            for k, v in patch_dict.items():
                if k in config:
                    config[k] = v
            if isinstance(config.get("env_runners"), dict):
                config["env_runners"]["num_gpus_per_env_runner"] = 0
                config["env_runners"]["num_env_runners"] = 0
            if isinstance(config.get("learners"), dict):
                config["learners"]["num_gpus_per_learner"] = 0
            if isinstance(config.get("evaluation_config"), dict):
                eval_cfg = config["evaluation_config"]
                if "num_env_runners" in eval_cfg:
                    eval_cfg["num_env_runners"] = 0
                if "num_workers" in eval_cfg:
                    eval_cfg["num_workers"] = 0
            return config

        # AlgorithmConfig 对象（新式）
        # 注意：不要用 hasattr(config, "num_rollout_workers") 这类访问，
        # 某些废弃字段会在属性访问阶段直接抛 ValueError。
        modern_patch = {
            "num_gpus": 0,
            "num_gpus_per_env_runner": 0,
            "num_gpus_per_learner": 0,
            "num_env_runners": 0,
            "evaluation_num_env_runners": 0,
            "evaluation_parallel_to_training": False,
        }
        if hasattr(config, "update_from_dict"):
            try:
                config = config.update_from_dict(modern_patch)
            except Exception:
                pass

        # 优先使用官方 builder 方法，兼容 RLlib 新 API。
        try:
            resources_fn = getattr(config, "resources", None)
            if callable(resources_fn):
                config = resources_fn(num_gpus=0)
        except Exception:
            pass
        try:
            env_runners_fn = getattr(config, "env_runners", None)
            if callable(env_runners_fn):
                config = env_runners_fn(num_env_runners=0, num_gpus_per_env_runner=0)
        except Exception:
            pass
        try:
            learners_fn = getattr(config, "learners", None)
            if callable(learners_fn):
                config = learners_fn(num_gpus_per_learner=0)
        except Exception:
            pass
        try:
            evaluation_fn = getattr(config, "evaluation", None)
            if callable(evaluation_fn):
                config = evaluation_fn(
                    evaluation_parallel_to_training=False,
                    evaluation_num_env_runners=0,
                )
        except Exception:
            pass

        # 最后补一下嵌套 evaluation_config（若存在）。
        try:
            eval_cfg = getattr(config, "evaluation_config", None)
            if isinstance(eval_cfg, dict):
                if "num_env_runners" in eval_cfg:
                    eval_cfg["num_env_runners"] = 0
                if "num_workers" in eval_cfg:
                    eval_cfg["num_workers"] = 0
        except Exception:
            pass

        return config

    # 统一走 RLlib 官方 checkpoint -> state 还原流程，先拼完整 worker/policy_states，
    # 再补丁 config，最后 from_state。这样可稳定兼容不同 checkpoint 结构。
    try:
        checkpoint_info = get_checkpoint_info(ckpt_path)
        state = Algorithm._checkpoint_info_to_algorithm_state(checkpoint_info=checkpoint_info)
        state["config"] = _patch_config_for_cpu_collect(state.get("config"))
        return Algorithm.from_state(state)
    except Exception:
        raise

def flatten_marl_observation(obs_dict, agent_list):
    flat_obs = []
    if "player_0" in obs_dict:
        p0_obs = obs_dict["player_0"]
        if isinstance(p0_obs, dict) and "observations" in p0_obs:
            flat_obs.append(p0_obs["observations"])
        else:
            flat_obs.append(p0_obs)
    for i in range(1, 6):
        key = f"player_{i}"
        if key in obs_dict:
            p_obs = obs_dict[key]
            if isinstance(p_obs, dict) and "observations" in p_obs:
                flat_obs.append(p_obs["observations"])
            else:
                flat_obs.append(p_obs)
    return np.concatenate(flat_obs, axis=0).astype(np.float32)

def flatten_marl_action(action_dict):
    p0_act = action_dict.get("player_0", np.zeros(5, dtype=np.float32))
    disc_acts = []
    for i in range(1, 6):
        key = f"player_{i}"
        val = action_dict.get(key, 0)
        disc_acts.append(float(val))
    return np.concatenate([p0_act, np.array(disc_acts, dtype=np.float32)])


# ==============================================================================
# 2. Ray Remote Worker (Fix: Action Clipping)
# ==============================================================================

@ray.remote
def worker_ray_task(agent_id, scen_id, checkpoint_root, env_cfg_dict, pm_root,
                    mode, target_count, output_dir, deterministic_ratio):

    algo = None
    env = None
    try:
        # 1. 注册模型
        register_only()

        # 2. 获取 Checkpoint 路径
        ckpt_path = get_best_checkpoint_path(checkpoint_root, mode="max")
        if ckpt_path is None:
            return f"❌ Checkpoint not found in {checkpoint_root}"

        # 3. 加载 Algorithm（避免直接反序列化 policy checkpoint 的兼容性问题）
        algo = load_algorithm_cpu_safe(ckpt_path)

        # 4. 环境准备
        # env_cfg_dict 来自 cfg.environment (= single_scenario_env.yaml)，扁平结构，
        # 顶层直接有 mode / scenario_mode / inside / state_config / channel_class 等键。
        # 与 train_ppo_baseline.py 传给 env_creator 的结构一致。
        pm = PathContext(pm_root)
        env_config_dict = dict(env_cfg_dict)
        env_config_dict["mode"] = mode
        if "scenario_mode" not in env_config_dict:
            env_config_dict["scenario_mode"] = "inside"
        scenario_mode = env_config_dict["scenario_mode"]

        if scenario_mode not in env_config_dict or mode not in env_config_dict.get(scenario_mode, {}):
            return f"❌ Config missing: {scenario_mode}.{mode}"

        env_config_dict[scenario_mode][mode]["active_scenario_list"] = [scen_id]
        env_config_dict["path_context"] = pm

        base_seed = 1000 if mode == 'training' else 2024
        seed = base_seed + (scen_id * 1000) + (agent_id * 100)
        env_config_dict['seed'] = seed

        env = env_creator(env_config_dict)
        agent_list = env.agents

        # 5. 收集循环
        save_dir = os.path.join(output_dir, mode)
        os.makedirs(save_dir, exist_ok=True)

        obs, info = env.reset()
        if hasattr(env, 'comm_env'):
            current_phys_ep = env.comm_env.env_state.episode_number
        else:
            return "❌ Environment structure mismatch: no comm_env"

        curr_obs_flat = []
        curr_act_flat = []
        curr_rew = []
        curr_done = []
        curr_infos = []

        collected_cnt = 0

        # === [核心修复] Policy Inference Helper ===
        def compute_actions(observation):
            actions = {}
            for aid, aobs in observation.items():
                explore = (np.random.rand() > deterministic_ratio)
                policy_id = "inter_slice_sched" if aid == "player_0" else "intra_slice_sched"

                # 1. Input Prep (Clip Obs, 保持输入结构)
                if isinstance(aobs, dict) and 'observations' in aobs:
                    inp = aobs.copy()
                    inp['observations'] = np.clip(aobs['observations'], -1.0, 1.0)
                elif isinstance(aobs, np.ndarray):
                    inp = np.clip(aobs, -1.0, 1.0)
                else:
                    inp = aobs

                # 2. Inference via Algorithm API
                act_out = algo.compute_single_action(
                    inp,
                    policy_id=policy_id,
                    explore=explore,
                )
                act = act_out[0] if isinstance(act_out, tuple) else act_out

                # 3. [Fix] Inter continuous action clipping
                # Baseline PPO 使用 Gaussian 分布，可能采样出 < -1.0 的值
                # 必须手动 Clip，否则 scores_to_rbs 会计算出负权重
                if aid == "player_0" and isinstance(act, np.ndarray):
                    act = np.clip(act, -1.0, 1.0)

                actions[aid] = act

            return actions

        while collected_cnt < target_count:
            # A. 计算动作
            action_dict = compute_actions(obs)

            # B. 环境步进
            next_obs, rewards, terminated, truncated, infos = env.step(action_dict)
            done_flag = terminated["__all__"] or truncated["__all__"]

            # C. 记录
            flat_o = flatten_marl_observation(obs, agent_list)
            curr_obs_flat.append(flat_o)

            flat_a = flatten_marl_action(action_dict)
            curr_act_flat.append(flat_a)

            r = rewards.get("player_0", 0.0)
            curr_rew.append(r)
            curr_done.append(done_flag)
            curr_infos.append(infos.get("player_0", {}))

            obs = next_obs

            if done_flag:
                ep_return = sum(curr_rew)

                hp_viols = 0
                for inf in curr_infos:
                    for k, v in inf.items():
                        if k.startswith("violation/slice_") and v > 0:
                            try:
                                s_idx = int(k.split('_')[1])
                                prio = inf.get(f"meta/slice_{s_idx}_priority", 0)
                                if prio > 0: hp_viols += 1
                            except: pass

                traj_data = {
                    "observations": np.array(curr_obs_flat, dtype=np.float32),
                    "actions": np.array(curr_act_flat, dtype=np.float32),
                    "rewards": np.array(curr_rew, dtype=np.float32),
                    "dones": np.array(curr_done, dtype=bool),
                    "episode_returns": ep_return,
                    "episode_length": len(curr_act_flat),
                    "hp_viols": hp_viols,
                    "meta_agent": agent_id,
                    "meta_scen": scen_id,
                    "meta_phys_ep": current_phys_ep
                }

                pkl_name = f"S{scen_id}_A{agent_id}_ep_{collected_cnt}_ret_{ep_return:.0f}.pkl"
                with open(os.path.join(save_dir, pkl_name), "wb") as f:
                    pickle.dump(traj_data, f)

                collected_cnt += 1

                curr_obs_flat = []
                curr_act_flat = []
                curr_rew = []
                curr_done = []
                curr_infos = []

                obs, info = env.reset()
                current_phys_ep = env.comm_env.env_state.episode_number

    except Exception as e:
        err_msg = traceback.format_exc()
        return f"❌ CRITICAL WORKER ERROR:\n{err_msg}"
    finally:
        # 显式释放资源，避免任务结束后 actor 残留占用 CPU。
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            if algo is not None and hasattr(algo, "stop"):
                algo.stop()
        except Exception:
            pass

    return f"✅ Agent {agent_id} Scen {scen_id} Done."


# ==============================================================================
# 3. Collector Class
# ==============================================================================

class RobustCollectorBaseline:
    def __init__(self, cfg, path_context):
        self.cfg = cfg
        self.path_context = path_context

        dt_cfg = cfg.dt_dataset_collect if 'dt_dataset_collect' in cfg else cfg
        self.output_dir = dt_cfg.output_path
        self.episodes_per_scenario_training = dt_cfg.episodes_per_scenario_training
        self.episodes_per_scenario_evaluating = dt_cfg.episodes_per_scenario_evaluating
        self.runs_per_episode = dt_cfg.runs_per_episode
        self.deterministic_ratio = dt_cfg.deterministic_ratio

        if hasattr(dt_cfg, 'MODEL_ZOO'):
            self.MODEL_ZOO = OmegaConf.to_container(dt_cfg.MODEL_ZOO, resolve=True)
        else:
            self.MODEL_ZOO = {}

        if hasattr(dt_cfg, 'CROSS_SCENARIO_IDS'):
            self.CROSS_SCENARIO_IDS = OmegaConf.to_container(dt_cfg.CROSS_SCENARIO_IDS, resolve=True)
        else:
            self.CROSS_SCENARIO_IDS = [0]

        for subdir in ["training", "evaluating", "discarded_training", "discarded_evaluating"]:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            register_only()

    def collect_parallel(self):
        tasks = []
        if hasattr(self.cfg, 'environment'):
            env_cfg_dict = OmegaConf.to_container(self.cfg.environment, resolve=True)
        else:
            env_cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
        pm_root = self.path_context.hydra_workdir

        # 限制每个 worker 任务的 CPU 配额，避免一次并发过高引发 OOM。
        # 可在 yaml 中通过 dt_dataset_collect_baseline.ray_cpus_per_task 覆盖。
        ray_cpus_per_task = int(self.cfg.get("ray_cpus_per_task", 8))

        print(f"🚀 [Baseline] Preparing tasks for {len(self.MODEL_ZOO)} Agents...")
        print(f"🧠 Ray task CPU quota: {ray_cpus_per_task} (lower concurrency, better memory stability)")
        for agent_id, checkpoint_path in self.MODEL_ZOO.items():
            if not os.path.exists(checkpoint_path):
                print(f"❌ Checkpoint path not found: {checkpoint_path}")
                continue
            for scen_id in self.CROSS_SCENARIO_IDS:
                target_train = self.episodes_per_scenario_training * self.runs_per_episode
                tasks.append(worker_ray_task.options(num_cpus=ray_cpus_per_task).remote(
                    agent_id, scen_id, checkpoint_path, env_cfg_dict, pm_root,
                    'training', target_train, self.output_dir, self.deterministic_ratio
                ))
                target_eval = self.episodes_per_scenario_evaluating * self.runs_per_episode
                tasks.append(worker_ray_task.options(num_cpus=ray_cpus_per_task).remote(
                    agent_id, scen_id, checkpoint_path, env_cfg_dict, pm_root,
                    'evaluating', target_eval, self.output_dir, self.deterministic_ratio
                ))

        print(f"🔥 Submitted {len(tasks)} Ray tasks. Waiting for completion...")
        pending = tasks
        pbar = tqdm(total=len(tasks), desc="Collecting")

        # [修改] 增加结果检查逻辑
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            pbar.update(len(done))
            for ref in done:
                try:
                    res = ray.get(ref)
                    # 检查是否包含错误标记
                    if isinstance(res, str) and "❌" in res:
                        tqdm.write(f"\n{res}\n") # 使用 tqdm.write 防止进度条错乱
                except Exception as e:
                    tqdm.write(f"❌ Task Failed with Exception: {e}")

        pbar.close()
        print("✅ All collection tasks finished.")

    def filter_best_of_n(self, mode, keep_count=40):
        print(f"\n🧹 Filtering {mode} dataset (Keep Top {keep_count})...")
        target_dir = os.path.join(self.output_dir, mode)
        discard_dir = os.path.join(self.output_dir, f"discarded_{mode}")
        if not os.path.exists(target_dir): return
        files = [f for f in os.listdir(target_dir) if f.endswith('.pkl')]
        group_map = {}
        for f_name in tqdm(files, desc="Indexing"):
            path = os.path.join(target_dir, f_name)
            try:
                with open(path, "rb") as f:
                    traj = pickle.load(f)
                scen = traj.get('meta_scen')
                phys_ep = traj.get('meta_phys_ep')
                hp = traj.get('hp_viols', 999)
                rew = traj['episode_returns']
                sort_key = (hp, -rew)
                if scen is not None and phys_ep is not None:
                    key = (scen, phys_ep)
                    if key not in group_map: group_map[key] = []
                    group_map[key].append({'filename': f_name, 'path': path, 'sort_key': sort_key})
            except: pass
        total_kept = 0
        for key, items in group_map.items():
            items.sort(key=lambda x: x['sort_key'])
            keep = items[:keep_count]
            drop = items[keep_count:]
            total_kept += len(keep)
            for item in drop:
                try:
                    shutil.move(item['path'], os.path.join(discard_dir, item['filename']))
                except: pass
        print(f"✅ Filter {mode} Done. Total kept: {total_kept}")

    def compute_metadata(self):
        print("\n📊 Computing Metadata (Flat Obs)...")
        train_dir = os.path.join(self.output_dir, "training")
        if not os.path.exists(train_dir): return
        files = [f for f in os.listdir(train_dir) if f.endswith('.pkl')]
        if not files:
            print("⚠️ No training data found for metadata.")
            return

        with open(os.path.join(train_dir, files[0]), "rb") as f:
            sample = pickle.load(f)
            obs_dim = sample['observations'].shape[1]
        print(f"   -> Detected Obs Dim: {obs_dim}")
        sum_obs = np.zeros(obs_dim)
        sq_sum_obs = np.zeros(obs_dim)
        count = 0
        for f_name in tqdm(files, desc="Scanning"):
            path = os.path.join(train_dir, f_name)
            try:
                with open(path, "rb") as f:
                    traj = pickle.load(f)
                obs = traj['observations']
                sum_obs += np.sum(obs, axis=0)
                sq_sum_obs += np.sum(obs ** 2, axis=0)
                count += obs.shape[0]
            except: pass

        if count == 0:
            print("⚠️ Count is 0, skipping.")
            return

        mean = sum_obs / count
        var = (sq_sum_obs / count) - (mean ** 2)
        std = np.sqrt(np.maximum(var, 1e-6))
        meta = {"obs_stats": {"flat": {"mean": mean.tolist(), "std": std.tolist()}}, "info": "Baseline Flattened Obs"}
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(meta, f)
        print("✅ Metadata saved.")


# ==============================================================================
# 4. Entry Point
# ==============================================================================

def _filter_single_teacher_per_scenario(output_dir, mode):
    """对每个场景只保留最优教师（best-of-N 后占比最高的），其余移到 discarded 目录。"""
    target_dir = os.path.join(output_dir, mode)
    discard_dir = os.path.join(output_dir, f"discarded_{mode}")
    if not os.path.exists(target_dir):
        return

    files = [f for f in os.listdir(target_dir) if f.endswith('.pkl')]
    from collections import Counter, defaultdict
    scen_agent_counts = defaultdict(Counter)
    file_meta = {}
    for f_name in files:
        path = os.path.join(target_dir, f_name)
        with open(path, "rb") as f:
            traj = pickle.load(f)
        scen = traj.get('meta_scen')
        agent = traj.get('meta_agent')
        scen_agent_counts[scen][agent] += 1
        file_meta[f_name] = (scen, agent)

    best_teacher = {}
    for scen, counts in sorted(scen_agent_counts.items()):
        best_agent = counts.most_common(1)[0][0]
        best_teacher[scen] = best_agent

    kept = removed = 0
    for f_name, (scen, agent) in file_meta.items():
        if scen in best_teacher and agent == best_teacher[scen]:
            kept += 1
        else:
            src = os.path.join(target_dir, f_name)
            shutil.move(src, os.path.join(discard_dir, f_name))
            removed += 1

    print(f"  [{mode}] Single-teacher filter: kept={kept}, removed={removed}")
    print(f"  [{mode}] Best teacher per scenario: {best_teacher}")


def _merge_eval_to_train(output_dir):
    """将 evaluating 目录的 pkl 文件合并到 training 目录。"""
    eval_dir = os.path.join(output_dir, 'evaluating')
    train_dir = os.path.join(output_dir, 'training')
    if not os.path.exists(eval_dir):
        return
    eval_files = [f for f in os.listdir(eval_dir) if f.endswith('.pkl')]
    moved = 0
    for f in eval_files:
        src = os.path.join(eval_dir, f)
        dst = os.path.join(train_dir, f)
        if os.path.exists(dst):
            dst = os.path.join(train_dir, 'eval_' + f)
        shutil.move(src, dst)
        moved += 1
    print(f"  Merged {moved} evaluating files into training")


def collect(cfg, path_context):
    collector = RobustCollectorBaseline(cfg, path_context)

    # 1. Collect
    collector.collect_parallel()
    keep_count_per_episode = cfg.keep_count_per_episode

    # 2. Filter best-of-N per episode
    collector.filter_best_of_n('training', keep_count=keep_count_per_episode)
    collector.filter_best_of_n('evaluating', keep_count=keep_count_per_episode)

    # 3. Single-teacher filter: keep only dominant teacher per scenario
    print("\n🔬 Filtering to single best teacher per scenario...")
    _filter_single_teacher_per_scenario(collector.output_dir, 'training')
    _filter_single_teacher_per_scenario(collector.output_dir, 'evaluating')

    # 4. Merge evaluating into training
    print("\n📦 Merging evaluating into training...")
    _merge_eval_to_train(collector.output_dir)

    # 5. Metadata (based on final training data)
    collector.compute_metadata()

    ray.shutdown()