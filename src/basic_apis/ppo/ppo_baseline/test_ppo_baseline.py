import ray
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
import numpy as np
import os
from tqdm import tqdm
from ray.tune import ExperimentAnalysis
import pandas as pd
from hydra.utils import get_class
from omegaconf import OmegaConf
from pathlib import Path

from src.basic_apis.ppo.ppo_baseline.ray_agent import TorchActionMaskModel, TorchDiagGaussian
from src.basic_apis.ppo.ppo_baseline.env_ray import env_creator


class PhysicsProbeBaseline:
    def __init__(self):
        self.records = []

    def capture(self, env, action, step, info=None):
        # 1. 获取组件容器 (适配 MARLCommEnv)
        if hasattr(env, 'comm_env'):
            # 直接从 MARLCommEnv 获取内部的 CommunicationEnv
            comps = env.comm_env.components
        elif hasattr(env, 'components'):
            comps = env.components
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'components'):
            comps = env.unwrapped.components

        # 2. 动作解析
        # Baseline 的 Action 结构比较复杂，通常是 [Inter-slice weights, Intra-slice algos]
        # 但最终的物理分配体现在 comps.ues.step() 之后的某些状态，或者我们只能看结果
        # 为了对比，我们主要看 comps.ues 的状态和 env.step_executor 生成的 sched_decision

        # 注意：Baseline 的 step_executor 在执行 step 时并没有把 sched_decision 存为 public 属性
        # 但 metrics 里有！
        metrics_hist = comps.metrics.metrics_hist
        if not metrics_hist['sched_decision']:
            return  # 还没数据

        # 获取最新一帧的数据
        # sched_decision 形状通常是 [Num_BS, Num_Users, Num_RBs]
        # Baseline 默认单基站，所以是 [1, 25, 135]
        sched_decision = metrics_hist['sched_decision'][-1]
        if sched_decision.ndim == 3:
            sched_decision = sched_decision[0]  # [25, 135]

        # 计算激活的 RBG 数量
        # 注意：Baseline 是基于 RB 的 (135)，我们需要映射回 RBG (27) 以便对比
        # 简单的做法是看有多少 RB 被分配了，然后除以 5
        active_rbs_count = np.sum(sched_decision > 0)
        active_rbgs_est = active_rbs_count / 5.0  # 估算值

        # 功率因子
        # Baseline 通常不直接输出 power_scale，它是隐含在 SINR 计算里的
        # 假设 Baseline 是满功率发射，scale_factor = 1.0?
        # 或者我们需要看它是否使用了 Power Scaling。
        # 查看 Baseline 代码 QuadrigaChannel.step_origin:
        # power_factor = self.transmission_power / self.num_available_rbs[0]
        # 这意味着 Baseline 是 **固定功率分配** (Equal Power per RB)，没有动态 Scaling！
        # 这是一个巨大的差异点！
        power_scale = 1.0  # Baseline 是静态的

        # 3. 用户级详情
        buf = metrics_hist['buffer_occupancies'][-1]
        # Baseline 似乎没有直接存 pkt_incoming_bits，只有 pkt_incoming (pkts)
        incoming = metrics_hist['pkt_incoming'][-1]

        # 识别有需求的用户
        demand_mask = (buf > 1e-9) | (incoming > 0)
        total_demand_users = np.sum(demand_mask)

        # 识别被服务的用户
        # sched_decision [25, 135], sum over RBs -> [25]
        user_alloc_sum = np.sum(sched_decision, axis=1)
        served_count = np.sum(user_alloc_sum > 0)

        # 4. 信道质量分析 (Spectral Efficiency)
        # metrics_hist['spectral_efficiencies'] -> [1, 25, 135] (Log2(1+SNR))
        se_matrix = metrics_hist['spectral_efficiencies'][-1]
        if se_matrix.ndim == 3: se_matrix = se_matrix[0]

        # 计算 **被分配资源** 的平均 SE
        # 只统计 sched_decision > 0 的位置的 SE
        mask = sched_decision > 0
        if np.sum(mask) > 0:
            avg_se = np.mean(se_matrix[mask])
        else:
            avg_se = 0.0

        # 5. 记录一帧的数据
        self.records.append({
            "step": step,
            "active_rbgs": active_rbgs_est,
            "power_scale": power_scale,  # 静态
            "demand_users": total_demand_users,
            "served_users": served_count,
            "avg_se": avg_se,
            # Baseline 的 violation 需要自己算或者从外部传，这里先留空
            "hp_violations": 0,
            "nhp_violations": 0
        })

    def report(self):
        """输出统计报告"""
        df = pd.DataFrame(self.records)
        if df.empty: return "No Data"

        print("\n" + "=" * 30 + " BASELINE PHYSICS REPORT " + "=" * 30)
        print(f"1. Power Strategy (Static):")
        print(f"   - Avg Active RBGs (Est): {df['active_rbgs'].mean():.2f} / 27")
        print(f"   - Power Mode: Equal Power per RB (Static)")

        print(f"2. Efficiency:")
        print(f"   - Avg Spectral Eff: {df['avg_se'].mean():.4f} bits/Hz")

        print(f"3. Service Saturation:")
        saturation = (df['served_users'] / (df['demand_users'] + 1e-6)).mean()
        print(f"   - Service Saturation: {saturation:.2%}")

        print("=" * 86 + "\n")


def test_ppo_baseline(cfg, path_manager):
    """使用加载的模型进行测试，不依赖 algo 的 workers"""

    # 1. 准备环境配置
    env_config = cfg.environment
    updates = cfg.env_updates
    env_config = OmegaConf.to_container(env_config, resolve=True)
    env_config['mode'] = updates['mode']
    scenario_mode = updates['scenario_mode']
    env_config['scenario_mode'] = scenario_mode
    env_config['model_name'] = updates['model_name']
    env_config[scenario_mode]['testing']['active_scenario_list'] = updates[scenario_mode]['testing'][
        'active_scenario_list']
    env_config["path_manager"] = path_manager

    checkpoint_path = cfg.checkpoint_path

    try:
        init_ray_and_register_env()
        # init_ray_and_register_env(env_config)

        # 2. 加载 checkpoint
        print(f"📦 正在加载checkpoint: {checkpoint_path}")
        algo = load_best_checkpoint(
            experiment_path=os.path.abspath(checkpoint_path),
            metric_name=cfg.metric_name,
            mode=cfg.mode
        )

        if not algo:
            raise RuntimeError("❌ 加载Algorithm失败")

        # ⭐ 3. 直接创建测试环境（不使用 algo 的 workers）
        test_env = env_creator(env_config)

        # 获取 agent 列表
        agent_list = None
        if hasattr(test_env, 'agents'):
            agent_list = test_env.agents
        elif hasattr(test_env, 'possible_agents'):
            agent_list = test_env.possible_agents

        # 4. 运行测试 episodes
        scenario_mode = env_config['scenario_mode']
        current_scenario = env_config[scenario_mode]['testing']['active_scenario_list'][0]
        scenario_skip_episodes = env_config[scenario_mode]['scenario_skip_episodes']
        initial_ep = env_config[scenario_mode]['testing']['initial_episode'] + current_scenario * scenario_skip_episodes
        test_episodes = env_config[scenario_mode]['testing']['test_episodes']

        print(f"🧪 开始测试 {test_episodes} 个episodes...")


        for episode in tqdm(
                range(initial_ep, initial_ep + test_episodes),
                desc="Running PPO Test Episodes"
        ):
            options = {"initial_episode": initial_ep} if episode == initial_ep else None
            # obs = test_env.reset(options=options)
            # obs = _process_observation(obs, agent_list)

            reset_result = test_env.reset(options=options)
            if isinstance(reset_result, tuple):
                obs, info = reset_result  # Gymnasium 新 API
            else:
                obs = reset_result  # 旧 API

            obs = _process_observation(obs, agent_list)

            if obs is None:
                tqdm.write(f"⚠️ Episode {episode}: 无法处理观察格式")
                continue

            done = False
            episode_length = 0
            max_length = 10000
            probe = PhysicsProbeBaseline()
            step = 0

            while not done and episode_length < max_length:
                # actions = {}
                # for agent_id, agent_obs in obs.items():
                #     # ⭐ 使用 algo.compute_single_action（不需要 workers）
                #     action = algo.compute_single_action(
                #         agent_obs,
                #         policy_id=agent_id,
                #         explore=False  # 测试时不探索
                #     )
                #     actions[agent_id] = action
                #
                # result = test_env.step(actions)

                actions = {}
                for agent_id, agent_obs in obs.items():
                    # ⭐ 使用 _compute_action 函数，它会自动映射 policy_id
                    actions[agent_id] = _compute_action(algo, agent_id, agent_obs)

                # 环境 step
                result = test_env.step(actions)

                obs, dones, episode_length = _process_step_result(
                    result, agent_list, episode_length
                )

                if obs is None:
                    tqdm.write(f"⚠️ Episode {episode}: step返回格式异常")
                    break

                done = _check_done(dones)
                probe.capture(test_env, actions, step)
                if done:
                    step = 0
                    probe.report()

            if episode_length >= max_length:
                tqdm.write(f"⚠️ Episode {episode} 超过最大长度限制")

        print("✅ 测试完成!")
        return True

    except Exception as e:
        tqdm.write(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shut_down_ray_and_env(test_env)


def _process_observation(obs, agent_list):
    """处理环境返回的观察"""
    if isinstance(obs, dict):
        return obs

    if isinstance(obs, tuple):
        # 提取字典格式的观察
        if len(obs) > 0 and isinstance(obs[0], dict):
            return obs[0]
        # 从tuple转换为字典
        elif agent_list and len(obs) == len(agent_list):
            return {agent: obs[i] for i, agent in enumerate(agent_list)}

    return None


# def _compute_action(algo, agent_id: str, observation):
#     """为指定agent计算动作"""
#     try:
#         # 确定policy_id
#         if agent_id.startswith("player_"):
#             agent_idx = int(agent_id.split("_")[-1])
#         else:
#             agent_idx = int(agent_id.split("_")[-1])
#
#         policy_id = "inter_slice_sched" if agent_idx == 0 else "intra_slice_sched"
#
#         # 使用PPO推理
#         action = algo.compute_single_action(
#             observation,
#             policy_id=policy_id,
#             explore=False
#         )
#         return action
#
#     except Exception as e:
#         tqdm.write(f"⚠️ 推理失败 ({agent_id}): {e}, 使用随机动作")
#         return np.random.uniform(-1, 1, size=2).astype(np.float32)

def _compute_action(algo, agent_id: str, observation):
    """为指定agent计算动作"""
    try:
        # 确定policy_id (保持原逻辑不变)
        if agent_id.startswith("player_"):
            agent_idx = int(agent_id.split("_")[-1])
        else:
            agent_idx = int(agent_id.split("_")[-1])

        policy_id = "inter_slice_sched" if agent_idx == 0 else "intra_slice_sched"

        # =======================================================
        # 🛠️ [修复代码开始] 强制裁剪观测值范围
        # =======================================================
        # 检查 observation 是否包含 'observations' 键 (根据你的报错日志结构)
        if isinstance(observation, dict):
            if 'observations' in observation:
                # 你的报错显示越界值为 -1.679，模型下限是 -1.0
                # 强制将数据限制在 [-1.0, 1.0] 之间，防止模型报错
                observation['observations'] = np.clip(
                    observation['observations'], -1.0, 1.0
                )
        elif isinstance(observation, np.ndarray):
            observation = np.clip(observation, -1.0, 1.0)
        # =======================================================
        # 🛠️ [修复代码结束]
        # =======================================================

        # 使用PPO推理
        action = algo.compute_single_action(
            observation,
            policy_id=policy_id,
            explore=False
        )
        return action

    except Exception as e:
        # 打印详细错误，方便确认是否修复
        import traceback
        tqdm.write(f"⚠️ 推理失败 ({agent_id}): {e}")
        # traceback.print_exc()

        # ⚠️ 注意：这里的 size=2 是导致 IndexError 的直接原因
        # 如果还是报错，说明上面的 clip 没生效，或者模型确实不兼容
        return np.random.uniform(-1, 1, size=2).astype(np.float32)


def _process_step_result(result, agent_list, current_length):
    """处理环境step的返回结果"""
    # 解析返回值
    if len(result) == 4:
        obs, rewards, dones, info = result
    elif len(result) == 5:
        obs, rewards, dones, truncated, info = result
        # 合并dones和truncated
        if isinstance(dones, dict) and isinstance(truncated, dict):
            for k in dones:
                dones[k] = dones[k] or truncated.get(k, False)
        else:
            dones = dones or truncated
    else:
        return None, None, current_length

    # 处理观察格式
    obs = _process_observation(obs, agent_list)

    return obs, dones, current_length + 1


def _check_done(dones):
    """检查episode是否结束"""
    if isinstance(dones, dict):
        return all(dones.values()) or dones.get("__all__", False)
    return bool(dones)

# def init_ray_and_register_env():
#     if ray.is_initialized():
#         ray.shutdown()
#     ray.init()
#     ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)
#     ModelCatalog.register_custom_action_dist("masked_gaussian", TorchDiagGaussian)
#     ray.tune.registry.register_env("marl_comm_env", env_creator)

def register_only():
    """仅注册自定义模型和环境，不初始化 Ray（供 Ray 已由外部初始化时调用）。"""
    ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)
    ModelCatalog.register_custom_action_dist("masked_gaussian", TorchDiagGaussian)
    ray.tune.registry.register_env("marl_comm_env", env_creator)


def init_ray_and_register_env(env_config=None):
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)
    ModelCatalog.register_custom_action_dist("masked_gaussian", TorchDiagGaussian)

    # ⭐ 如果提供了 env_config，使用闭包捕获它
    if env_config is not None:
        ray.tune.registry.register_env(
            "marl_comm_env",
            # lambda config: env_creator({**env_config, **config})
            lambda config: env_creator({**config, **env_config})
        )
    else:
        ray.tune.registry.register_env("marl_comm_env", env_creator)


def shut_down_ray_and_env(env):
    if env:
        env.close()
    if ray.is_initialized():
        ray.shutdown()


def safely_get_nested_value(data: dict, path: str, default=None):
    """安全地获取嵌套字典中的值"""
    try:
        keys = path.split('/')
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    except:
        return default


def load_best_checkpoint(
        experiment_path: str,
        metric_name: str = "evaluation/env_runners/policy_reward_max/inter_slice_sched",
        mode: str = "max"
):
    """
    使用正确的映射关系加载最佳checkpoint
    每3个iteration保存一个checkpoint
    """
    try:
        analysis = ExperimentAnalysis(experiment_path)
        trial = analysis.trials[0]
        trial_dataframe = analysis.trial_dataframes.get(trial.trial_id)
        trial_dir = trial.path if hasattr(trial, 'path') else trial.local_path

        # if trial_dataframe is None:
        #     print("❌ 无法获取训练历史数据")
        #     return None

        # 建立正确的映射关系: checkpoint_number → iteration
        # checkpoint_N → iteration (N+1)*3
        def checkpoint_to_iteration(checkpoint_number):
            # return (checkpoint_number + 1) * 3
            return (checkpoint_number + 1) * 1 # 最后相乘的数字是save_freq，1代表每训练1k步保存一次checkpoint

        # 获取所有可用checkpoints
        available_checkpoints = {}
        if os.path.exists(trial_dir):
            for item in os.listdir(trial_dir):
                if item.startswith('checkpoint_'):
                    try:
                        cp_number = int(item.split('_')[-1])
                        available_checkpoints[cp_number] = item
                    except:
                        continue

        print(f"📁 找到 {len(available_checkpoints)} 个checkpoints (000000-{max(available_checkpoints.keys()):06d})")

        # 在有checkpoint的iterations中找最佳metric
        best_checkpoint = None
        best_value = float('-inf') if mode == "max" else float('inf')
        best_iteration = None
        best_cp_number = None

        for cp_number, cp_name in available_checkpoints.items():
            corresponding_iteration = checkpoint_to_iteration(cp_number)

            # 在训练历史中查找这个iteration的metric值
            matching_rows = trial_dataframe[
                trial_dataframe['training_iteration'] == corresponding_iteration
                ]

            if not matching_rows.empty:
                row = matching_rows.iloc[0]

                if metric_name in trial_dataframe.columns:
                    metric_value = row[metric_name]
                else:
                    row_dict = row.to_dict()
                    metric_value = safely_get_nested_value(row_dict, metric_name)

                if metric_value is not None and not pd.isna(metric_value):
                    is_better = (mode == "max" and metric_value > best_value) or \
                                (mode == "min" and metric_value < best_value)

                    if is_better and cp_number >= 5:
                        best_value = metric_value
                        best_iteration = corresponding_iteration
                        best_checkpoint = cp_name
                        best_cp_number = cp_number

        if best_checkpoint:
            checkpoint_path = os.path.join(trial_dir, best_checkpoint)

            policies_dir = os.path.join(checkpoint_path, "policies")
            if os.path.exists(policies_dir):
                policy_names = os.listdir(policies_dir)
                print(f"✅ Checkpoint 中的 Policy 名称: {policy_names}")

            print(f"✅ 找到最佳checkpoint (正确映射):")
            print(f"   📊 {metric_name}: {best_value}")
            print(f"   🔢 Training iteration: {best_iteration}")
            print(f"   📁 Checkpoint: {best_checkpoint} (#{best_cp_number:06d})")
            print(f"   📂 路径: {checkpoint_path}")

            # 加载Algorithm
            agent = Algorithm.from_checkpoint(checkpoint_path)
            print(f"✅ 成功加载 {type(agent).__name__} agent!")
            return agent
        else:
            print("❌ 未找到有效的checkpoint")
            return None

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None


def load_specific_checkpoint_by_index(experiment_path, index):
    """根据索引加载特定的 Checkpoint"""
    try:
        # 1. 找到 Trial 目录
        analysis = ExperimentAnalysis(experiment_path)
        if not analysis.trials:
            print("❌ No trials found in analysis.")
            return None
        trial = analysis.trials[0]
        trial_dir = trial.path if hasattr(trial, 'path') else trial.local_path

        # 2. 构造文件夹名称 checkpoint_000000
        checkpoint_name = f"checkpoint_{index:06d}"
        checkpoint_path = os.path.join(trial_dir, checkpoint_name)

        if os.path.exists(checkpoint_path):
            print(f"📦 Loading Specific Checkpoint: {checkpoint_name}")
            return Algorithm.from_checkpoint(checkpoint_path)
        else:
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            return None
    except Exception as e:
        print(f"❌ Load Specific ({index}) Failed: {e}")
        return None


def _run_evaluation_episodes(env_config, algo, test_episodes, desc="Testing"):
    """
    通用测试循环：给环境和算法，跑 N 个 episode，返回统计结果。
    返回 dict 包含：mean_return, hp_violation_rate, nhp_violation_rate, n_episodes
    """
    test_env = None
    try:
        test_env = env_creator(env_config)
        agent_list = getattr(test_env, 'agents', getattr(test_env, 'possible_agents', None))

        metrics_aggregator = []

        scenario_mode = env_config['scenario_mode']
        current_scenario = env_config[scenario_mode]['testing']['active_scenario_list'][0]
        scenario_skip_episodes = env_config[scenario_mode]['scenario_skip_episodes']
        initial_ep = env_config[scenario_mode]['testing']['initial_episode'] + current_scenario * scenario_skip_episodes

        for episode in tqdm(range(initial_ep, initial_ep + test_episodes), desc=desc, leave=False):
            options = {"initial_episode": initial_ep} if episode == initial_ep else None

            reset_result = test_env.reset(options=options)
            if isinstance(reset_result, tuple):
                obs, _ = reset_result
            else:
                obs = reset_result

            obs = _process_observation(obs, agent_list)
            if obs is None: continue

            done = False
            episode_length = 0
            max_length = 10000
            ep_reward = 0.0
            ep_hp_viol_sum = ep_nhp_viol_sum = 0
            ep_hp_active = ep_nhp_active = 0
            step = 0

            while not done and episode_length < max_length:
                actions = {}
                for agent_id, agent_obs in obs.items():
                    actions[agent_id] = _compute_action(algo, agent_id, agent_obs)

                result = test_env.step(actions)

                # 提取 info（在 _process_step_result 丢弃之前）
                raw_info = {}
                if len(result) == 4:
                    _, rewards_raw, _, raw_info = result
                elif len(result) == 5:
                    _, rewards_raw, _, _, raw_info = result
                else:
                    rewards_raw = {}

                if isinstance(rewards_raw, dict):
                    ep_reward += sum(rewards_raw.values())
                else:
                    ep_reward += float(rewards_raw)

                # 逐步累计 violation（与 metric_value.py 对齐）
                _, _, hp_v, nhp_v, hp_a, nhp_a = _compute_step_metrics(raw_info)
                ep_hp_viol_sum += hp_v
                ep_nhp_viol_sum += nhp_v
                ep_hp_active += hp_a
                ep_nhp_active += nhp_a

                obs, dones, episode_length = _process_step_result(result, agent_list, episode_length)
                if obs is None: break

                done = _check_done(dones)
                step += 1

                if done:
                    hp_viol_rate  = ep_hp_viol_sum  / ep_hp_active  if ep_hp_active  > 0 else 0.0
                    nhp_viol_rate = ep_nhp_viol_sum / ep_nhp_active if ep_nhp_active > 0 else 0.0
                    metrics_aggregator.append({
                        'episode': episode,
                        'episode_return': ep_reward,
                        'hp_violation_rate': hp_viol_rate,
                        'nhp_violation_rate': nhp_viol_rate,
                        'episode_length': episode_length,
                    })

    except Exception as e:
        tqdm.write(f"❌ Evaluation Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if test_env:
            test_env.close()

    if metrics_aggregator:
        df = pd.DataFrame(metrics_aggregator)
        return df.mean(numeric_only=True).to_dict()
    return {}


def test_ppo_baseline_finetune(cfg, path_manager):
    """
    新的 Finetune 测试入口
    功能：加载 test_ckp_idx 列表中的 checkpoint 以及 Best Checkpoint，进行对比测试。
    """
    # 1. 准备环境配置
    env_config = cfg.environment
    updates = cfg.env_updates
    env_config = OmegaConf.to_container(env_config, resolve=True)

    # 应用 env_updates
    env_config['mode'] = updates['mode']
    scenario_mode = updates['scenario_mode']
    env_config['scenario_mode'] = scenario_mode

    # 设置 Scenario List
    for phase in ['training', 'testing', 'evaluating']:
        if phase in updates[scenario_mode]:
            env_config[scenario_mode][phase]['active_scenario_list'] = \
                updates[scenario_mode][phase]['active_scenario_list']

    env_config["path_manager"] = path_manager

    # 2. 初始化 Ray
    checkpoint_path = cfg.checkpoint_path
    try:
        init_ray_and_register_env(env_config)

        # 3. 定义要测试的任务列表
        # 结构: {"name": str, "algo_loader": function}
        tasks = []

        # (A) 添加指定 Index 的 Checkpoints
        if hasattr(cfg, 'test_ckp_idx') and cfg.test_ckp_idx:
            for idx in cfg.test_ckp_idx:
                tasks.append({
                    "name": f"CKP_{idx}",
                    "ckp_index": int(idx),
                    "loader": lambda p=checkpoint_path, i=idx: load_specific_checkpoint_by_index(p, i)
                })

        # (B) 添加 Best Checkpoint
        tasks.append({
            "name": "Best_Model",
            "ckp_index": -1,
            "loader": lambda p=checkpoint_path: load_best_checkpoint(
                os.path.abspath(p), cfg.metric_name, cfg.mode
            )
        })

        results_table = []
        test_episodes = env_config[scenario_mode]['testing'].get('test_episodes', 5)

        print(f"\n🚀 Start Finetune Testing: {len(tasks)} models to evaluate.")
        print(f"   Test Scenario: {env_config[scenario_mode]['testing']['active_scenario_list']}")
        print(f"   Episodes per model: {test_episodes}\n")

        # 4. 循环评估
        for task_idx, task in enumerate(tasks):
            print(f"\n👉 Evaluating Model: {task['name']}")

            algo = task['loader']()
            if algo is None:
                print(f"   ⚠️ Skipping {task['name']} (Load Failed)")
                continue

            env_config['model_name'] = task['name']

            metrics = _run_evaluation_episodes(
                env_config,
                algo,
                test_episodes,
                desc=f"Testing {task['name']}"
            )

            if metrics:
                # 解析 checkpoint index（从名称中提取步数）
                ckp_index = task.get('ckp_index', -1)
                row = {
                    "ckp_name": task['name'],
                    "ckp_index": ckp_index,
                    "mean_return": metrics.get('episode_return', 0.0),
                    "hp_violation_rate": metrics.get('hp_violation_rate', 0.0),
                    "nhp_violation_rate": metrics.get('nhp_violation_rate', 0.0),
                    "episode_length": metrics.get('episode_length', 0.0),
                }
                results_table.append(row)
                print(f"   Return={row['mean_return']:.2f}, HP_Viol={row['hp_violation_rate']:.4f}")

            if hasattr(algo, 'stop'):
                algo.stop()
            del algo

        # 保存 CSV 结果
        if results_table:
            results_df = pd.DataFrame(results_table)
            target_scen = env_config[scenario_mode]['testing']['active_scenario_list'][0]
            save_dir = os.path.join(
                cfg.get('save_root', 'data/channel_generality/transfer_ppo'),
                f"scenario_{target_scen}"
            )
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, "finetune_adaptation_curve.csv")
            results_df.to_csv(csv_path, index=False)
            print(f"\n✅ Results saved to: {csv_path}")
            print(results_df.to_string(index=False))

        return True

    except Exception as e:
        print(f"❌ Finetune Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if ray.is_initialized():
            ray.shutdown()


# =============================================================================
# Multi-scenario PPO Baseline Testing
# =============================================================================

def _compute_step_metrics(info: dict, num_slices: int = 5):
    """
    与 metric_value.calc_intent_distance / calc_slice_violations 对齐的单步指标计算：
    - Distance: 每个活跃 slice 仅计入违约（负值）drift 的最差指标（min），HP/NHP 分别求和
    - Violation: 每个活跃 slice 整体判断（mean(drifts) < 0 → 1次违约），HP/NHP 分别计数
    支持 agent-keyed dict（自动展开取第一个含 drift/ 键的 agent info）。
    返回: (hp_dist, nhp_dist, hp_viols, nhp_viols, hp_active, nhp_active)
    """
    flat_info = info
    if info and not any(k.startswith("drift/") or k.startswith("meta/") for k in info):
        for v in info.values():
            if isinstance(v, dict) and any(k.startswith("drift/") or k.startswith("meta/") for k in v):
                flat_info = v
                break

    hp_dist = nhp_dist = 0.0
    hp_viols = nhp_viols = hp_active = nhp_active = 0
    for s_idx in range(num_slices):
        is_hp = flat_info.get(f"meta/slice_{s_idx}_priority", 0) > 0
        drifts = [float(flat_info.get(f"drift/slice_{s_idx}_{met}", 0.0))
                  for met in ("thr", "rel", "lat")
                  if flat_info.get(f"meta/slice_{s_idx}_{met}_req", 0) > 0]
        if not drifts:
            continue
        if is_hp:
            hp_active += 1
        else:
            nhp_active += 1
        mean_d = float(np.mean(drifts))
        if mean_d < 0:
            if is_hp:
                hp_dist += mean_d
                hp_viols += 1
            else:
                nhp_dist += mean_d
                nhp_viols += 1
    return hp_dist, nhp_dist, hp_viols, nhp_viols, hp_active, nhp_active


def _run_scenario_test(env_config: dict, algo, test_episodes: int, scenario_id: int) -> dict:
    """
    在单个场景上运行 test_episodes 个 episode，返回统计指标：
    - mean_episode_return
    - hp_violation_rate
    - nhp_violation_rate
    - n_episodes
    """
    test_env = None
    try:
        test_env = env_creator(env_config)
        agent_list = getattr(test_env, "agents", getattr(test_env, "possible_agents", None))

        scenario_mode = env_config["scenario_mode"]
        scenario_skip = env_config[scenario_mode]["scenario_skip_episodes"]
        initial_ep = env_config[scenario_mode]["testing"]["initial_episode"] + scenario_id * scenario_skip

        episode_returns = []
        ep_hp_dists = []
        ep_nhp_dists = []
        ep_hp_viols = []
        ep_nhp_viols = []
        step_hp_dists_all = []
        step_nhp_dists_all = []

        for ep_offset in tqdm(range(test_episodes), desc=f"  Scenario {scenario_id}", leave=False):
            options = {"initial_episode": initial_ep} if ep_offset == 0 else None
            reset_result = test_env.reset(options=options)
            obs, _ = reset_result if isinstance(reset_result, tuple) else (reset_result, {})
            obs = _process_observation(obs, agent_list)
            if obs is None:
                continue

            done = False
            ep_length = ep_reward = 0.0
            ep_hp_dist_sum = ep_nhp_dist_sum = 0.0
            ep_hp_viol_sum = ep_nhp_viol_sum = 0
            ep_hp_active = ep_nhp_active = 0
            step_hp_dists = []
            step_nhp_dists = []

            while not done and ep_length < 10000:
                actions = {aid: _compute_action(algo, aid, aobs) for aid, aobs in obs.items()}
                result = test_env.step(actions)

                if len(result) == 5:
                    obs_new, rewards, dones, truncated, info = result
                    if isinstance(dones, dict) and isinstance(truncated, dict):
                        for k in dones:
                            dones[k] = dones[k] or truncated.get(k, False)
                elif len(result) == 4:
                    obs_new, rewards, dones, info = result
                else:
                    break

                ep_reward += sum(rewards.values()) if isinstance(rewards, dict) else float(rewards)

                # 逐步累计 distance 和 violation（与 metric_value.py 对齐）
                step_info = info if isinstance(info, dict) else {}
                hp_d, nhp_d, hp_v, nhp_v, hp_a, nhp_a = _compute_step_metrics(step_info)
                ep_hp_dist_sum += hp_d
                ep_nhp_dist_sum += nhp_d
                ep_hp_viol_sum += hp_v
                ep_nhp_viol_sum += nhp_v
                ep_hp_active += hp_a
                ep_nhp_active += nhp_a
                step_hp_dists.append(hp_d / hp_a if hp_a > 0 else 0.0)
                step_nhp_dists.append(nhp_d / nhp_a if nhp_a > 0 else 0.0)

                obs = _process_observation(obs_new, agent_list)
                ep_length += 1
                done = _check_done(dones)

            episode_returns.append(ep_reward)
            hp_dist_ep  = ep_hp_dist_sum  / ep_hp_active  if ep_hp_active  > 0 else 0.0
            nhp_dist_ep = ep_nhp_dist_sum / ep_nhp_active if ep_nhp_active > 0 else 0.0
            hp_viol_ep  = ep_hp_viol_sum  / ep_hp_active  if ep_hp_active  > 0 else 0.0
            nhp_viol_ep = ep_nhp_viol_sum / ep_nhp_active if ep_nhp_active > 0 else 0.0
            ep_hp_dists.append(float(hp_dist_ep))
            ep_nhp_dists.append(float(nhp_dist_ep))
            ep_hp_viols.append(float(hp_viol_ep))
            ep_nhp_viols.append(float(nhp_viol_ep))
            step_hp_dists_all.extend(step_hp_dists)
            step_nhp_dists_all.extend(step_nhp_dists)

        return {
            "scenario_id": scenario_id,
            "mean_episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "hp_violation_rate": float(np.mean(ep_hp_viols)) if ep_hp_viols else float("nan"),
            "nhp_violation_rate": float(np.mean(ep_nhp_viols)) if ep_nhp_viols else float("nan"),
            "hp_dist_mean": float(np.mean(ep_hp_dists)) if ep_hp_dists else float("nan"),
            "nhp_dist_mean": float(np.mean(ep_nhp_dists)) if ep_nhp_dists else float("nan"),
            "ep_hp_dists": ep_hp_dists,
            "ep_nhp_dists": ep_nhp_dists,
            "ep_hp_viols": ep_hp_viols,
            "ep_nhp_viols": ep_nhp_viols,
            "ep_rewards": episode_returns,
            "step_hp_dists": step_hp_dists_all,
            "step_nhp_dists": step_nhp_dists_all,
            "n_episodes": len(episode_returns),
        }

    except Exception as e:
        tqdm.write(f"❌ Scenario {scenario_id} test error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "scenario_id": scenario_id,
            "mean_episode_return": float("nan"),
            "hp_violation_rate": float("nan"),
            "nhp_violation_rate": float("nan"),
            "n_episodes": 0,
        }
    finally:
        if test_env:
            test_env.close()


def test_ppo_baseline_multi(cfg, path_manager):
    """
    Multi-scenario PPO Baseline 评估入口（含 seed 循环）。

    流程：
    1. 加载一次 checkpoint（multi-scenario PPO 模型）
    2. 在每个场景 × 每个 seed 上跑 test_episodes 个 episode
    3. 汇总 mean ± std，保存 CSV 和标准 JSON（供绘图脚本读取）
    """
    import copy
    import json

    env_config_base = OmegaConf.to_container(cfg.environment, resolve=True)
    env_config_base["path_manager"] = path_manager

    checkpoint_path = cfg.checkpoint_path
    test_scenarios = list(cfg.env_updates.inside.testing.active_scenario_list)
    test_episodes = int(cfg.test_episodes)
    scenario_mode = cfg.env_updates.scenario_mode
    test_seeds = list(cfg.get('test_seeds', [42]))
    save_root = str(cfg.get('save_root', 'data/channel_generality/ppo_multi'))
    clean_before_save = bool(cfg.get('clean_before_save', False))

    print(f"\n{'='*65}")
    print(f"  Multi-scenario PPO Baseline Test")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Scenarios  : {test_scenarios}")
    print(f"  Seeds      : {test_seeds}")
    print(f"  Episodes   : {test_episodes} per scenario per seed")
    print(f"{'='*65}\n")

    def _save_json_file(data, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    try:
        if clean_before_save:
            for subdir in ("metric_json", "metric_raw"):
                old_dir = os.path.join(save_root, subdir)
                if os.path.exists(old_dir):
                    import shutil
                    shutil.rmtree(old_dir)
            print(f"[Asset] cleaned old PPO-multi metrics under: {save_root}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"PPO-multi checkpoint_path not found: {checkpoint_path}")

        init_ray_and_register_env()

        print(f"Loading checkpoint: {checkpoint_path}")
        algo = load_best_checkpoint(
            experiment_path=os.path.abspath(checkpoint_path),
            metric_name=cfg.metric_name,
            mode=cfg.mode,
        )
        if algo is None:
            raise RuntimeError("Failed to load checkpoint — check checkpoint_path in YAML")

        all_results = []
        scenario_summaries = {}

        for scenario_id in test_scenarios:
            seed_returns = []
            seed_hp_viols = []
            seed_nhp_viols = []
            seed_hp_dists = []
            seed_nhp_dists = []

            for seed in test_seeds:
                np.random.seed(seed)

                env_cfg = copy.deepcopy(env_config_base)
                env_cfg["mode"] = "testing"
                env_cfg["scenario_mode"] = scenario_mode
                env_cfg["model_name"] = f"scenario_{scenario_id}_seed{seed}"
                env_cfg[scenario_mode]["testing"]["active_scenario_list"] = [scenario_id]

                result = _run_scenario_test(env_cfg, algo, test_episodes, scenario_id)

                seed_returns.append(result['mean_episode_return'])
                seed_hp_viols.append(result['hp_violation_rate'])
                seed_nhp_viols.append(result['nhp_violation_rate'])
                hp_dist = result.get('hp_dist_mean', float('nan'))
                nhp_dist = result.get('nhp_dist_mean', float('nan'))
                seed_hp_dists.append(hp_dist)
                seed_nhp_dists.append(nhp_dist)

                row = {**result, "seed": seed}
                all_results.append(row)

                print(f"  Scenario {scenario_id} Seed {seed}: "
                      f"Return={result['mean_episode_return']:.2f} | "
                      f"HP_viol={result['hp_violation_rate']:.4f} | "
                      f"HP_Dist={hp_dist:.4f}")

                # 保存 seed 级别 JSON（标准格式）
                json_dir = os.path.join(save_root, "metric_json", f"scenario_{scenario_id}", f"seed_{seed}")
                _save_json_file(
                    {"hp_violations": result.get('ep_hp_viols', []), "mean": result['hp_violation_rate']},
                    os.path.join(json_dir, "hp_violations.json")
                )
                _save_json_file(
                    {"nhp_violations": result.get('ep_nhp_viols', []), "mean": result['nhp_violation_rate']},
                    os.path.join(json_dir, "nhp_violations.json")
                )
                _save_json_file(
                    {"rewards": result.get('ep_rewards', []), "mean": result['mean_episode_return']},
                    os.path.join(json_dir, "episode_rewards.json")
                )
                ep_hp_dists = result.get('ep_hp_dists', [])
                ep_nhp_dists = result.get('ep_nhp_dists', [])
                step_hp_dists = result.get('step_hp_dists', [])
                step_nhp_dists = result.get('step_nhp_dists', [])
                _save_json_file(
                    {"hp_distance": ep_hp_dists, "mean": hp_dist},
                    os.path.join(json_dir, "hp_distance.json")
                )
                _save_json_file(
                    {"nhp_distance": ep_nhp_dists, "mean": nhp_dist},
                    os.path.join(json_dir, "nhp_distance.json")
                )

                # step-level NPZ（ppo_baseline 没有 step-level rewards，只保存 distance 均值作为占位）
                raw_dir = os.path.join(save_root, "metric_raw", f"scenario_{scenario_id}")
                os.makedirs(raw_dir, exist_ok=True)
                np.savez(
                    os.path.join(raw_dir, f"ep_seed{seed}.npz"),
                    ep_rewards=np.array(result.get('ep_rewards', []), dtype=np.float64),
                    hp_viols=np.array(result.get('ep_hp_viols', []), dtype=np.float64),
                    nhp_viols=np.array(result.get('ep_nhp_viols', []), dtype=np.float64),
                    step_hp_dist=np.array(step_hp_dists, dtype=np.float64) if step_hp_dists else np.array(ep_hp_dists, dtype=np.float64),
                    step_nhp_dist=np.array(step_nhp_dists, dtype=np.float64) if step_nhp_dists else np.array(ep_nhp_dists, dtype=np.float64),
                )

            # 场景汇总
            valid_hp_dists = [d for d in seed_hp_dists if not np.isnan(d)]
            valid_nhp_dists = [d for d in seed_nhp_dists if not np.isnan(d)]
            scenario_summary = {
                "scenario_id": scenario_id,
                "seeds": test_seeds,
                "reward_mean": float(np.mean(seed_returns)),
                "reward_std": float(np.std(seed_returns)),
                "hp_viol_mean": float(np.mean(seed_hp_viols)),
                "hp_viol_std": float(np.std(seed_hp_viols)),
                "nhp_viol_mean": float(np.mean(seed_nhp_viols)),
                "nhp_viol_std": float(np.std(seed_nhp_viols)),
                "hp_dist_mean": float(np.mean(valid_hp_dists)) if valid_hp_dists else float("nan"),
                "hp_dist_std": float(np.std(valid_hp_dists)) if valid_hp_dists else float("nan"),
                "nhp_dist_mean": float(np.mean(valid_nhp_dists)) if valid_nhp_dists else float("nan"),
                "nhp_dist_std": float(np.std(valid_nhp_dists)) if valid_nhp_dists else float("nan"),
            }
            scenario_summaries[f"scenario_{scenario_id}"] = scenario_summary

            summary_path = os.path.join(save_root, "metric_json", f"scenario_{scenario_id}", "summary.json")
            _save_json_file(scenario_summary, summary_path)

            print(f"  Scenario {scenario_id} Summary: "
                  f"Return={scenario_summary['reward_mean']:.2f}±{scenario_summary['reward_std']:.2f} | "
                  f"HP_Viol={scenario_summary['hp_viol_mean']:.4f}±{scenario_summary['hp_viol_std']:.4f} | "
                  f"HP_Dist={scenario_summary['hp_dist_mean']:.4f}")

        # 打印汇总表
        print(f"\n{'='*65}")
        print(f"  SUMMARY: Multi-scenario PPO (mean±std over {len(test_seeds)} seeds)")
        print(f"{'='*65}")
        for scen_key, s in scenario_summaries.items():
            print(f"  {scen_key}: Return={s['reward_mean']:.2f}±{s['reward_std']:.2f} "
                  f"HP={s['hp_viol_mean']:.4f}±{s['hp_viol_std']:.4f}")

        # 保存全局 CSV（含 seed 列）
        output_dir = path_manager.get_save_metrics_dir_path()
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "multi_scenario_ppo_results.csv")
        df = pd.DataFrame(all_results)
        df.to_csv(results_path, index=False)
        print(f"\n✅ Results saved to: {results_path}")

        # 保存全局 JSON
        global_json = {
            "method": "ppo_multi",
            "test_seeds": test_seeds,
            "scenarios": scenario_summaries,
        }
        _save_json_file(global_json, os.path.join(save_root, "metric_json", "global_summary.json"))

        if hasattr(algo, "stop"):
            algo.stop()

        return True

    except Exception as e:
        print(f"❌ Multi-scenario test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if ray.is_initialized():
            ray.shutdown()

