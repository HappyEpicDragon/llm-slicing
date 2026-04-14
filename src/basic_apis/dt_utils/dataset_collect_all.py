import os
import numpy as np
import torch
import pickle
import json
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from omegaconf import OmegaConf

# === Local Imports ===
from src.basic_apis.ppo.ppo_ha_weighted.hierarchical_slicing_env import HierarchicalSlicingEnv
from src.basic_apis.ppo.ppo_ha_weighted.agent_hierarchical import HierarchicalSmartPolicy
from src.basic_apis.network_slicing_business.path_manager import PathManager


# =========================================================
# 独立的 Worker 函数 (必须定义在类外面，以便多进程 Pickle)
# =========================================================
def worker_collection_task(args):
    """
    单个进程执行的任务：加载一个模型，跑一个场景，收集数据
    """

    # [新增] 强制限制当前进程只使用单核进行数学运算！！！
    # 必须放在函数最开头，在 import torch 之后生效
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    import torch
    torch.set_num_threads(1)  # 限制 PyTorch 内部线程数

    # 解包参数
    (agent_id, scen_id, model_path, env_cfg_dict, pm_root,
     mode, target_count, output_dir, deterministic_ratio, worker_nice) = args

    # 低优先级执行，减少对同机其他实验的抢占（仅 Linux 生效）
    try:
        if worker_nice is not None:
            os.nice(int(worker_nice))
    except Exception:
        pass

    # 1. 重建 PathManager
    pm = PathManager(pm_root)

    # 2. 重建配置 & 强制覆盖场景
    # 将字典转回 OmegaConf 以便操作
    full_env_cfg = OmegaConf.create(env_cfg_dict)

    # 强制设置模式
    full_env_cfg.env_settings.mode = mode

    # 强制覆盖 active_scenario_list
    scenario_mode = full_env_cfg.env_settings.scenario_mode
    target_cfg = full_env_cfg.env_settings[scenario_mode][mode]
    target_cfg.active_scenario_list = [scen_id]

    # 3. 初始化环境
    # Seed Strategy: Base + Scen*1000 + Agent*100
    base_seed = 1000 if mode == 'training' else 2024
    seed = base_seed + (scen_id * 1000) + (agent_id * 100)

    def _make_env():
        return HierarchicalSlicingEnv(full_env_cfg.env_settings, np.random.default_rng(seed), pm)

    env = DummyVecEnv([_make_env])

    # 4. 加载模型
    try:
        model = PPO.load(
            model_path,
            env=env,
            device='cpu',
            custom_objects={
                "learning_rate": 0.0,
                "clip_range": 0.1,
                "HierarchicalSmartPolicy": HierarchicalSmartPolicy
            }
        )
    except Exception as e:
        # Fallback
        model = PPO.load(model_path, env=env, device='cpu')

    # 5. 执行收集循环
    save_dir = os.path.join(output_dir, mode)
    os.makedirs(save_dir, exist_ok=True)

    obs = env.reset()
    # [修正点 1] 在 Reset 后立即获取当前 Episode ID
    # 这是当前正在进行的 Episode
    current_phys_ep = env.get_attr("current_episode_idx")[0]
    curr_obs, curr_act, curr_rew, curr_done, curr_info = [], [], [], [], []
    collected_cnt = 0

    def _count_episode_violations(infos):
        hp = 0
        nhp = 0
        for inf in infos:
            if not isinstance(inf, dict):
                continue
            for k, v in inf.items():
                if not k.startswith("violation/slice_") or v <= 0:
                    continue
                s_idx = int(k.split('_')[1])
                prio = inf.get(f"meta/slice_{s_idx}_priority", 0)
                if prio > 0:
                    hp += 1
                else:
                    nhp += 1
        return hp, nhp

    while collected_cnt < target_count:
        # 混合策略
        use_deterministic = (np.random.rand() < deterministic_ratio)
        action, _ = model.predict(obs, deterministic=use_deterministic)

        next_obs, rewards, dones, infos = env.step(action)
        info = infos[0]

        # 存储
        step_obs = {k: v[0].copy() for k, v in obs.items()}
        curr_obs.append(step_obs)
        curr_act.append(action[0].copy())
        curr_rew.append(rewards[0])
        curr_done.append(dones[0])
        curr_info.append(info)

        obs = next_obs

        if dones[0]:
            ep_return = sum(curr_rew)

            hp_viols, nhp_viols = _count_episode_violations(curr_info)

            # 筛选：只过滤极端崩盘 (-300k)
            # if ep_return > -300000.0:
            if True:
                traj_data = {
                    "observations": curr_obs,
                    "actions": np.array(curr_act, dtype=np.float32),
                    "rewards": np.array(curr_rew, dtype=np.float32),
                    "dones": np.array(curr_done, dtype=bool),
                    "episode_returns": ep_return,
                    "episode_length": len(curr_act),
                    "hp_viols": hp_viols,
                    "nhp_viols": nhp_viols,
                    "meta_agent": agent_id,
                    "meta_scen": scen_id,
                    # [修正点 2] 使用循环开始时记录的 ID，而不是现在去 get_attr
                    "meta_phys_ep": current_phys_ep
                }

                pkl_name = f"S{scen_id}_A{agent_id}_ep_{collected_cnt}_ret_{ep_return:.0f}.pkl"
                with open(os.path.join(save_dir, pkl_name), "wb") as f:
                    pickle.dump(traj_data, f)

                collected_cnt += 1

            # Reset
            curr_obs, curr_act, curr_rew, curr_done, curr_info = [], [], [], [], []
            # [修正点 3] 更新 ID 为下一个 Episode (因为 DummyVecEnv 已经自动 reset 了)
            # 此时 env.current_episode_idx 已经指向了下一个
            current_phys_ep = env.get_attr("current_episode_idx")[0]

    env.close()
    return f"Finished: Agent {agent_id} on Scenario {scen_id} ({mode})"


class RobustCollector:
    def __init__(self, cfg, path_manager):
        self.cfg = cfg
        self.path_manager = path_manager

        dt_cfg = cfg.dt_dataset_collect if 'dt_dataset_collect' in cfg else cfg
        self.output_dir = dt_cfg.output_path
        self.episodes_per_scenario_training = dt_cfg.episodes_per_scenario_training
        self.episodes_per_scenario_evaluating = dt_cfg.episodes_per_scenario_evaluating
        self.runs_per_episode = dt_cfg.runs_per_episode
        self.deterministic_ratio = dt_cfg.deterministic_ratio
        self.max_workers_cfg = dt_cfg.get('max_workers', None)
        self.cpu_utilization_target = float(dt_cfg.get('cpu_utilization_target', 0.5))
        self.worker_nice = int(dt_cfg.get('worker_nice', 5))
        self.strict_model_check = bool(dt_cfg.get('strict_model_check', True))
        self.clean_output_path = bool(dt_cfg.get('clean_output_path', False))
        # A2 消融：数据策展模式，'lexicographic'（默认）或 'reward_only'
        self.curation_mode = str(dt_cfg.get('curation_mode', 'lexicographic'))

        if hasattr(dt_cfg, 'MODEL_ZOO'):
            self.MODEL_ZOO = OmegaConf.to_container(dt_cfg.MODEL_ZOO, resolve=True)
        else:
            self.MODEL_ZOO = {}

        if hasattr(dt_cfg, 'CROSS_SCENARIO_IDS'):
            self.CROSS_SCENARIO_IDS = OmegaConf.to_container(dt_cfg.CROSS_SCENARIO_IDS, resolve=True)
        else:
            self.CROSS_SCENARIO_IDS = [0]

        # 创建目录结构（可选先清理，避免手动删除旧资产）
        if self.clean_output_path and os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(os.path.join(self.output_dir, "training"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "evaluating"), exist_ok=True)
        # [新增] 垃圾桶目录，用于存放被筛选掉的文件（后悔药）
        os.makedirs(os.path.join(self.output_dir, "discarded_training"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "discarded_evaluating"), exist_ok=True)

    def _resolve_max_workers(self, n_tasks: int, max_workers=None):
        """解析并发度：支持显式配置；未配置时采用保守默认，避免吃满整机。"""
        if n_tasks <= 0:
            return 1

        # 1) 命令参数优先
        if max_workers is not None:
            return max(1, min(int(max_workers), n_tasks))

        # 2) YAML 显式配置
        if self.max_workers_cfg is not None:
            return max(1, min(int(self.max_workers_cfg), n_tasks))

        # 3) 自适应默认：目标占用率 + 当前系统负载感知
        cpu_cnt = max(1, os.cpu_count() or 1)
        target_workers = max(1, int(cpu_cnt * self.cpu_utilization_target))

        try:
            # load1 代表当前 runnable backlog，负载高时主动降并发
            load1 = os.getloadavg()[0]
            free_budget = max(1, int(cpu_cnt - load1))
            target_workers = min(target_workers, free_budget)
        except Exception:
            pass

        return max(1, min(target_workers, n_tasks))

    def collect_cross_scenarios_parallel(self, max_workers=None):
        """Step 1: 海量收集"""
        tasks = []
        if hasattr(self.cfg, 'environment'):
            env_cfg_dict = OmegaConf.to_container(self.cfg.environment, resolve=True)
        else:
            env_cfg_dict = OmegaConf.to_container(self.cfg.environment, resolve=True)

        pm_root = self.path_manager.hydra_workdir

        print(f"🚀 Preparing tasks for {len(self.MODEL_ZOO)} Agents x {len(self.CROSS_SCENARIO_IDS)} Scenarios...")

        missing_models = []
        for agent_id, model_path in self.MODEL_ZOO.items():
            if not os.path.exists(model_path):
                missing_models.append((agent_id, model_path))
        if missing_models:
            details = "\n".join([f"agent={aid}, path={mp}" for aid, mp in missing_models])
            if self.strict_model_check:
                raise FileNotFoundError(f"Missing MODEL_ZOO checkpoints:\n{details}")
            print(f"⚠️ Missing MODEL_ZOO checkpoints (will skip):\n{details}")

        for agent_id, model_path in self.MODEL_ZOO.items():
            if not os.path.exists(model_path):
                print(f"❌ Skipping Agent {agent_id}: Model not found")
                continue

            for scen_id in self.CROSS_SCENARIO_IDS:
                # 1. 训练集任务
                target_train = self.episodes_per_scenario_training * self.runs_per_episode
                args_train = (
                    agent_id, scen_id, model_path, env_cfg_dict, pm_root,
                    'training', target_train, self.output_dir, self.deterministic_ratio, self.worker_nice
                )
                tasks.append(args_train)

                # 2. 验证集任务
                # [关键修改] 验证集也进行冗余收集，以便筛选出高质量验证集
                target_eval = self.episodes_per_scenario_evaluating * self.runs_per_episode
                args_eval = (
                    agent_id, scen_id, model_path, env_cfg_dict, pm_root,
                    'evaluating', target_eval, self.output_dir, self.deterministic_ratio, self.worker_nice
                )
                tasks.append(args_eval)

        max_workers = self._resolve_max_workers(len(tasks), max_workers=max_workers)
        print(f"🔥 Launching {len(tasks)} tasks with max_workers={max_workers}, "
              f"cpu_target={self.cpu_utilization_target:.2f}, worker_nice={self.worker_nice} ...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker_collection_task, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Collection Progress"):
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ Task failed: {e}")

    def filter_best_of_n(self, mode, keep_count=40):
        """
        Step 2: 核心筛选逻辑 (Fixed Count Best-of-N)
        按 (Scenario, PhysicalEpisode) 分组，强制每组保留 Top N 条轨迹。

        Args:
            keep_count: 每个物理 Episode 保留的轨迹数量 (例如 40)
        """
        print(f"\n🧹 Filtering {mode} dataset (Keep Top {keep_count} per Episode)...")
        target_dir = os.path.join(self.output_dir, mode)
        discard_dir = os.path.join(self.output_dir, f"discarded_{mode}")

        if not os.path.exists(target_dir): return

        files = [f for f in os.listdir(target_dir) if f.endswith('.pkl')]
        group_map = {}

        print(f"   -> Scanning {len(files)} raw trajectories...")
        for f_name in tqdm(files):
            path = os.path.join(target_dir, f_name)
            try:
                with open(path, "rb") as f:
                    traj = pickle.load(f)

                scen = traj.get('meta_scen')
                phys_ep = traj.get('meta_phys_ep')

                # 排序键值：由 curation_mode 决定策略
                hp = traj.get('hp_viols', 999)
                nhp = traj.get('nhp_viols', 999)
                rew = traj['episode_returns']

                if self.curation_mode == 'reward_only':
                    # A2 消融：仅按 episode reward 排序，不考虑字典序约束
                    sort_key = (-rew,)
                else:
                    # 默认：字典序策展 (HP Viol, NHP Viol, -Reward)
                    sort_key = (hp, nhp, -rew)

                # 必须保证 key 包含 scen 和 phys_ep，才能精确分组
                if scen is not None and phys_ep is not None:
                    key = (scen, phys_ep)
                    if key not in group_map: group_map[key] = []

                    group_map[key].append({
                        'filename': f_name,
                        'path': path,
                        'sort_key': sort_key
                    })
                else:
                    # 如果元数据缺失，直接移入垃圾桶 (异常数据)
                    print(f"⚠️ Warning: Missing meta in {f_name}, discarding.")
                    try:
                        shutil.move(path, os.path.join(discard_dir, f_name))
                    except:
                        pass

            except Exception as e:
                print(f"Error reading {f_name}: {e}")

        # 2. 排序与移动
        total_kept = 0
        total_discarded = 0
        groups_processed = 0

        print(f"   -> Processing {len(group_map)} physical episodes...")

        for key, items in group_map.items():
            groups_processed += 1

            # 排序：好 -> 坏
            items.sort(key=lambda x: x['sort_key'])

            # [关键修改] 强制保留固定数量 (keep_count)
            # 如果某组总数不足 keep_count，则全保留 (但打印警告)
            if len(items) < keep_count:
                print(f"⚠️ Warning: Group {key} only has {len(items)} trajectories (Expected {keep_count})")
                n_keep = len(items)
            else:
                n_keep = keep_count

            keep_list = items[:n_keep]
            drop_list = items[n_keep:]

            total_kept += len(keep_list)
            total_discarded += len(drop_list)

            # 移动被淘汰的文件
            for item in drop_list:
                try:
                    src = item['path']
                    dst = os.path.join(discard_dir, item['filename'])
                    shutil.move(src, dst)
                except OSError as e:
                    # 忽略文件移动错误 (例如目标已存在)
                    pass

        print(f"✅ Filter {mode} Done.")
        print(f"   - Processed Groups: {groups_processed}")
        print(f"   - Kept Total: {total_kept} (Avg {total_kept / groups_processed:.1f} per ep)")
        print(f"   - Discarded: {total_discarded}")

    def compute_metadata(self):
        """Step 3: 元数据分析 (Enhanced)"""
        print("\n📊 Computing Global Metadata (Enhanced)...")
        train_dir = os.path.join(self.output_dir, "training")

        stats = {
            "inter": {"sum": 0, "sq_sum": 0, "count": 0},
            "intra": {"sum": 0, "sq_sum": 0, "count": 0},
            "global": {"sum": 0, "sq_sum": 0, "count": 0}
        }

        scenario_returns = {}
        source_stats = {}
        action_counts = {}

        if not os.path.exists(train_dir): return

        files = [f for f in os.listdir(train_dir) if f.endswith('.pkl')]
        print(f"Scanning {len(files)} refined trajectories...")

        for f_name in tqdm(files):
            try:
                with open(os.path.join(train_dir, f_name), "rb") as f:
                    traj = pickle.load(f)
            except:
                continue

            scen_id = traj.get('meta_scen', -1)
            agent_id = traj.get('meta_agent', -1)
            ep_ret = traj["episode_returns"]

            if scen_id not in scenario_returns: scenario_returns[scen_id] = []
            scenario_returns[scen_id].append(ep_ret)

            if scen_id not in source_stats: source_stats[scen_id] = {}
            source_stats[scen_id][agent_id] = source_stats[scen_id].get(agent_id, 0) + 1

            actions = traj['actions']
            if actions.ndim == 2:
                T, dims = actions.shape
                for d in range(dims):
                    if d not in action_counts: action_counts[d] = {}
                    vals, counts = np.unique(actions[:, d], return_counts=True)
                    for v, c in zip(vals, counts):
                        v = int(v)
                        action_counts[d][v] = action_counts[d].get(v, 0) + int(c)

            # Obs Norm Stats
            inter = np.array([o['inter_feat'] for o in traj['observations']])
            intra = np.array([o['intra_feat'] for o in traj['observations']])
            glob = np.array([o['global_feat'] for o in traj['observations']])

            flat_inter = inter.reshape(-1, 4)
            stats["inter"]["sum"] += flat_inter.sum(axis=0)
            stats["inter"]["sq_sum"] += (flat_inter ** 2).sum(axis=0)
            stats["inter"]["count"] += flat_inter.shape[0]

            flat_intra = intra.reshape(-1, 5)
            stats["intra"]["sum"] += flat_intra.sum(axis=0)
            stats["intra"]["sq_sum"] += (flat_intra ** 2).sum(axis=0)
            stats["intra"]["count"] += flat_intra.shape[0]

            flat_glob = glob.reshape(-1, 2)
            stats["global"]["sum"] += flat_glob.sum(axis=0)
            stats["global"]["sq_sum"] += (flat_glob ** 2).sum(axis=0)
            stats["global"]["count"] += flat_glob.shape[0]

        # Save Metadata
        metadata = {
            "obs_stats": {},
            "scenario_stats": {},
            "action_dist": action_counts,
            "expert_contrib": source_stats
        }

        for k in stats:
            N = stats[k]["count"]
            if N == 0: continue
            mean = stats[k]["sum"] / N
            var = (stats[k]["sq_sum"] / N) - (mean ** 2)
            std = np.sqrt(np.maximum(var, 1e-6))
            metadata["obs_stats"][k] = {"mean": mean.tolist(), "std": std.tolist()}

        print("\n" + "=" * 80)
        print(f"{'SCENARIO':<10} | {'MIN':<10} | {'MEAN':<10} | {'MAX':<10} | {'P95':<12}")
        print("-" * 80)
        for s_id in sorted(scenario_returns.keys()):
            rets = np.array(scenario_returns[s_id])
            meta = {
                "min": float(np.min(rets)),
                "mean": float(np.mean(rets)),
                "max": float(np.max(rets)),
                "p95": float(np.percentile(rets, 95))
            }
            metadata["scenario_stats"][str(s_id)] = meta
            print(
                f"Scenario {s_id:<2} | {meta['min']:<10.0f} | {meta['mean']:<10.0f} | {meta['max']:<10.0f} | {meta['p95']:<12.0f}")
        print("=" * 80)

        print("\n🏆 Expert Contribution (Filtered Dataset)")
        for s_id in sorted(source_stats.keys()):
            print(f"Scenario {s_id}: {source_stats[s_id]}")

        save_path = os.path.join(self.output_dir, "metadata.json")
        with open(save_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"✅ Metadata saved to {save_path}")


# =====================================================
# 后处理：单教师过滤 + 合并
# =====================================================
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


# =====================================================
# Hydra 接口函数
# =====================================================
def collect(cfg, path_manager):
    collector = RobustCollector(cfg, path_manager)

    # 1. 并行海量收集 (Training & Evaluating 都会收集冗余数据)
    collector.collect_cross_scenarios_parallel()

    keep_count_per_episode = cfg.keep_count_per_episode

    # 2. 筛选训练集 (Top N per episode)
    collector.filter_best_of_n(mode='training', keep_count=keep_count_per_episode)

    # 3. 筛选验证集 (Top N per episode)
    collector.filter_best_of_n(mode='evaluating', keep_count=keep_count_per_episode)

    # 4. 单教师过滤：每个场景只保留 best-of-N 后占比最高的教师
    print("\n🔬 Filtering to single best teacher per scenario...")
    _filter_single_teacher_per_scenario(collector.output_dir, 'training')
    _filter_single_teacher_per_scenario(collector.output_dir, 'evaluating')

    # 5. 合并 evaluating 到 training（闭环 metric 代替 val loss 做模型选择）
    print("\n📦 Merging evaluating into training...")
    _merge_eval_to_train(collector.output_dir)

    # 6. 生成元数据 (基于最终的 Training 数据)
    collector.compute_metadata()