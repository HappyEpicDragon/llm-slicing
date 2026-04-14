import numpy as np
import pandas as pd


class PhysicsProbe:
    """
    全能物理层探针 (Compatible with PPO-OneShot & Baseline)
    用于捕捉功率动态、频谱效率和服务饱和度。
    """

    def __init__(self):
        self.records = []

    def capture(self, env, action, step, info):
        """
        在每个 Step 结束时调用，抓取物理层快照
        """
        # === 1. 自动解包环境 (适配 DummyVecEnv / Wrapper) ===
        if hasattr(env, 'envs'):
            real_env = env.envs[0]
        elif hasattr(env, 'unwrapped'):
            real_env = env.unwrapped
        else:
            real_env = env

        # 适配 Baseline 的嵌套结构 (MARLCommEnv -> CommunicationEnv)
        if hasattr(real_env, 'comm_env'):
            real_env = real_env.comm_env

        # === 2. 获取基础数据 ===
        # 尝试获取 last_raw_obs (PPO环境特有) 或直接从组件获取 (Baseline)
        if hasattr(real_env, 'last_raw_obs') and real_env.last_raw_obs is not None:
            raw_obs = real_env.last_raw_obs
            buf = raw_obs.get("buffer_occupancies", np.zeros(real_env.components.ues.max_number_ues))
            # 兼容 bits 和 pkts
            incoming = raw_obs.get("pkt_incoming_bits",
                                   raw_obs.get("pkt_incoming", np.zeros(real_env.components.ues.max_number_ues)))
            csi = raw_obs.get("target_cell_power")
        elif hasattr(real_env, 'components'):
            # Baseline 模式: 直接读组件历史
            metrics = real_env.components.metrics.metrics_hist
            if not metrics['buffer_occupancies']: return  # 无数据
            buf = metrics['buffer_occupancies'][-1]
            incoming = metrics['pkt_incoming'][-1]  # Baseline 通常只存 pkts
            csi = metrics['target_cell_power'][-1]  # Baseline 存的是 tensor
        else:
            print("Warning: PhysicsProbe could not find observation data.")
            return

        # 获取环境参数
        # max_users = real_env.components.ues.max_number_ues
        # max_bs_power = real_env.config.basestations.total_power  # 或者是 100.0
        # num_phys_rbs = int(real_env.config.basestation_config.num_available_rbs[0])

        # 获取环境参数
        # [修复] 优先从环境实例属性获取，失败则尝试从 Config 获取，最后兜底
        if hasattr(real_env, 'max_bs_power'):
            max_bs_power = float(real_env.max_bs_power)
        else:
            # 尝试从 Config 路径获取 (兼容 PPO 和 Baseline 的不同结构)
            try:
                # PPO Config Structure: config.components.basestations.total_power
                max_bs_power = float(real_env.config.components.basestations.total_power)
            except:
                try:
                    # Alternative Structure
                    max_bs_power = float(real_env.config.basestations.total_power)
                except:
                    # Default Fallback
                    max_bs_power = 100.0

        # 获取 RB 信息
        # 同样增加鲁棒性
        try:
            if hasattr(real_env, 'num_phys_rbs'):
                num_phys_rbs = int(real_env.num_phys_rbs)
            elif hasattr(real_env, 'config') and hasattr(real_env.config, 'basestation_config'):
                num_phys_rbs = int(real_env.config.basestation_config.num_available_rbs[0])
            else:
                num_phys_rbs = 135  # Default
        except:
            num_phys_rbs = 135


        # 估算 RBG 大小 (通常是 5)
        rbg_size = 5
        num_rbgs = num_phys_rbs // rbg_size

        # === 3. 动作解析与功率计算 ===
        active_rbgs_count = 0
        user_alloc_map = np.zeros(num_rbgs, dtype=int)  # [RBG_ID] -> User_ID (1-based)

        # 情况 A: PPO-OneShot (Action 是 Flat Array [27])
        if isinstance(action, np.ndarray) and action.ndim == 1 and len(action) == num_rbgs:
            active_rbgs_count = np.sum(action > 0)
            user_alloc_map = action

            # PPO 动态功率
            power_scale = info.get("power_scale_factor", 0.0)
            if power_scale == 0 and active_rbgs_count > 0:
                power_scale = max_bs_power / active_rbgs_count
            elif active_rbgs_count == 0:
                power_scale = 0.0

        # 情况 B: Baseline (Action 是 dict, 物理分配在 sched_decision)
        else:
            # Baseline 的 sched_decision 是 [BS, Users, RBs] -> [1, 25, 135]
            if hasattr(real_env, 'components'):
                hist = real_env.components.metrics.metrics_hist
                if hist['sched_decision']:
                    sched = hist['sched_decision'][-1]
                    if sched.ndim == 3: sched = sched[0]

                    # 统计激活的 RB (物理层)
                    active_rbs = np.sum(sched > 0)
                    active_rbgs_count = active_rbs / 5.0  # 估算

                    # 映射回 RBG map (简化：只要该 RBG 内有分配就算)
                    for r in range(num_rbgs):
                        start = r * rbg_size
                        end = start + rbg_size
                        # 看看这段谁分到了
                        users_in_rbg = np.where(np.sum(sched[:, start:end], axis=1) > 0)[0]
                        if len(users_in_rbg) > 0:
                            user_alloc_map[r] = users_in_rbg[0] + 1  # 取第一个，1-based

            # Baseline 静态功率 (假设)
            power_scale = 1.0

            # === 4. 用户级详情与饱和度 ===
        # 识别有需求的用户
        demand_mask = (buf > 1e-9) | (incoming > 1e-9)
        total_demand_users = np.sum(demand_mask)

        # 识别被服务的用户
        served_users = set(user_alloc_map[user_alloc_map > 0] - 1)
        served_count = len(served_users)

        # === 5. 信道质量分析 (Spectral Efficiency) ===
        # 预处理 CSI [Users, RBs]
        if csi.ndim > 2: csi = np.squeeze(csi)
        if csi.shape[0] == num_phys_rbs: csi = csi.T

        allocated_gain_sum = 0.0
        valid_allocs = 0

        for r in range(num_rbgs):
            u_id = user_alloc_map[r]
            if u_id > 0:
                u_idx = u_id - 1
                start_rb = r * rbg_size
                end_rb = start_rb + rbg_size

                # 取该用户在该 RBG 的平均 CSI
                if u_idx < csi.shape[0] and end_rb <= csi.shape[1]:
                    gain = np.mean(csi[u_idx, start_rb:end_rb])
                    allocated_gain_sum += gain
                    valid_allocs += 1

        avg_se = 0.0
        if valid_allocs > 0:
            avg_allocated_gain = allocated_gain_sum / valid_allocs
            # 估算 SE (Spectral Efficiency) ~ log2(1 + SNR)
            noise = 1e-14
            # 注意：Baseline 的 power_scale 是 1.0，但它其实是归一化后的
            # 为了公平对比，我们假设 PPO 的 scale 是相对于 Baseline 的倍数
            # 或者直接用物理公式：SNR = Gain * (Total_Power / Active_RBs) / Noise
            # 这里的 power_scale 已经反映了分配给每个 RBG 的相对功率
            avg_snr = avg_allocated_gain * power_scale * (max_bs_power / num_rbgs) / noise
            avg_se = np.log2(1 + avg_snr)

        # === 6. 记录 ===
        self.records.append({
            "step": step,
            "active_rbgs": active_rbgs_count,
            "power_scale": power_scale,
            "demand_users": total_demand_users,
            "served_users": served_count,
            "avg_se": avg_se,
            # 如果有 Info 里的 Metrics，也记下来
            "hp_viol": info.get("metrics/hp_violations_total", 0) if info else 0,
            "nhp_viol": info.get("metrics/nhp_violations_total", 0) if info else 0
        })

    def report(self):
        """输出统计报告"""
        df = pd.DataFrame(self.records)
        if df.empty: return "No Data"

        print("\n" + "=" * 30 + " PHYSICS PROBE REPORT " + "=" * 30)
        print(f"1. Power Strategy:")
        print(f"   - Avg Active RBGs: {df['active_rbgs'].mean():.2f} / 27")
        print(f"   - Avg Power Scale: {df['power_scale'].mean():.4f}")

        print(f"2. Efficiency:")
        print(f"   - Avg Spectral Eff: {df['avg_se'].mean():.4f} bits/Hz")

        print(f"3. Service Saturation:")
        # 防止除0
        saturation = (df['served_users'] / (df['demand_users'] + 1e-6)).mean()
        print(f"   - Service Saturation: {saturation:.2%}")
        print(f"   - Avg Demand Users: {df['demand_users'].mean():.1f}")
        print(f"   - Avg Served Users: {df['served_users'].mean():.1f}")

        print("=" * 86 + "\n")


class SLAProbe:
    def __init__(self, env):
        self.records = []

        # === [关键修复] 自动解包环境 ===
        # 1. SB3 DummyVecEnv / SubprocVecEnv
        if hasattr(env, 'envs'):
            self.env = env.envs[0]
        # 2. Gym Wrapper
        elif hasattr(env, 'unwrapped'):
            self.env = env.unwrapped
        # 3. 裸环境
        else:
            self.env = env

        # 预计算全局 ID 到 (Slice, Local_ID) 的映射
        self.uid_to_slice_map = {}

        # 现在可以通过 self.env 安全访问 components 了
        # 注意：适配 Baseline 环境可能没有 num_slices 属性，需做容错
        if hasattr(self.env, 'num_slices'):
            # PPO-OneShot 环境
            num_slices = self.env.num_slices
            slice_assoc = self.env.components.slices.ue_assoc
        elif hasattr(self.env, 'components'):
            # Baseline 环境
            num_slices = self.env.components.slices.max_number_slices
            slice_assoc = self.env.components.slices.ue_assoc
        else:
            print("Warning: SLAProbe could not find slice association.")
            return

        for s_idx in range(num_slices):
            u_ids = np.where(slice_assoc[s_idx] > 0)[0]
            for i, u_id in enumerate(u_ids):
                self.uid_to_slice_map[u_id] = (s_idx, i)

    def capture(self, action, step, info):
        """
        捕获每一帧的 SLA 状态
        """
        # 获取 Raw Obs
        # PPO-OneShot 有 last_raw_obs
        raw_obs = getattr(self.env, 'last_raw_obs', None)

        # Baseline 只能从 components.metrics 读
        if raw_obs is None and hasattr(self.env, 'components'):
            metrics = self.env.components.metrics.metrics_hist
            if not metrics['buffer_occupancies']: return

            # 构造一个临时的 raw_obs 字典以便复用逻辑
            raw_obs = {
                "buffer_occupancies": metrics['buffer_occupancies'][-1],
                "buffer_latencies": metrics['buffer_latencies'][-1],
                "dropped_pkts": metrics['dropped_pkts'][-1],
                # Baseline 并没有直接计算 intent_drift，这里可能拿不到
                # 如果是 Baseline，我们只能跳过 Drift 检查，只看 Buffer/Drops
                "intent_drift": None
            }

        if raw_obs is None: return

        # 1. 获取关键状态

        # 动作解析 (兼容 PPO Array 和 Baseline Dict)
        user_alloc_counts = np.zeros(self.env.components.ues.max_number_ues, dtype=int)

        if isinstance(action, np.ndarray):  # PPO-OneShot
            if action.ndim > 1: action = action[0]  # 解包 Batch
            for u_id in action:
                if u_id > 0: user_alloc_counts[u_id - 1] += 1
        elif isinstance(action, dict):  # Baseline
            # Baseline 的 action 是权重，我们需要看物理分配结果 (sched_decision)
            # 尝试从 info 或者 history 获取
            pass  # Baseline 的动作解析比较复杂，暂略，主要看结果指标

        # 状态
        buffers = raw_obs.get("buffer_occupancies")
        latencies = raw_obs.get("buffer_latencies")
        drops = raw_obs.get("dropped_pkts")

        # Drift (PPO Only)
        drifts = raw_obs.get("intent_drift")

        # 2. 遍历用户
        max_users = self.env.components.ues.max_number_ues
        for u_id in range(max_users):
            s_idx, local_idx = self.uid_to_slice_map.get(u_id, (-1, -1))
            if s_idx == -1: continue

            # 获取优先级
            req = self.env.components.slices.requirements.get(f'slice_{s_idx}', {})
            prio = req.get('priority', 0)

            # 获取 Drift
            min_drift = 0.0
            if drifts is not None and local_idx < drifts.shape[1]:
                user_drifts = drifts[s_idx, local_idx, :]
                valid_drifts = user_drifts[user_drifts > -1.5]
                if len(valid_drifts) > 0:
                    min_drift = np.min(valid_drifts)

            # 3. 判定状态
            status = "OK"
            if min_drift < 0:
                status = "VIOLATION"
            elif drops[u_id] > 0:
                status = "VIOLATION"  # Baseline 这种没 Drift 的，看 Drop 也算违约
            elif min_drift < 0.1 and drifts is not None:
                status = "WARNING"

            has_traffic = (buffers[u_id] > 1e-9) or (drops[u_id] > 0)

            if status == "VIOLATION" or (status == "WARNING" and has_traffic) or (user_alloc_counts[u_id] > 0):
                self.records.append({
                    "step": step,
                    "user": u_id,
                    "slice": s_idx,
                    "prio": "HP" if prio > 0 else "NHP",
                    "alloc": user_alloc_counts[u_id],
                    "buffer": buffers[u_id],
                    "latency": latencies[u_id],
                    "drops": drops[u_id],
                    "drift": min_drift,
                    "status": status
                })

    def report(self, episode_idx):
        """
        打印当前 Episode 的 SLA 诊断报告
        """
        df = pd.DataFrame(self.records)
        if df.empty: return

        print(f"\n{'=' * 40} SLA DIAGNOSIS (Ep {episode_idx}) {'=' * 40}")

        viol_df = df[df['status'] == 'VIOLATION']
        if len(viol_df) > 0:
            print(f"🚨 Total Violation Frames: {len(viol_df)}")

            nhp_viols = viol_df[viol_df['prio'] == 'NHP']
            if len(nhp_viols) > 0:
                starved = nhp_viols[(nhp_viols['alloc'] == 0) & (nhp_viols['buffer'] > 0)]
                ineffective = nhp_viols[nhp_viols['alloc'] > 0]
                dropped = nhp_viols[nhp_viols['drops'] > 0]

                print(f"   [NHP Failure Analysis]")
                print(f"   - Total: {len(nhp_viols)}")
                print(f"   - Type A (Starvation): {len(starved)} ({len(starved) / len(nhp_viols):.1%})")
                print(f"   - Type B (Ineffective): {len(ineffective)} ({len(ineffective) / len(nhp_viols):.1%})")
                print(f"   - Type C (Packet Drops): {len(dropped)}")
        else:
            print("✅ Perfect Episode! No Violations.")

        if len(viol_df) > 0:
            print(f"\n[Snapshot: Last 5 Violations]")
            print(
                f"{'Step':<5} | {'User':<4} | {'Prio':<3} | {'Alloc':<5} | {'Buf':<6} | {'Lat':<6} | {'Drift':<7} | {'Reason'}")
            print("-" * 65)
            for _, row in viol_df.tail(5).iterrows():
                reason = "Unknown"
                if row['alloc'] == 0:
                    reason = "Starved"
                elif row['drops'] > 0:
                    reason = "Dropped"
                elif row['latency'] > 0.1:
                    reason = "High Lat"
                else:
                    reason = "Low Rate"
                print(
                    f"{row['step']:<5} | {row['user']:<4} | {row['prio']:<3} | {row['alloc']:<5} | {row['buffer']:.4f} | {row['latency']:.4f} | {row['drift']:.4f} | {reason}")

        print("=" * 86 + "\n")
