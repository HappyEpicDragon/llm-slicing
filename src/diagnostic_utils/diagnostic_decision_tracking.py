"""
核心诊断：追踪"差决策"的产生过程

目标：记录每个导致violation的决策时刻，还原当时的完整状态
     包括：观察、动作、奖励、intent drift、buffer状态等

用法：
1. 将 DecisionTracker 集成到 global_slicing_env.py
2. 运行测试时自动记录所有"坏决策"
3. 事后分析：为什么模型做出了错误的选择
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class DecisionTracker:
    """
    决策追踪器：记录每个timestep的决策过程

    特别关注：
    1. 违约发生前的决策
    2. 违约发生时的决策
    3. 违约发生后的决策
    """

    def __init__(self, save_dir='./diagnostic'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 当前episode的记录
        self.episode_data = []

        # 违约追踪
        self.violation_history = defaultdict(list)  # {(slice, user, metric): [timesteps]}
        self.last_intent_drift = None

        # 决策质量统计
        self.good_decisions = 0
        self.bad_decisions = 0

    def record_timestep(self,
                        timestep: int,
                        episode: int,
                        observation: dict,
                        action: np.ndarray,
                        reward_components: dict,
                        metrics: dict,
                        allocation_result: dict):
        """
        记录单个timestep的完整信息

        Args:
            timestep: 当前时间步
            episode: 当前episode
            observation: 模型看到的观察 (归一化后的)
            action: 模型输出的动作
            reward_components: {base, shaping, guidance, total}
            metrics: 物理层返回的指标
            allocation_result: 实际分配结果 (user_rbs, user_power, etc)
        """

        # === 1. 提取Intent Drift (核心指标) ===
        intent_drift = metrics.get('intent_drift')  # [Slices, Users, 3]

        if intent_drift is None:
            return  # 没有drift数据，跳过

        # === 2. 检测违约 ===
        violations = self._detect_violations(intent_drift, timestep)

        # === 3. 分析动作质量 ===
        action_quality = self._analyze_action_quality(
            action,
            observation,
            metrics,
            allocation_result,
            violations
        )

        # === 4. 构建记录 ===
        record = {
            'timestep': int(timestep),
            'episode': int(episode),

            # === 决策输入 ===
            'observation': {
                'buffer_status': observation.get('user_buffer_status', np.array([])).tolist(),
                'intent_drift_raw': intent_drift.tolist(),
                'csi_mean_per_user': self._extract_csi_summary(observation),
                'slice_priority': observation.get('slice_priority', np.array([])).tolist(),
            },

            # === 决策输出 ===
            'action': {
                'user_ids': action.tolist(),
                'allocation_summary': self._summarize_allocation(action),
            },

            # === 决策结果 ===
            'allocation_result': {
                'user_rbs_count': allocation_result.get('user_rbs_count', []),
                'user_power': allocation_result.get('user_power', []),
                'slice_rbs_count': allocation_result.get('slice_rbs_count', []),
            },

            # === 物理层反馈 ===
            'metrics': {
                'throughput_bits': metrics.get('pkt_effective_thr', np.array([])).tolist(),
                'buffer_occupancies': metrics.get('buffer_occupancies', np.array([])).tolist(),
                'intent_drift': intent_drift.tolist(),
            },

            # === 奖励信号 ===
            'reward_components': reward_components,

            # === 违约信息 ===
            'violations': violations,

            # === 动作质量评估 ===
            'action_quality': action_quality,
        }

        # === 5. 判断是否为"关键决策" ===
        is_critical = self._is_critical_decision(violations, action_quality)
        record['is_critical'] = is_critical

        # 保存到episode数据
        self.episode_data.append(record)

        # 更新统计
        if action_quality['is_good']:
            self.good_decisions += 1
        else:
            self.bad_decisions += 1

        # 更新违约历史
        self.last_intent_drift = intent_drift

    def _detect_violations(self, intent_drift: np.ndarray, timestep: int) -> List[Dict]:
        """
        检测当前timestep的违约情况

        Returns:
            List of violations: [
                {
                    'slice': int,
                    'user': int,
                    'metric': str,
                    'drift': float,
                    'severity': str
                },
                ...
            ]
        """
        violations = []

        for s_idx in range(intent_drift.shape[0]):
            for u_idx in range(intent_drift.shape[1]):
                for m_idx, m_name in enumerate(['throughput', 'reliability', 'latency']):
                    drift = intent_drift[s_idx, u_idx, m_idx]

                    # 只关心有效的drift值
                    if drift <= -1.5:  # -2表示无效
                        continue

                    # 检测违约
                    if drift < 0:
                        severity = 'critical' if drift < -0.5 else 'moderate' if drift < -0.2 else 'minor'

                        violations.append({
                            'slice': int(s_idx),
                            'user': int(u_idx),
                            'metric': m_name,
                            'drift': float(drift),
                            'severity': severity
                        })

                        # 记录到历史
                        key = (s_idx, u_idx, m_name)
                        self.violation_history[key].append(timestep)

        return violations

    def _analyze_action_quality(self,
                                action: np.ndarray,
                                observation: dict,
                                metrics: dict,
                                allocation_result: dict,
                                violations: List[Dict]) -> Dict:
        """
        分析动作质量：这个动作是好还是坏？为什么？

        评估维度：
        1. 资源分配是否合理 (HP优先?)
        2. Buffer积压用户是否得到资源
        3. 违约用户是否得到更多资源
        4. 资源浪费 (分配给无需求用户)
        """

        buffer_status = observation.get('user_buffer_status', np.zeros(25))
        slice_priority = observation.get('slice_priority', np.zeros(5))
        user_to_slice = observation.get('user_to_slice', np.zeros((5, 25)))

        # 计算每个用户的优先级
        user_priority = np.zeros(25)
        for u in range(25):
            for s in range(5):
                if user_to_slice[s, u] > 0:
                    user_priority[u] = slice_priority[s]

        # 统计分配
        user_rbs_count = np.zeros(25)
        for rbg_idx, u_id in enumerate(action):
            if u_id > 0:
                user_rbs_count[u_id - 1] += 1

        # === 评估1: HP用户是否得到足够资源 ===
        hp_users = np.where(user_priority > 0)[0]
        hp_allocated = user_rbs_count[hp_users].sum() if len(hp_users) > 0 else 0
        total_allocated = user_rbs_count.sum()
        hp_allocation_ratio = hp_allocated / max(total_allocated, 1)

        # === 评估2: Buffer积压用户是否得到资源 ===
        high_buffer_users = np.where(buffer_status > 0.1)[0]  # Buffer > 10%
        high_buffer_allocated = user_rbs_count[high_buffer_users].sum() if len(high_buffer_users) > 0 else 0

        # === 评估3: 违约用户是否得到更多资源 ===
        violated_users = set()
        for v in violations:
            violated_users.add(v['user'])

        violated_allocated = sum(user_rbs_count[u] for u in violated_users)

        # === 评估4: 资源浪费 ===
        low_buffer_low_priority = np.where((buffer_status < 0.01) & (user_priority == 0))[0]
        wasted_allocation = user_rbs_count[low_buffer_low_priority].sum()

        # === 综合判断 ===
        issues = []

        if hp_allocation_ratio < 0.3 and len(hp_users) > 0:
            issues.append({
                'type': 'hp_underserved',
                'description': f'HP users only got {hp_allocation_ratio:.1%} of resources',
                'severity': 'critical'
            })

        if len(high_buffer_users) > 0 and high_buffer_allocated < len(high_buffer_users) * 2:
            issues.append({
                'type': 'buffer_backlog',
                'description': f'{len(high_buffer_users)} users with high buffer underserved',
                'severity': 'high'
            })

        if len(violated_users) > 0 and violated_allocated < len(violated_users) * 2:
            issues.append({
                'type': 'violation_not_addressed',
                'description': f'{len(violated_users)} violated users underserved',
                'severity': 'high'
            })

        if wasted_allocation > total_allocated * 0.3:
            issues.append({
                'type': 'resource_waste',
                'description': f'{wasted_allocation:.0f} RBs wasted on low-priority idle users',
                'severity': 'moderate'
            })

        return {
            'is_good': len([x for x in issues if x['severity'] in ['critical', 'high']]) == 0,
            'issues': issues,
            'stats': {
                'hp_allocation_ratio': float(hp_allocation_ratio),
                'high_buffer_served': int(high_buffer_allocated),
                'violated_served': int(violated_allocated),
                'wasted': int(wasted_allocation),
            }
        }

    def _is_critical_decision(self, violations: List[Dict], action_quality: Dict) -> bool:
        """判断是否为关键决策（值得重点分析）"""

        # 1. 有严重违约
        has_critical_violation = any(v['severity'] == 'critical' for v in violations)

        # 2. 有多个违约
        has_many_violations = len(violations) >= 3

        # 3. 动作质量差且有违约
        bad_action_with_violation = (not action_quality['is_good']) and len(violations) > 0

        return has_critical_violation or has_many_violations or bad_action_with_violation

    def _extract_csi_summary(self, observation: dict) -> List[float]:
        """提取CSI的概要信息 (每个用户的平均信道质量)"""
        csi_raw = observation.get('csi_current_rb', np.array([]))

        if len(csi_raw) == 0:
            return []

        # CSI shape: [Users * RBGs] = 675, reshape to [Users, RBGs]
        num_users = 25
        num_rbgs = 27

        try:
            csi_matrix = csi_raw.reshape(num_users, num_rbgs)
            return np.mean(csi_matrix, axis=1).tolist()
        except:
            return []

    def _summarize_allocation(self, action: np.ndarray) -> Dict:
        """总结分配结果"""
        user_counts = defaultdict(int)

        for rbg_idx, u_id in enumerate(action):
            if u_id > 0:
                user_counts[int(u_id)] += 1

        return {
            'num_active_users': len(user_counts),
            'max_rbs_per_user': max(user_counts.values()) if user_counts else 0,
            'min_rbs_per_user': min(user_counts.values()) if user_counts else 0,
            'user_distribution': dict(user_counts),
        }

    def save_episode(self, episode_id: int):
        """保存当前episode的所有记录"""
        if len(self.episode_data) == 0:
            return

        # === 1. 保存完整时间序列 ===
        full_path = self.save_dir / f'ep{episode_id}_full.jsonl'
        with open(full_path, 'w') as f:
            for record in self.episode_data:
                json.dump(record, f, cls=NumpyEncoder)
                f.write('\n')

        # === 2. 保存关键决策 ===
        critical_decisions = [r for r in self.episode_data if r['is_critical']]
        if len(critical_decisions) > 0:
            critical_path = self.save_dir / f'ep{episode_id}_critical.jsonl'
            with open(critical_path, 'w') as f:
                for record in critical_decisions:
                    json.dump(record, f, cls=NumpyEncoder)
                    f.write('\n')

        # === 3. 生成汇总报告 ===
        summary = self._generate_episode_summary(episode_id)
        summary_path = self.save_dir / f'ep{episode_id}_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)

        print(f"\n{'=' * 70}")
        print(f"Episode {episode_id} Decision Tracking Summary")
        print(f"{'=' * 70}")
        print(f"Total Decisions: {len(self.episode_data)}")
        print(f"  Good: {self.good_decisions} ({self.good_decisions / len(self.episode_data):.1%})")
        print(f"  Bad:  {self.bad_decisions} ({self.bad_decisions / len(self.episode_data):.1%})")
        print(f"Critical Decisions: {len(critical_decisions)}")
        print(f"Total Violations: {summary['total_violations']}")
        print(f"  By Severity:")
        for sev, count in summary['violations_by_severity'].items():
            print(f"    {sev}: {count}")
        print(f"\nFiles saved:")
        print(f"  Full:     {full_path}")
        if len(critical_decisions) > 0:
            print(f"  Critical: {critical_path}")
        print(f"  Summary:  {summary_path}")
        print(f"{'=' * 70}\n")

        # 重置
        self.episode_data = []
        self.good_decisions = 0
        self.bad_decisions = 0

    def _generate_episode_summary(self, episode_id: int) -> Dict:
        """生成episode汇总统计"""

        # 统计违约
        all_violations = []
        for record in self.episode_data:
            all_violations.extend(record['violations'])

        violations_by_severity = defaultdict(int)
        violations_by_metric = defaultdict(int)
        violations_by_slice = defaultdict(int)

        for v in all_violations:
            violations_by_severity[v['severity']] += 1
            violations_by_metric[v['metric']] += 1
            violations_by_slice[v['slice']] += 1

        # 统计动作质量问题
        all_issues = defaultdict(int)
        for record in self.episode_data:
            for issue in record['action_quality']['issues']:
                all_issues[issue['type']] += 1

        # 统计奖励
        rewards = [r['reward_components'] for r in self.episode_data]
        avg_base = np.mean([r['base'] for r in rewards])
        avg_shaping = np.mean([r['shaping'] for r in rewards])
        avg_guidance = np.mean([r['guidance'] for r in rewards])
        avg_total = np.mean([r['total'] for r in rewards])

        return {
            'episode': episode_id,
            'total_timesteps': len(self.episode_data),
            'total_violations': len(all_violations),
            'violations_by_severity': dict(violations_by_severity),
            'violations_by_metric': dict(violations_by_metric),
            'violations_by_slice': dict(violations_by_slice),
            'action_quality': {
                'good_ratio': self.good_decisions / len(self.episode_data),
                'common_issues': dict(all_issues),
            },
            'reward_stats': {
                'base_mean': float(avg_base),
                'shaping_mean': float(avg_shaping),
                'guidance_mean': float(avg_guidance),
                'total_mean': float(avg_total),
            }
        }


# ========================================
# 集成到 global_slicing_env.py
# ========================================

"""
在 global_slicing_env.py 中添加：

1. __init__ 中初始化：
   self.decision_tracker = DecisionTracker()

2. step() 中记录：

   # 在计算完reward之后，物理执行完成后
   allocation_result = {
       'user_rbs_count': np.bincount(user_ids[user_ids > 0] - 1, minlength=self.max_users).tolist(),
       'user_power': user_allocated_power,
       'slice_rbs_count': slice_allocated_rbs,
   }

   self.decision_tracker.record_timestep(
       timestep=self.current_timestep,
       episode=self.current_episode_idx,
       observation=obs,  # 返回给agent的观察
       action=action,
       reward_components={
           'base': self.last_base_reward,
           'shaping': self.last_shaping_reward,
           'guidance': self.last_guidance_reward,
           'total': reward,
       },
       metrics=full_metrics_data,
       allocation_result=allocation_result
   )

3. reset() 或 step(done=True) 中保存：

   if done:
       self.decision_tracker.save_episode(self.current_episode_idx)
"""


# ========================================
# 分析脚本
# ========================================

def analyze_critical_decisions(episode_id: int, top_k: int = 10):
    """
    分析最差的K个决策，找出共性
    """
    import pandas as pd

    log_dir = Path('/home/claude/diagnostics/decisions')
    critical_file = log_dir / f'ep{episode_id}_critical.jsonl'

    if not critical_file.exists():
        print(f"No critical decisions file found for episode {episode_id}")
        return

    # 读取数据
    records = []
    with open(critical_file, 'r') as f:
        for line in f:
            records.append(json.loads(line))

    print(f"\n{'=' * 70}")
    print(f"Critical Decision Analysis - Episode {episode_id}")
    print(f"{'=' * 70}\n")
    print(f"Total critical decisions: {len(records)}")

    # === 1. 按时间分布 ===
    timesteps = [r['timestep'] for r in records]
    early = sum(1 for t in timesteps if t < 333)
    mid = sum(1 for t in timesteps if 333 <= t < 666)
    late = sum(1 for t in timesteps if t >= 666)

    print(f"\nTemporal Distribution:")
    print(f"  Early (0-333):   {early} ({early / len(records):.1%})")
    print(f"  Mid (333-666):   {mid} ({mid / len(records):.1%})")
    print(f"  Late (666-1000): {late} ({late / len(records):.1%})")

    # === 2. 最常见的问题类型 ===
    issue_counts = defaultdict(int)
    for r in records:
        for issue in r['action_quality']['issues']:
            issue_counts[issue['type']] += 1

    print(f"\nMost Common Issues:")
    for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {issue_type}: {count} times ({count / len(records):.1%})")

    # === 3. 找出"最差"的几个决策 ===
    # 定义差的程度：违约数量 + 问题严重性
    def badness_score(record):
        num_violations = len(record['violations'])
        critical_violations = sum(1 for v in record['violations'] if v['severity'] == 'critical')
        num_issues = len(record['action_quality']['issues'])

        return num_violations + critical_violations * 2 + num_issues

    worst_decisions = sorted(records, key=badness_score, reverse=True)[:top_k]

    print(f"\nTop {top_k} Worst Decisions:")
    for i, r in enumerate(worst_decisions, 1):
        print(f"\n[{i}] Timestep {r['timestep']}:")
        print(f"  Violations: {len(r['violations'])}")
        for v in r['violations'][:3]:  # 显示前3个
            print(f"    - Slice {v['slice']}, User {v['user']}, {v['metric']}: {v['drift']:.3f} ({v['severity']})")

        print(f"  Issues:")
        for issue in r['action_quality']['issues']:
            print(f"    - [{issue['severity']}] {issue['description']}")

        print(f"  Reward: base={r['reward_components']['base']:.3f}, "
              f"shaping={r['reward_components']['shaping']:.3f}, "
              f"total={r['reward_components']['total']:.3f}")

    # === 4. 保存详细分析 ===
    analysis_path = log_dir / f'ep{episode_id}_worst_decisions.json'
    with open(analysis_path, 'w') as f:
        json.dump(worst_decisions, f, indent=2, cls=NumpyEncoder)

    print(f"\n💾 Detailed analysis saved to: {analysis_path}")
    print(f"{'=' * 70}\n")


def find_decision_patterns(episode_range: range):
    """
    跨多个episodes寻找"坏决策"的共同模式
    """
    log_dir = Path('/home/claude/diagnostics/decisions')

    all_critical_decisions = []

    for ep_id in episode_range:
        critical_file = log_dir / f'ep{ep_id}_critical.jsonl'
        if critical_file.exists():
            with open(critical_file, 'r') as f:
                for line in f:
                    record = json.loads(line)
                    record['episode'] = ep_id
                    all_critical_decisions.append(record)

    if len(all_critical_decisions) == 0:
        print("No critical decisions found in the specified range")
        return

    print(f"\n{'=' * 70}")
    print(f"Pattern Analysis - Episodes {episode_range.start}-{episode_range.stop - 1}")
    print(f"{'=' * 70}\n")
    print(f"Total critical decisions: {len(all_critical_decisions)}")

    # === 模式1: HP用户服务不足 ===
    hp_underserved = [r for r in all_critical_decisions
                      if any(issue['type'] == 'hp_underserved'
                             for issue in r['action_quality']['issues'])]

    if len(hp_underserved) > 0:
        print(f"\n🔴 Pattern 1: HP Users Underserved")
        print(f"  Frequency: {len(hp_underserved)} / {len(all_critical_decisions)} "
              f"({len(hp_underserved) / len(all_critical_decisions):.1%})")

        # 分析这些情况下的共同特征
        avg_hp_ratio = np.mean([r['action_quality']['stats']['hp_allocation_ratio']
                                for r in hp_underserved])
        print(f"  Avg HP allocation ratio: {avg_hp_ratio:.1%}")

        # 检查buffer状态
        avg_hp_buffer = []
        for r in hp_underserved:
            buffers = r['observation']['buffer_status']
            priorities = r['observation']['slice_priority']
            # 粗略估计：假设前5个用户是HP (实际需要更精确的映射)
            if len(buffers) >= 5:
                avg_hp_buffer.extend(buffers[:5])

        if len(avg_hp_buffer) > 0:
            print(f"  Avg HP buffer when underserved: {np.mean(avg_hp_buffer):.3f}")

    # === 模式2: Buffer积压未处理 ===
    buffer_backlog = [r for r in all_critical_decisions
                      if any(issue['type'] == 'buffer_backlog'
                             for issue in r['action_quality']['issues'])]

    if len(buffer_backlog) > 0:
        print(f"\n⚠️  Pattern 2: Buffer Backlog Not Addressed")
        print(f"  Frequency: {len(buffer_backlog)} / {len(all_critical_decisions)} "
              f"({len(buffer_backlog) / len(all_critical_decisions):.1%})")

        # 看看这些决策的reward是怎样的
        avg_reward = np.mean([r['reward_components']['total'] for r in buffer_backlog])
        avg_shaping = np.mean([r['reward_components']['shaping'] for r in buffer_backlog])
        print(f"  Avg reward when buffer backlog: {avg_reward:.3f}")
        print(f"  Avg shaping reward: {avg_shaping:.3f}")

    # === 模式3: 资源浪费 ===
    resource_waste = [r for r in all_critical_decisions
                      if any(issue['type'] == 'resource_waste'
                             for issue in r['action_quality']['issues'])]

    if len(resource_waste) > 0:
        print(f"\n⚠️  Pattern 3: Resource Waste")
        print(f"  Frequency: {len(resource_waste)} / {len(all_critical_decisions)} "
              f"({len(resource_waste) / len(all_critical_decisions):.1%})")

        avg_wasted = np.mean([r['action_quality']['stats']['wasted'] for r in resource_waste])
        print(f"  Avg RBs wasted: {avg_wasted:.1f}")

    print(f"\n{'=' * 70}\n")


# ========================================
# 使用示例
# ========================================

if __name__ == "__main__":
    # 1. 分析单个episode的关键决策
    analyze_critical_decisions(episode_id=60, top_k=10)

    # 2. 跨多个episodes寻找模式
    find_decision_patterns(range(60, 80))