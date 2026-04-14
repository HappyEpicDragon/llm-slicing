"""
诊断2: Reward-Action因果分析

目标：追踪"reward信号如何影响未来的action选择"
     回答：模型是否从错误中学习？还是reward信号误导了模型？

关键问题：
1. 当reward惩罚某个action后，下次遇到类似情况，action是否改变？
2. 当某个action导致violation，但reward却不够负，模型是否意识到这是错误？
3. Shaping reward是否与violation趋势相反？(高shaping但高violation)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class RewardActionAnalyzer:
    """
    分析reward信号与action选择的因果关系
    """

    def __init__(self, save_dir='./diagnostic'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 记录每个timestep的关键信息
        self.history = []

        # 追踪"相似情况"下的决策变化
        self.state_action_memory = deque(maxlen=1000)  # 记录最近1000个(state, action)

    def record(self,
               timestep: int,
               observation: dict,
               action: np.ndarray,
               reward_components: dict,
               next_observation: dict,
               violations: List[Dict]):
        """
        记录单个timestep的信息

        特别关注：
        1. State -> Action 的映射
        2. Action -> Reward 的反馈
        3. Violation -> Reward 的对应关系
        """

        # 提取状态特征 (降维，便于比较)
        state_features = self._extract_state_features(observation)

        # 提取动作特征
        action_features = self._extract_action_features(action)

        # 构建记录
        record = {
            'timestep': int(timestep),
            'state_features': state_features,
            'action_features': action_features,
            'reward_components': reward_components,
            'num_violations': len(violations),
            'violation_severity': sum(1 for v in violations if v['severity'] == 'critical'),
        }

        self.history.append(record)

        # 保存到记忆库 (用于后续查找"相似状态")
        self.state_action_memory.append({
            'timestep': timestep,
            'state_features': state_features,
            'action_features': action_features,
            'reward': reward_components['total'],
        })

    def _extract_state_features(self, observation: dict) -> Dict:
        """
        从高维观察中提取关键特征

        目标：能够判断"两个state是否相似"
        """
        buffer_status = observation.get('user_buffer_status', np.zeros(25))
        slice_priority = observation.get('slice_priority', np.zeros(5))
        intent_drift = observation.get('intent_drift', np.zeros((5, 5, 3)))

        # 聚合特征
        features = {
            # Buffer状态
            'avg_buffer': float(np.mean(buffer_status)),
            'max_buffer': float(np.max(buffer_status)),
            'high_buffer_count': int(np.sum(buffer_status > 0.1)),

            # 优先级分布
            'hp_ratio': float(np.sum(slice_priority > 0) / len(slice_priority)),

            # Violation状态 (从intent_drift推断)
            'avg_drift': float(np.mean(intent_drift[intent_drift > -1.5])) if np.any(intent_drift > -1.5) else 0.0,
            'violation_count': int(np.sum((intent_drift < 0) & (intent_drift > -1.5))),
        }

        return features

    def _extract_action_features(self, action: np.ndarray) -> Dict:
        """
        从action中提取关键特征
        """
        user_counts = defaultdict(int)
        for u_id in action:
            if u_id > 0:
                user_counts[int(u_id)] += 1

        features = {
            'num_active_users': len(user_counts),
            'max_rbs_per_user': max(user_counts.values()) if user_counts else 0,
            'allocation_entropy': self._compute_entropy(list(user_counts.values())),
        }

        return features

    def _compute_entropy(self, distribution: List[int]) -> float:
        """计算分配的熵 (衡量资源分配的均匀性)"""
        if len(distribution) == 0:
            return 0.0

        total = sum(distribution)
        if total == 0:
            return 0.0

        probs = [x / total for x in distribution]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)

        return float(entropy)

    def analyze_episode(self, episode_id: int):
        """
        分析整个episode的reward-action因果关系

        输出：
        1. Reward与Violation的相关性
        2. "坏决策"是否得到足够惩罚
        3. 模型是否在重复相同的错误
        """
        if len(self.history) == 0:
            return

        print(f"\n{'=' * 70}")
        print(f"Reward-Action Causal Analysis - Episode {episode_id}")
        print(f"{'=' * 70}\n")

        # === 1. Reward vs Violation 相关性 ===
        self._analyze_reward_violation_correlation()

        # === 2. 识别"误导性reward" ===
        misleading_cases = self._find_misleading_rewards()

        # === 3. 检查"重复错误" ===
        repeated_mistakes = self._find_repeated_mistakes()

        # === 4. Shaping reward的作用分析 ===
        self._analyze_shaping_effect()

        # === 5. 保存分析结果 ===
        analysis = {
            'episode': episode_id,
            'misleading_rewards': misleading_cases,
            'repeated_mistakes': repeated_mistakes,
        }

        output_path = self.save_dir / f'ep{episode_id}_analysis.json'
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, cls=NumpyEncoder)

        print(f"\n💾 Analysis saved to: {output_path}")
        print(f"{'=' * 70}\n")

        # 重置
        self.history = []

    def _analyze_reward_violation_correlation(self):
        """
        分析reward与violation的相关性

        理想情况：violation多 -> reward低 (负相关)
        问题情况：violation多但reward不够低
        """
        rewards = [r['reward_components']['total'] for r in self.history]
        base_rewards = [r['reward_components']['base'] for r in self.history]
        shaping_rewards = [r['reward_components']['shaping'] for r in self.history]
        violations = [r['num_violations'] for r in self.history]

        # 计算相关系数
        from scipy.stats import pearsonr, spearmanr

        corr_total, p_total = spearmanr(violations, rewards)
        corr_base, p_base = spearmanr(violations, base_rewards)
        corr_shaping, p_shaping = spearmanr(violations, shaping_rewards)

        print("📊 Reward-Violation Correlation:")
        print(f"  Total Reward:   ρ = {corr_total:+.3f} (p={p_total:.4f})")
        print(f"  Base Reward:    ρ = {corr_base:+.3f} (p={p_base:.4f})")
        print(f"  Shaping Reward: ρ = {corr_shaping:+.3f} (p={p_shaping:.4f})")

        # 判断
        if corr_total > -0.3:
            print("\n  ⚠️  WARNING: Total reward weakly correlated with violations!")
            print("      → Model may not learn to avoid violations")

        if corr_base < -0.5 and corr_total > -0.3:
            print("\n  🔴 CRITICAL: Base reward punishes violations, but shaping cancels it!")
            print("      → Shaping reward is misleading the model")

        if corr_shaping > 0.2:
            print("\n  🔴 CRITICAL: Shaping reward POSITIVELY correlated with violations!")
            print("      → Model gets rewarded for causing violations")

        print()

    def _find_misleading_rewards(self) -> List[Dict]:
        """
        找出"误导性reward"的case

        定义：violation严重，但total reward不够低
        """
        misleading = []

        for r in self.history:
            # 严重违约：violation >= 3 或 有critical violation
            is_severe = (r['num_violations'] >= 3) or (r['violation_severity'] >= 1)

            # 但reward不够低：total > -0.5
            reward_too_high = r['reward_components']['total'] > -0.5

            if is_severe and reward_too_high:
                misleading.append({
                    'timestep': r['timestep'],
                    'num_violations': r['num_violations'],
                    'violation_severity': r['violation_severity'],
                    'base_reward': r['reward_components']['base'],
                    'shaping_reward': r['reward_components']['shaping'],
                    'total_reward': r['reward_components']['total'],
                    'cancellation_rate': r['reward_components']['shaping'] / abs(r['reward_components']['base']) if
                    r['reward_components']['base'] < 0 else 0,
                })

        if len(misleading) > 0:
            print(
                f"🚨 Misleading Reward Cases: {len(misleading)} / {len(self.history)} ({len(misleading) / len(self.history):.1%})")
            print(f"\n  Top 5 worst cases:")

            # 按cancellation_rate排序
            worst = sorted(misleading, key=lambda x: -x['cancellation_rate'])[:5]
            for i, case in enumerate(worst, 1):
                print(f"\n  [{i}] Timestep {case['timestep']}:")
                print(f"      Violations: {case['num_violations']} (severity: {case['violation_severity']})")
                print(f"      Base: {case['base_reward']:.3f}, Shaping: {case['shaping_reward']:+.3f}")
                print(f"      Total: {case['total_reward']:.3f}")
                print(f"      ⚠️  Cancellation Rate: {case['cancellation_rate']:.1%}")
        else:
            print("✅ No misleading rewards detected")

        print()
        return misleading

    def _find_repeated_mistakes(self) -> List[Dict]:
        """
        检查模型是否在"相似情况"下重复相同的错误

        逻辑：
        1. 找到导致violation的决策A
        2. 看后续是否有"相似的state"
        3. 检查在相似state下，action是否改变
        """
        repeated = []

        # 找出所有"坏决策" (有violation的)
        bad_decisions = [r for r in self.history if r['num_violations'] > 0]

        for bad_decision in bad_decisions:
            bad_timestep = bad_decision['timestep']
            bad_state = bad_decision['state_features']
            bad_action = bad_decision['action_features']

            # 在后续timesteps中找"相似state"
            for future_record in self.history:
                if future_record['timestep'] <= bad_timestep:
                    continue

                future_state = future_record['state_features']
                future_action = future_record['action_features']

                # 判断state相似性
                state_similarity = self._compute_state_similarity(bad_state, future_state)

                if state_similarity > 0.8:  # 高度相似
                    # 检查action是否改变
                    action_similarity = self._compute_action_similarity(bad_action, future_action)

                    if action_similarity > 0.8:  # action也相似
                        # 检查结果是否改善
                        if future_record['num_violations'] > 0:
                            repeated.append({
                                'original_timestep': bad_timestep,
                                'repeated_timestep': future_record['timestep'],
                                'state_similarity': state_similarity,
                                'action_similarity': action_similarity,
                                'original_violations': bad_decision['num_violations'],
                                'repeated_violations': future_record['num_violations'],
                            })
                            break  # 只记录第一次重复

        if len(repeated) > 0:
            print(f"🔄 Repeated Mistakes: {len(repeated)} cases")
            print(f"  → Model is NOT learning from errors in similar situations")
            print(f"\n  Examples:")
            for i, case in enumerate(repeated[:3], 1):
                print(f"    [{i}] Timestep {case['original_timestep']} → {case['repeated_timestep']}")
                print(f"        State similarity: {case['state_similarity']:.2f}")
                print(f"        Action similarity: {case['action_similarity']:.2f}")
                print(f"        Violations: {case['original_violations']} → {case['repeated_violations']}")
        else:
            print("✅ No obvious repeated mistakes detected")

        print()
        return repeated

    def _compute_state_similarity(self, state1: Dict, state2: Dict) -> float:
        """
        计算两个state的相似度

        使用简单的特征距离
        """
        # 归一化特征
        features = ['avg_buffer', 'max_buffer', 'high_buffer_count', 'hp_ratio', 'avg_drift', 'violation_count']

        distances = []
        for feat in features:
            v1 = state1.get(feat, 0)
            v2 = state2.get(feat, 0)

            # 归一化到[0, 1]
            if feat in ['avg_buffer', 'max_buffer', 'hp_ratio']:
                norm_diff = abs(v1 - v2)
            elif feat == 'high_buffer_count':
                norm_diff = abs(v1 - v2) / 25.0
            elif feat == 'avg_drift':
                norm_diff = abs(v1 - v2) / 2.0
            elif feat == 'violation_count':
                norm_diff = abs(v1 - v2) / 50.0
            else:
                norm_diff = abs(v1 - v2)

            distances.append(norm_diff)

        # 相似度 = 1 - 平均距离
        avg_distance = np.mean(distances)
        similarity = max(0, 1 - avg_distance)

        return similarity

    def _compute_action_similarity(self, action1: Dict, action2: Dict) -> float:
        """计算两个action的相似度"""
        features = ['num_active_users', 'max_rbs_per_user', 'allocation_entropy']

        distances = []
        for feat in features:
            v1 = action1.get(feat, 0)
            v2 = action2.get(feat, 0)

            # 归一化
            if feat == 'num_active_users':
                norm_diff = abs(v1 - v2) / 25.0
            elif feat == 'max_rbs_per_user':
                norm_diff = abs(v1 - v2) / 27.0
            elif feat == 'allocation_entropy':
                norm_diff = abs(v1 - v2) / 5.0
            else:
                norm_diff = abs(v1 - v2)

            distances.append(norm_diff)

        avg_distance = np.mean(distances)
        similarity = max(0, 1 - avg_distance)

        return similarity

    def _analyze_shaping_effect(self):
        """
        分析Shaping reward的实际效果

        问题：Shaping是否真的帮助了学习？还是反而误导？
        """
        print("📊 Shaping Reward Effect Analysis:")

        # 按shaping reward分组
        high_shaping = [r for r in self.history if r['reward_components']['shaping'] > 0.5]
        low_shaping = [r for r in self.history if r['reward_components']['shaping'] < -0.2]

        if len(high_shaping) > 0:
            avg_violations_high = np.mean([r['num_violations'] for r in high_shaping])
            print(f"\n  When shaping reward is HIGH (>0.5):")
            print(f"    Avg violations: {avg_violations_high:.2f}")
            print(
                f"    Frequency: {len(high_shaping)} / {len(self.history)} ({len(high_shaping) / len(self.history):.1%})")

        if len(low_shaping) > 0:
            avg_violations_low = np.mean([r['num_violations'] for r in low_shaping])
            print(f"\n  When shaping reward is LOW (<-0.2):")
            print(f"    Avg violations: {avg_violations_low:.2f}")
            print(
                f"    Frequency: {len(low_shaping)} / {len(self.history)} ({len(low_shaping) / len(self.history):.1%})")

        # 如果高shaping伴随高violation，那就是问题
        if len(high_shaping) > 0 and avg_violations_high > 2.0:
            print(f"\n  🔴 PROBLEM: High shaping reward often accompanies violations!")
            print(f"      → Shaping reward is NOT aligned with SLA satisfaction")

        print()


def plot_reward_violation_timeline(episode_id: int):
    """
    绘制reward和violation随时间的变化

    可视化：
    1. Total reward vs timestep
    2. Violations vs timestep
    3. Base vs Shaping vs Guidance
    """
    log_dir = Path('/home/claude/diagnostics/reward_action')

    # 读取分析数据 (需要从decision_tracking的数据中提取)
    decision_log = Path('/home/claude/diagnostics/decisions') / f'ep{episode_id}_full.jsonl'

    if not decision_log.exists():
        print(f"Decision log not found for episode {episode_id}")
        return

    timesteps = []
    base_rewards = []
    shaping_rewards = []
    total_rewards = []
    violations = []

    with open(decision_log, 'r') as f:
        for line in f:
            r = json.loads(line)
            timesteps.append(r['timestep'])
            base_rewards.append(r['reward_components']['base'])
            shaping_rewards.append(r['reward_components']['shaping'])
            total_rewards.append(r['reward_components']['total'])
            violations.append(len(r['violations']))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Reward components
    axes[0].plot(timesteps, base_rewards, label='Base', color='red', alpha=0.7)
    axes[0].plot(timesteps, shaping_rewards, label='Shaping', color='blue', alpha=0.7)
    axes[0].plot(timesteps, total_rewards, label='Total', color='black', linewidth=2)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].set_ylabel('Reward')
    axes[0].set_title(f'Reward Components - Episode {episode_id}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Violations
    axes[1].bar(timesteps, violations, color='orange', alpha=0.6, width=1.0)
    axes[1].set_ylabel('Number of Violations')
    axes[1].set_title('Violations Over Time')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Correlation view
    # 用scatter plot显示 base vs violations
    axes[2].scatter(violations, base_rewards, alpha=0.5, s=10, c=timesteps, cmap='viridis')
    axes[2].set_xlabel('Number of Violations')
    axes[2].set_ylabel('Base Reward')
    axes[2].set_title('Base Reward vs Violations (colored by time)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path('/home/claude/diagnostics/reward_action') / f'ep{episode_id}_timeline.png'
    plt.savefig(output_path, dpi=150)
    print(f"📊 Timeline plot saved to: {output_path}")
    plt.close()


# ========================================
# 集成示例
# ========================================

"""
在 global_slicing_env.py 中：

1. __init__ 中初始化:
   self.reward_action_analyzer = RewardActionAnalyzer()

2. step() 中记录:
   self.reward_action_analyzer.record(
       timestep=self.current_timestep,
       observation=obs,
       action=action,
       reward_components={...},
       next_observation=next_obs,
       violations=violations
   )

3. Episode结束时分析:
   if done:
       self.reward_action_analyzer.analyze_episode(self.current_episode_idx)
"""

if __name__ == "__main__":
    # 示例用法
    print("This is a module, import it in your analysis scripts")