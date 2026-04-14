"""
诊断3: 成功vs失败Episodes对比分析

目标：对比"好episode"和"差episode"，找出关键差异
     回答：什么导致了某些episodes表现好，某些表现差？

关键问题：
1. 好episode和差episode的初始条件有何不同？(traffic pattern, channel quality)
2. 好episode的决策策略有什么特征？
3. 差episode在哪个阶段开始崩溃？
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


class SuccessFailureComparator:
    """
    对比成功和失败的episodes
    """

    def __init__(self,
                 test_range: range,
                 violation_threshold: float = 5.0,
                 decision_log_dir: str = './diagnostic'):
        """
        Args:
            test_range: 测试episodes的范围 (e.g., range(60, 80))
            violation_threshold: 违约阈值，超过此值认为是"失败"
            decision_log_dir: 决策日志目录
        """
        self.test_range = test_range
        self.violation_threshold = violation_threshold
        self.decision_log_dir = Path(decision_log_dir)

        # 收集所有episodes的汇总数据
        self.episode_summaries = {}

    def load_episode_summaries(self):
        """
        加载所有episodes的汇总信息
        """
        print("Loading episode summaries...")

        for ep_id in self.test_range:
            summary_file = self.decision_log_dir / f'ep{ep_id}_summary.json'

            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    self.episode_summaries[ep_id] = json.load(f)

        print(f"Loaded {len(self.episode_summaries)} episodes")

    def classify_episodes(self) -> tuple:
        """
        将episodes分类为"成功"和"失败"

        Returns:
            (success_episodes, failure_episodes)
        """
        success = []
        failure = []

        for ep_id, summary in self.episode_summaries.items():
            total_violations = summary['total_violations']

            # 归一化：violations per timestep
            violations_per_step = total_violations / summary['total_timesteps']

            if violations_per_step < self.violation_threshold / 1000:  # 5.0 violations in 1000 steps = 0.005 per step
                success.append(ep_id)
            else:
                failure.append(ep_id)

        return success, failure

    def compare_episodes(self):
        """
        主分析函数：对比成功和失败episodes
        """
        if len(self.episode_summaries) == 0:
            self.load_episode_summaries()

        success_eps, failure_eps = self.classify_episodes()

        print(f"\n{'=' * 70}")
        print("Success vs Failure Episode Comparison")
        print(f"{'=' * 70}\n")
        print(f"Success Episodes: {len(success_eps)} - {success_eps}")
        print(f"Failure Episodes: {len(failure_eps)} - {failure_eps}")
        print()

        if len(success_eps) == 0:
            print("⚠️  No success episodes found! All episodes are failures.")
            return

        if len(failure_eps) == 0:
            print("✅ All episodes are successful!")
            return

        # === 1. 对比违约统计 ===
        self._compare_violations(success_eps, failure_eps)

        # === 2. 对比reward分布 ===
        self._compare_rewards(success_eps, failure_eps)

        # === 3. 对比动作质量 ===
        self._compare_action_quality(success_eps, failure_eps)

        # === 4. 时间演化对比 ===
        self._compare_temporal_evolution(success_eps, failure_eps)

        # === 5. 识别关键差异 ===
        self._identify_key_differences(success_eps, failure_eps)

    def _compare_violations(self, success_eps: List[int], failure_eps: List[int]):
        """对比违约统计"""
        print("📊 Violation Statistics:\n")

        # Success episodes
        success_violations = [self.episode_summaries[ep]['total_violations'] for ep in success_eps]
        success_by_severity = defaultdict(list)
        for ep in success_eps:
            for sev, count in self.episode_summaries[ep]['violations_by_severity'].items():
                success_by_severity[sev].append(count)

        # Failure episodes
        failure_violations = [self.episode_summaries[ep]['total_violations'] for ep in failure_eps]
        failure_by_severity = defaultdict(list)
        for ep in failure_eps:
            for sev, count in self.episode_summaries[ep]['violations_by_severity'].items():
                failure_by_severity[sev].append(count)

        print(f"  Success Episodes:")
        print(f"    Total violations: {np.mean(success_violations):.1f} ± {np.std(success_violations):.1f}")
        for sev in ['critical', 'moderate', 'minor']:
            if sev in success_by_severity:
                print(f"    {sev}: {np.mean(success_by_severity[sev]):.1f} ± {np.std(success_by_severity[sev]):.1f}")

        print(f"\n  Failure Episodes:")
        print(f"    Total violations: {np.mean(failure_violations):.1f} ± {np.std(failure_violations):.1f}")
        for sev in ['critical', 'moderate', 'minor']:
            if sev in failure_by_severity:
                print(f"    {sev}: {np.mean(failure_by_severity[sev]):.1f} ± {np.std(failure_by_severity[sev]):.1f}")

        # 统计显著性检验
        from scipy.stats import mannwhitneyu
        stat, p_value = mannwhitneyu(success_violations, failure_violations)
        print(f"\n  Statistical Significance: p = {p_value:.4f}")
        if p_value < 0.05:
            print(f"    ✅ Significant difference between success and failure")
        else:
            print(f"    ⚠️  No significant difference")

        print()

    def _compare_rewards(self, success_eps: List[int], failure_eps: List[int]):
        """对比reward分布"""
        print("💰 Reward Statistics:\n")

        # Collect reward stats
        success_rewards = {
            'base': [],
            'shaping': [],
            'total': []
        }
        failure_rewards = {
            'base': [],
            'shaping': [],
            'total': []
        }

        for ep in success_eps:
            stats = self.episode_summaries[ep]['reward_stats']
            success_rewards['base'].append(stats['base_mean'])
            success_rewards['shaping'].append(stats['shaping_mean'])
            success_rewards['total'].append(stats['total_mean'])

        for ep in failure_eps:
            stats = self.episode_summaries[ep]['reward_stats']
            failure_rewards['base'].append(stats['base_mean'])
            failure_rewards['shaping'].append(stats['shaping_mean'])
            failure_rewards['total'].append(stats['total_mean'])

        print(f"  Success Episodes:")
        for key in ['base', 'shaping', 'total']:
            print(f"    {key:8s}: {np.mean(success_rewards[key]):+.3f} ± {np.std(success_rewards[key]):.3f}")

        print(f"\n  Failure Episodes:")
        for key in ['base', 'shaping', 'total']:
            print(f"    {key:8s}: {np.mean(failure_rewards[key]):+.3f} ± {np.std(failure_rewards[key]):.3f}")

        # === 关键分析：shaping是否起作用 ===
        print(f"\n  📊 Shaping Reward Analysis:")
        success_shaping_mean = np.mean(success_rewards['shaping'])
        failure_shaping_mean = np.mean(failure_rewards['shaping'])

        if failure_shaping_mean > success_shaping_mean:
            print(f"    🔴 PROBLEM: Failure episodes have HIGHER shaping reward!")
            print(f"       Success: {success_shaping_mean:+.3f}")
            print(f"       Failure: {failure_shaping_mean:+.3f}")
            print(f"       → Shaping reward is MISLEADING the model")
        else:
            print(f"    ✅ Shaping reward aligns with success")

        print()

    def _compare_action_quality(self, success_eps: List[int], failure_eps: List[int]):
        """对比动作质量"""
        print("🎯 Action Quality:\n")

        # Collect issue counts
        success_issues = defaultdict(list)
        failure_issues = defaultdict(list)

        for ep in success_eps:
            issues = self.episode_summaries[ep]['action_quality']['common_issues']
            for issue_type, count in issues.items():
                success_issues[issue_type].append(count)

        for ep in failure_eps:
            issues = self.episode_summaries[ep]['action_quality']['common_issues']
            for issue_type, count in issues.items():
                failure_issues[issue_type].append(count)

        # 找出差异最大的问题
        all_issue_types = set(success_issues.keys()) | set(failure_issues.keys())

        issue_diffs = []
        for issue_type in all_issue_types:
            success_avg = np.mean(success_issues[issue_type]) if issue_type in success_issues else 0
            failure_avg = np.mean(failure_issues[issue_type]) if issue_type in failure_issues else 0

            diff = failure_avg - success_avg
            issue_diffs.append((issue_type, success_avg, failure_avg, diff))

        # 按差异排序
        issue_diffs.sort(key=lambda x: -abs(x[3]))

        print("  Top Issues (ranked by difference):\n")
        for issue_type, success_avg, failure_avg, diff in issue_diffs[:5]:
            print(f"    {issue_type}:")
            print(f"      Success: {success_avg:.1f}")
            print(f"      Failure: {failure_avg:.1f}")
            print(f"      Diff:    {diff:+.1f}")

            if diff > 10:
                print(f"      → 🔴 Key differentiator!")
            print()

    def _compare_temporal_evolution(self, success_eps: List[int], failure_eps: List[int]):
        """对比时间演化模式"""
        print("⏱️  Temporal Evolution:\n")

        # 选择一个代表性的success和failure episode
        if len(success_eps) > 0 and len(failure_eps) > 0:
            rep_success = success_eps[len(success_eps) // 2]  # 中位数
            rep_failure = failure_eps[len(failure_eps) // 2]

            print(f"  Analyzing representative episodes:")
            print(f"    Success: Episode {rep_success}")
            print(f"    Failure: Episode {rep_failure}\n")

            # 读取完整时间序列
            success_timeline = self._load_episode_timeline(rep_success)
            failure_timeline = self._load_episode_timeline(rep_failure)

            if success_timeline and failure_timeline:
                # 分析early/mid/late阶段
                stages = {
                    'early': (0, 333),
                    'mid': (333, 666),
                    'late': (666, 1000)
                }

                print("  Violations by stage:")
                for stage_name, (start, end) in stages.items():
                    success_violations = sum(1 for r in success_timeline
                                             if start <= r['timestep'] < end and len(r['violations']) > 0)
                    failure_violations = sum(1 for r in failure_timeline
                                             if start <= r['timestep'] < end and len(r['violations']) > 0)

                    print(f"    {stage_name:5s}: Success={success_violations:3d}, Failure={failure_violations:3d} "
                          f"(diff={failure_violations - success_violations:+3d})")

                # 识别"崩溃点"
                failure_violation_counts = [len(r['violations']) for r in failure_timeline]
                collapse_point = self._find_collapse_point(failure_violation_counts)

                if collapse_point:
                    print(f"\n  🔴 Failure episode collapse point: ~timestep {collapse_point}")
                    print(f"     → Analyze decisions around this timestep")

        print()

    def _load_episode_timeline(self, episode_id: int) -> List[Dict]:
        """加载episode的完整时间序列"""
        timeline_file = self.decision_log_dir / f'ep{episode_id}_full.jsonl'

        if not timeline_file.exists():
            return []

        timeline = []
        with open(timeline_file, 'r') as f:
            for line in f:
                timeline.append(json.loads(line))

        return timeline

    def _find_collapse_point(self, violation_counts: List[int]) -> int:
        """
        找出violation开始激增的"崩溃点"

        使用滑动窗口检测
        """
        window_size = 50
        threshold = 3  # 平均每个timestep超过3个violations

        for i in range(len(violation_counts) - window_size):
            window = violation_counts[i:i + window_size]
            if np.mean(window) > threshold:
                return i

        return None

    def _identify_key_differences(self, success_eps: List[int], failure_eps: List[int]):
        """
        总结关键差异
        """
        print("🔑 Key Differentiators:\n")

        # 1. Reward signal quality
        success_rewards = [self.episode_summaries[ep]['reward_stats'] for ep in success_eps]
        failure_rewards = [self.episode_summaries[ep]['reward_stats'] for ep in failure_eps]

        success_base_std = np.mean([r['base_mean'] for r in success_rewards])
        failure_base_std = np.mean([r['base_mean'] for r in failure_rewards])

        if abs(success_base_std - failure_base_std) < 0.1:
            print("  1. Base reward is SIMILAR between success and failure")
            print(f"     → Problem is NOT in SLA penalty signal")
        else:
            print("  1. Base reward differs significantly")
            print(f"     Success: {success_base_std:.3f}")
            print(f"     Failure: {failure_base_std:.3f}")

        # 2. Action quality
        success_good_ratio = np.mean([self.episode_summaries[ep]['action_quality']['good_ratio']
                                      for ep in success_eps])
        failure_good_ratio = np.mean([self.episode_summaries[ep]['action_quality']['good_ratio']
                                      for ep in failure_eps])

        print(f"\n  2. Action Quality:")
        print(f"     Success: {success_good_ratio:.1%} good decisions")
        print(f"     Failure: {failure_good_ratio:.1%} good decisions")

        if success_good_ratio - failure_good_ratio > 0.2:
            print(f"     → 🔴 Action quality is KEY differentiator!")

        print(f"\n{'=' * 70}\n")


def plot_success_failure_comparison(test_range: range):
    """
    可视化success vs failure的对比
    """
    comparator = SuccessFailureComparator(test_range)
    comparator.load_episode_summaries()
    success_eps, failure_eps = comparator.classify_episodes()

    if len(success_eps) == 0 or len(failure_eps) == 0:
        print("Cannot plot: need both success and failure episodes")
        return

    # 准备数据
    success_data = {
        'violations': [comparator.episode_summaries[ep]['total_violations'] for ep in success_eps],
        'base_reward': [comparator.episode_summaries[ep]['reward_stats']['base_mean'] for ep in success_eps],
        'shaping_reward': [comparator.episode_summaries[ep]['reward_stats']['shaping_mean'] for ep in success_eps],
        'total_reward': [comparator.episode_summaries[ep]['reward_stats']['total_mean'] for ep in success_eps],
    }

    failure_data = {
        'violations': [comparator.episode_summaries[ep]['total_violations'] for ep in failure_eps],
        'base_reward': [comparator.episode_summaries[ep]['reward_stats']['base_mean'] for ep in failure_eps],
        'shaping_reward': [comparator.episode_summaries[ep]['reward_stats']['shaping_mean'] for ep in failure_eps],
        'total_reward': [comparator.episode_summaries[ep]['reward_stats']['total_mean'] for ep in failure_eps],
    }

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Violation distribution
    axes[0, 0].hist(success_data['violations'], bins=10, alpha=0.6, label='Success', color='green')
    axes[0, 0].hist(failure_data['violations'], bins=10, alpha=0.6, label='Failure', color='red')
    axes[0, 0].set_xlabel('Total Violations')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Violation Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Base reward
    axes[0, 1].boxplot([success_data['base_reward'], failure_data['base_reward']],
                       labels=['Success', 'Failure'])
    axes[0, 1].set_ylabel('Base Reward')
    axes[0, 1].set_title('Base Reward Comparison')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Shaping reward
    axes[1, 0].boxplot([success_data['shaping_reward'], failure_data['shaping_reward']],
                       labels=['Success', 'Failure'])
    axes[1, 0].set_ylabel('Shaping Reward')
    axes[1, 0].set_title('Shaping Reward Comparison')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Scatter: violations vs total reward
    axes[1, 1].scatter(success_data['violations'], success_data['total_reward'],
                       alpha=0.6, label='Success', color='green', s=100)
    axes[1, 1].scatter(failure_data['violations'], failure_data['total_reward'],
                       alpha=0.6, label='Failure', color='red', s=100)
    axes[1, 1].set_xlabel('Total Violations')
    axes[1, 1].set_ylabel('Total Reward')
    axes[1, 1].set_title('Violations vs Reward')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path('./diagnostic') / 'success_failure_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"📊 Comparison plot saved to: {output_path}")
    plt.close()


# ========================================
# 使用示例
# ========================================

if __name__ == "__main__":
    # 分析测试集 (episodes 60-79)
    comparator = SuccessFailureComparator(test_range=range(60, 80))
    comparator.compare_episodes()

    # 绘图
    plot_success_failure_comparison(range(60, 80))