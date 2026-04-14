import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pathlib import Path
import json


class RecurrentPPODiagnostics:
    """Recurrent PPO诊断工具"""

    def __init__(self, save_dir="./diagnostics"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 统计数据
        self.action_stats = defaultdict(list)
        self.reward_stats = []
        self.lstm_state_norms = []
        self.episode_allocations = []

    def collect_step_data(self,
                          timestep: int,
                          action: np.ndarray,
                          reward: float,
                          observation: dict,
                          lstm_states: dict = None):
        """收集单步数据"""
        user_id, power_level = action

        # 动作统计
        self.action_stats['user_ids'].append(int(user_id))
        self.action_stats['power_levels'].append(int(power_level))
        self.action_stats['timesteps'].append(timestep)

        # 奖励统计
        self.reward_stats.append({
            'timestep': timestep,
            'reward': reward
        })

        # LSTM状态范数（如果提供）
        if lstm_states:
            for batch_size, (h, c) in lstm_states.items():
                h_norm = torch.norm(h).item()
                c_norm = torch.norm(c).item()
                self.lstm_state_norms.append({
                    'timestep': timestep,
                    'h_norm': h_norm,
                    'c_norm': c_norm
                })

    def collect_episode_data(self, episode_num: int, rb_allocations: np.ndarray):
        """收集整个episode的RB分配"""
        self.episode_allocations.append({
            'episode': episode_num,
            'allocations': rb_allocations.copy()
        })

    def analyze_action_distribution(self):
        """分析动作分布"""
        user_ids = self.action_stats['user_ids']
        power_levels = self.action_stats['power_levels']

        # 用户选择频率
        user_counter = Counter(user_ids)
        power_counter = Counter(power_levels)

        # 计算熵
        def entropy(counts):
            probs = np.array(list(counts.values())) / sum(counts.values())
            return -np.sum(probs * np.log(probs + 1e-10))

        user_entropy = entropy(user_counter)
        power_entropy = entropy(power_counter)

        results = {
            'user_selection': {
                'distribution': dict(user_counter),
                'entropy': float(user_entropy),
                'unique_users': len(user_counter),
                'most_common': user_counter.most_common(5)
            },
            'power_selection': {
                'distribution': dict(power_counter),
                'entropy': float(power_entropy),
                'unique_levels': len(power_counter),
                'most_common': power_counter.most_common(5)
            }
        }

        # 打印摘要
        print("\n" + "=" * 80)
        print("动作分布诊断")
        print("=" * 80)
        print(f"\n用户选择:")
        print(f"  熵: {user_entropy:.3f} (最大: {np.log(26):.3f})")
        print(f"  使用的用户数: {len(user_counter)}/26")
        print(f"  前5常用用户: {user_counter.most_common(5)}")

        print(f"\n功率选择:")
        print(f"  熵: {power_entropy:.3f} (最大: {np.log(10):.3f})")
        print(f"  使用的功率等级: {len(power_counter)}/10")
        print(f"  前5常用等级: {power_counter.most_common(5)}")
        print("=" * 80)

        return results

    def analyze_exploration(self):
        """分析探索程度"""
        user_ids = np.array(self.action_stats['user_ids'])
        power_levels = np.array(self.action_stats['power_levels'])

        # 统计唯一的(user, power)组合
        action_pairs = set(zip(user_ids, power_levels))

        # 时间序列的多样性
        window_size = 135  # 一个episode
        n_windows = len(user_ids) // window_size

        window_diversities = []
        for i in range(n_windows):
            start = i * window_size
            end = (i + 1) * window_size
            window_actions = set(zip(
                user_ids[start:end],
                power_levels[start:end]
            ))
            window_diversities.append(len(window_actions))

        results = {
            'total_unique_actions': len(action_pairs),
            'max_possible_actions': 26 * 10,
            'coverage': len(action_pairs) / (26 * 10),
            'avg_window_diversity': float(np.mean(window_diversities)) if window_diversities else 0,
        }

        print(f"\n探索程度:")
        print(f"  唯一动作组合: {len(action_pairs)}/{26 * 10} ({results['coverage']:.2%})")
        print(f"  平均每episode唯一动作数: {results['avg_window_diversity']:.1f}")

        return results

    def plot_diagnostics(self):
        """生成诊断图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        user_ids = self.action_stats['user_ids']
        power_levels = self.action_stats['power_levels']
        timesteps = self.action_stats['timesteps']

        # 1. 用户选择分布
        ax = axes[0, 0]
        user_counter = Counter(user_ids)
        ax.bar(user_counter.keys(), user_counter.values())
        ax.set_xlabel('User ID')
        ax.set_ylabel('Frequency')
        ax.set_title('User Selection Distribution')
        ax.grid(True, alpha=0.3)

        # 2. 功率选择分布
        ax = axes[0, 1]
        power_counter = Counter(power_levels)
        ax.bar(power_counter.keys(), power_counter.values())
        ax.set_xlabel('Power Level')
        ax.set_ylabel('Frequency')
        ax.set_title('Power Level Distribution')
        ax.grid(True, alpha=0.3)

        # 3. 用户选择时间序列（前1000步）
        ax = axes[0, 2]
        plot_steps = min(1000, len(user_ids))
        ax.scatter(range(plot_steps), user_ids[:plot_steps], alpha=0.5, s=1)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('User ID')
        ax.set_title('User Selection Over Time (First 1000 steps)')
        ax.grid(True, alpha=0.3)

        # 4. 功率选择时间序列
        ax = axes[1, 0]
        ax.scatter(range(plot_steps), power_levels[:plot_steps], alpha=0.5, s=1)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Power Level')
        ax.set_title('Power Selection Over Time (First 1000 steps)')
        ax.grid(True, alpha=0.3)

        # 5. 奖励曲线
        ax = axes[1, 1]
        if self.reward_stats:
            rewards = [r['reward'] for r in self.reward_stats]
            # 移动平均
            window = 100
            if len(rewards) > window:
                smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
                ax.plot(smoothed, label='Smoothed (window=100)')
            ax.plot(rewards, alpha=0.3, label='Raw')
            ax.set_xlabel('Step')
            ax.set_ylabel('Reward')
            ax.set_title('Reward Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 6. LSTM状态范数
        ax = axes[1, 2]
        if self.lstm_state_norms:
            h_norms = [s['h_norm'] for s in self.lstm_state_norms]
            c_norms = [s['c_norm'] for s in self.lstm_state_norms]
            ax.plot(h_norms, label='Hidden state norm', alpha=0.7)
            ax.plot(c_norms, label='Cell state norm', alpha=0.7)
            ax.set_xlabel('Step')
            ax.set_ylabel('L2 Norm')
            ax.set_title('LSTM State Norms')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / 'diagnostics_plots.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ 诊断图表保存至: {save_path}")
        plt.close()

    def save_report(self):
        """保存完整报告"""
        action_dist = self.analyze_action_distribution()
        exploration = self.analyze_exploration()

        report = {
            'action_distribution': action_dist,
            'exploration': exploration,
            'total_steps': len(self.action_stats['user_ids']),
        }

        # 保存JSON
        json_path = self.save_dir / 'diagnosis_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✓ 诊断报告保存至: {json_path}")

        # 生成图表
        self.plot_diagnostics()

        return report