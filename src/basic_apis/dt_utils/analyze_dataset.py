import os
import pickle
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetAnalyzer:
    def __init__(self, dataset_path, output_path="dataset_report"):
        self.dataset_path = dataset_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

        self.stats = []

    def load_and_scan(self):
        files = [f for f in os.listdir(self.dataset_path) if f.endswith('.pkl')]
        print(f"🔍 Scanning {len(files)} trajectories in {self.dataset_path}...")

        for f_name in tqdm(files):
            try:
                # 解析文件名元数据 (S{scen}_A{agent}...)
                # 即使文件名不规范，通常内部 traj 也有 meta 信息
                parts = f_name.split('_')

                with open(os.path.join(self.dataset_path, f_name), "rb") as f:
                    traj = pickle.load(f)

                # 提取关键指标
                entry = {
                    "filename": f_name,
                    "agent_id": traj.get('meta_agent', -1),
                    "scen_id": traj.get('meta_scen', -1),
                    "return": traj['episode_returns'],
                    "length": traj['episode_length'],
                    "hp_viols": traj.get('hp_viols', 0),
                    # 如果有记录 nhp violations 也可以加在这里
                }

                # 文件名 Fallback (如果内部 meta 缺失)
                if entry['agent_id'] == -1 and len(parts) >= 2:
                    entry['scen_id'] = int(parts[0][1:])  # S0 -> 0
                    entry['agent_id'] = int(parts[1][1:])  # A0 -> 0

                self.stats.append(entry)

            except Exception as e:
                print(f"⚠️ Error reading {f_name}: {e}")

        self.df = pd.DataFrame(self.stats)
        print(f"✅ Loaded {len(self.df)} valid records.")

    def generate_report(self):
        if self.df.empty:
            print("❌ No data to analyze.")
            return

        report = {}

        # 1. 场景视角：Reward 分布
        # 它可以告诉你每个场景的“难度”和“分值范围”
        print("\n📊 Analyzing Scenario Statistics...")
        scen_stats = self.df.groupby('scen_id')['return'].describe().to_dict(orient='index')
        report['scenario_stats'] = scen_stats

        # 2. 专家矩阵：哪个专家在哪个场景最强？(Mean Return)
        # Pivot Table: Index=Scen, Col=Agent, Value=Mean Return
        pivot_table = self.df.pivot_table(index='scen_id', columns='agent_id', values='return', aggfunc='mean')
        report['expert_matrix_mean'] = pivot_table.to_dict()

        # 3. 违约矩阵：哪个专家最稳？(Mean HP Violations)
        pivot_viol = self.df.pivot_table(index='scen_id', columns='agent_id', values='hp_viols', aggfunc='mean')
        report['expert_matrix_viol'] = pivot_viol.to_dict()

        # 4. 保存 JSON
        json_path = os.path.join(self.output_path, "comprehensive_metadata.json")

        # 处理 DataFrame 转 JSON 的兼容性
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(NpEncoder, self).default(obj)

        with open(json_path, 'w') as f:
            json.dump(report, f, indent=4, cls=NpEncoder)

        print(f"💾 Detailed JSON report saved to {json_path}")

        # 5. (可选) 打印终端摘要
        print("\n" + "=" * 50)
        print("🏆 Best Performing Agents per Scenario (by Mean Return)")
        print("=" * 50)
        for scen_id in sorted(self.df['scen_id'].unique()):
            sub = self.df[self.df['scen_id'] == scen_id]
            best_agent = sub.groupby('agent_id')['return'].mean().idxmax()
            best_score = sub.groupby('agent_id')['return'].mean().max()
            print(f"Scenario {scen_id}: Agent {best_agent} (Score: {best_score:.2f})")


if __name__ == "__main__":
    # 修改为你的 Stage 3 数据集路径
    DATA_PATH = "/root/decision_transformer_slicing/data/channel_generality/dt/dataset/mix/training"
    OUTPUT_PATH = os.path.join(DATA_PATH, 'analysis')
    analyzer = DatasetAnalyzer(DATA_PATH, OUTPUT_PATH)
    analyzer.load_and_scan()
    analyzer.generate_report()