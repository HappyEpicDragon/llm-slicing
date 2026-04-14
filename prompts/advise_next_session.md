# 下一步探索的提示词

将以下内容粘贴到新会话中：

---

## 背景

我们在做网络切片资源调度的论文实验，代码仓库在 /root/decision_transformer_slicing。

### 实验架构
- **环境**：5个网络切片场景（scenario 0-4 训练，scenario 5-9 测试），每场景有3-5个切片（HP高优先级/NHP低优先级），135个PRB资源块
- **分层动作空间（HA）**：Dim0 = codebook条目（11类，控制跨切片PRB分配的不均匀程度），Dims1-5 = intra-slice调度器（RR/PF/MT三选一）
- **关键环境机制**：`urgency_rank`动态映射（按需求排序后分配codebook模板）+ `critical_rescue`（紧急资源重分配）
- **教师策略**：PPO-HA-weighted，5个单场景专家(Agent0-4)分别在scenario 0-4上训练，另有一个联合模型(scenario_0to4)在5个场景上联合训练
- **DT训练数据**：来自5个单场景专家的轨迹，经best-of-N + single-teacher筛选

### 已完成的关键发现

1. **Dim0(codebook)对性能不敏感**：强制替换Dim0为act0/act5/act7，性能变化范围仅0.02-0.04（hp_viol维度）。原因是`urgency_rank`动态映射吸收了codebook的差异。

2. **Dims1-5(intra调度)是性能瓶颈**：强制全部使用RR后，DT性能翻倍提升，接近PPO-HA-weighted水平。DT学到的PF/MT模式在测试场景有害。
   - Intra sweep数据（scenarios [5,6,9], 5eps, seed=0）：
     ```
     DT original:  hp_viol=0.0834, nhp_viol=0.2792
     All RR:        hp_viol=0.0333, nhp_viol=0.1592  ← 大幅提升
     All PF:        hp_viol=0.2703, nhp_viol=0.6417
     ppo_ha_wt:     hp_viol=0.0439, nhp_viol=0.1659
     ```

3. **DT+fixedRR仍不如PPO-HA-weighted**：完整5seeds测试中，dt_fixed_rr在某些场景（特别是S5的nhp_viol）仍落后于ppo_ha_wt。数据在 `data/channel_generality/dt_fixed_rr/metric_json/`。

4. **已尝试但失败的方向**：
   - **多教师数据集**（去掉single-teacher筛选，保留所有agent在所有scenario上的轨迹）：DT的RTG条件化太弱，无法从混合数据中学到正确策略。数据在 `data/channel_generality/dt_multi_teacher/`
   - **Violation-based RTG**（用metric直接构造RTG替代reward）：在单教师数据集上无效（无动作多样性可供条件化）。数据在 `data/channel_generality/dt_viol_rtg/`

5. **根本原因**：
   - 原始reward与实际metric相关性极低（correlation=0.07）
   - 训练数据中每个场景只有一种固化动作（act1占45%，act7占32%），DT无法学到条件化策略
   - PPO-HA-weighted测试时用的是联合模型(act5)，而DT训练数据来自单场景专家(act1/act7)——结构性不对等

### 当前未解决的问题

**dt_fixed_rr在某些测试场景下不如ppo_ha_weighted**。我想搞清楚：

1. **Oracle分析**：对于测试场景5-9中的每一个，到底什么样的(Dim0, Dims1-5)组合能取得最好的成绩？可以通过在测试环境中做exhaustive search来找到——11种codebook × 3种intra策略 = 33种组合，每种跑5个episode就够。这能告诉我们"最优策略"的上界在哪里。

2. **PPO-HA-weighted到底做对了什么**：在测试时录制PPO-HA-weighted的逐步动作(Dim0和Dims1-5)，和DT+fixedRR对比，找出具体哪些时间步的决策差异导致了性能差距。

3. **能否通过训练场景学到测试场景的最优策略**：如果Oracle分析发现最优策略是某个特定组合（如act5+RR），那问题就变成"如何让DT学会选act5"。如果最优策略是场景相关的（不同测试场景需要不同组合），那需要DT具备场景识别+策略切换能力。

### 关键文件路径
- 环境配置：`conf/environment/env_ha.yaml`
- DT测试代码（已有force_dim0/force_intra/use_viol_rtg支持）：`src/basic_apis/dt_utils/test.py`
- Codebook构造：`src/basic_apis/codebook_utils.py`
- 分层环境：`src/basic_apis/ppo/ppo_ha_weighted/hierarchical_slicing_env.py`
- 场景切片配置：`data/static/association/ep_{0-9}.npz`
- 各方法测试结果：`data/channel_generality/{dt,dt_fixed_rr,ppo_ha_weighted,ppo_multi}/metric_json/`
- Sweep脚本：`scripts/sweep_dim0.py`, `scripts/sweep_intra.py`
- 实验手册：`experiment_manual_todo_by_figure.md`

请先做Oracle分析（第1点），然后基于结果决定后续方向。
