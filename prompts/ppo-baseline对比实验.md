任务：PPO-MLP (dt_v2) 决策瓶颈诊断
背景
PPO-MLP 单场景专家（Mean comb=0.9323）比 PPO-baseline（Mean comb=0.1055）差约 9 倍。两者共享同一仿真器底层，主要差异在动作空间设计。本会话的目标是精确定量 PPO-MLP 的性能瓶颈来自哪些设计选择。

已确认的架构差异（来自源码对比）
维度	PPO-baseline	PPO-MLP
Inter-slice	连续 5D → scores_to_rbs（RBG 级）	11-codebook + 排序键
排序	观测侧启发式 sort_slices（按 ues*traffic 升序）	动作的一部分（5×Discrete(5)）
观测 drift	3 维独立 (thr/rel/lat)	1 维均值 slice_drift_mean
PRB 映射	2 阶段（分配 → 切片内）	3 阶段（Rescue → 切片内 → Sweep）
Intra	Discrete(3) RR/PF/MT	同
需要执行的诊断实验（按优先级）
实验 1：Codebook 选择分布 + 实际分配比例
对 S0–S4 各场景专家（data/channel_generality/dt_v2_mlp_ent05/ppo_s{0..4}/best_model/best_model.zip），deterministic rollout 20 episode：

# 每步记录：
action[0]  # codebook index
inter_quotas  # 各切片实际 PRB 数（需要经过 codebook + ordering 解码）
输出：

各场景的 codebook index 频率直方图
各切片实际 PRB 配额比例的 mean/std
与 PPO-baseline 对标：PPO-baseline 在对应场景的配额比例分布（来自另一台服务器，等我提供）
实验 2：排序动作的稳定性与正确性
每步记录 action[1:6]，转换为 np.argsort(order_actions)[::-1] 得到服务顺序：

统计每个场景中 Top-3 最常出现的排序
对比 PPO-baseline 的 sort_slices 启发式（按 ues_per_slice * traffic_req 升序），看两者是否趋同
消融实验：将 PPO-MLP 的 set_forced_ordering 设为 PPO-baseline 的 sort_slices 启发式结果，评测 20 episode，看 comb 是否改善
这需要在 eval_ppo_v2_per_scenario_expert.py 基础上做修改：每步从环境状态中算出 sort_slices 顺序，然后用 env.set_forced_ordering() 覆盖
实验 3：Intra-slice 调度器选择分布
统计 action[6:11] 每个切片选 RR(0)/PF(1)/MT(2) 的频率。输出格式：

| 场景 | 切片 | HP? | RR% | PF% | MT% |
实验 4：S2 step-level 违约溯源（comb=1.5011，最差场景）
跑 1 个 episode，每步记录：

inter_mode + 经 ordering 后各切片 PRB 配额
各切片的 intent_drift（(5, 5, 3) 矩阵中的 min 值）
哪些切片在违约（drift<0），是 HP 还是 NHP
last_rescue_rbs（Critical Rescue 抢了多少 PRB）
分析目标：违约是因为 codebook 给的 PRB 不够，还是 ordering 给错了切片，还是 intra 调度选错了？

模型路径
data/channel_generality/dt_v2_mlp_ent05/ppo_s{0,1,2,3,4}/best_model/best_model.zip
关键代码位置
环境：src/basic_apis/dt_v2/env_v2.py → HierarchicalSlicingEnvV2
Codebook：src/basic_apis/codebook_utils.py → build_dirichlet_inter_quota_codebook
排序逻辑：env_v2.py L49-68
评测参考：scripts/eval_ppo_v2_per_scenario_expert.py
sort_slices 启发式（对标用）：src/basic_apis/ppo/ppo_baseline/agent_ib_sched.py L516-535
11 条 Codebook 模板（已确认）
idx  0: [27,27,27,27,27]  均匀
idx  1: [30,28,27,26,24]
idx  2: [31,29,27,25,23]
idx  3: [33,29,27,24,22]
idx  4: [36,30,26,23,20]
idx  5: [39,31,26,22,17]
idx  6: [44,32,25,19,15]
idx  7: [51,32,22,17,13]
idx  8: [60,31,19,14,11]
idx  9: [70,28,16,11,10]
idx 10: [79,23,13,10,10]
输出要求
四个实验的结果表，重点回答：

Codebook 利用率如何？是否只用了少数几个？
排序是否与 sort_slices 启发式一致？替换后性能变化多大？
Intra 调度偏好是什么？
S2 的违约根因是什么？


任务：PPO-baseline 单场景专家的决策分布数据采集
背景
我在另一台服务器上已完成 PPO-MLP 和 PPO-baseline 的源码交叉对比。现在需要你帮我采集 PPO-baseline 单场景专家（S0–S4）的决策分布数据，用于与 PPO-MLP 的决策对标。

PPO-baseline 架构要点（已确认）
Multi-agent：player_0（inter）+ player_1..5（intra）
Inter 动作：Box(-1, 1, shape=(5,))，解码流程：
action["player_0"] 经 sorted_slices 重映射
不活跃切片位置置为 -1
scores_to_rbs(action, total_rbs=27, association) → 每切片 RBG 数 × 5 = PRB 数
公式：PRB_ratio[s] = (action[s]+1) / sum(action+1)
Intra 动作：player_{s+1} = Discrete(3), 0=RR, 1=PF, 2=MT
排序：sort_slices 按 ues_per_slice * traffic_req 升序，观测和动作解码都用此排序
代码位置：src/basic_apis/ppo/ppo_baseline/agent_ib_sched.py
需要采集的数据（S0–S4 各自的单场景专家）
数据 1：Inter-slice 分配比例分布
对 S0–S4 每个专家，加载 best checkpoint，在对应场景上 deterministic rollout 20 episode，记录每步：

# 获取 player_0 的 raw action（sorted_slices 映射之前或之后均可，但需标注）
raw_action = action["player_0"]  # shape (5,)
# 计算实际 PRB 比例（unsorted，按真实切片 ID 顺序）
# 需要考虑 sorted_slices 的逆映射
# 最终我需要的是：slice_0 分到 X%, slice_1 分到 Y%, ...（按切片原始编号）
输出格式（按切片原始编号，不是 sorted 后的位置）：

## 场景 S0
| 切片 | mean_prb_ratio | std | min | max | median |
|------|---------------|-----|-----|-----|--------|
| 0    |               |     |     |     |        |
| 1    |               |     |     |     |        |
| 2    |               |     |     |     |        |
| 3    |               |     |     |     |        |
| 4    |               |     |     |     |        |
数据 2：分配比例的时序轨迹（任选 1 个 episode）
对 S0 场景，选一个 episode，输出所有 1000 步的 5 切片 PRB 比例时序。如果数据量太大，可以每 10 步采样一次（100 个点）。

目的：看分配比例是固定不变、缓慢漂移、还是剧烈波动。PPO-MLP 的 codebook 是逐步独立选择，如果 PPO-baseline 的比例几乎不变，说明一个固定的好 codebook 就够了。

数据 3：Intra-slice 调度器选择分布
| 场景 | 切片 | RR% | PF% | MT% |
|------|------|-----|-----|-----|
| S0   | 0    |     |     |     |
| ...  | ...  |     |     |     |
数据 4：sort_slices 排序结果
每步的 self.sorted_slices 值（由 sort_slices 函数返回），统计最常见的排序和频率。

## 场景 S0
最常见排序 Top 3:
1. [x, x, x, x, x] — XX%
2. [x, x, x, x, x] — XX%
3. [x, x, x, x, x] — XX%
数据 5：分配比例的聚类分析
把所有步的 5 维 PRB 比例向量做 K-Means (k=5,10,15)，看几个 cluster 能覆盖 90% 的数据。

这直接回答：PPO-baseline 的连续动作在实际使用中是否退化为少数几种模式？

如果 5 个 cluster 就能覆盖 90%，说明 PPO-MLP 的 11-codebook 在理论上足够（可能需要调整模板值）
如果需要 15+ 个 cluster，说明连续性确实是核心优势
模型路径
data/channel_generality/ppo_baseline/checkpoints/scenario_{0,1,2,3,4}/
（具体格式可能是 Ray checkpoint 目录）

输出要求
请把以上 5 组数据按结构化格式返回。重点数据是 数据 1（分配比例统计）和 数据 5（聚类分析），这两组直接决定了 PPO-MLP 的 codebook 是否是性能瓶颈。


> ## 任务：PPO-baseline vs PPO-MLP 决策对标结果汇合与改进方案制定
>
> ### 背景
>
> PPO-MLP 单场景专家（Mean comb=0.9323）比 PPO-baseline（Mean comb=0.1055）差约 9 倍。我们已经在两台服务器上分别完成了决策分布诊断，现在需要汇合结果，定位根因并制定改进方案。
>
> ### 已完成的源码级架构对比结论
>
> | 优先级 | 嫌疑因素 | 说明 |
> |-------|---------|------|
> | 1 | **Codebook 精度不足** | PPO-MLP 只有 11 个固定 PRB 模板（均匀 27:27:27:27:27 到极端 79:23:13:10:10），相邻模板跳跃 3-7%；PPO-baseline 用连续 5D `Box(-1,1)` + `scores_to_rbs` 可输出任意比例 |
> | 2 | **排序作为动作 vs 观测侧启发式** | PPO-MLP 让策略学 5 个 Discrete(5) 排序键（120 种排列），PPO-baseline 用 `sort_slices`（按 `ues*traffic` 升序）固定在观测侧 |
> | 3 | **观测 drift 压缩** | PPO-baseline 有 3 维独立 intent_drift（thr/rel/lat），PPO-MLP 压缩为 1 维 `slice_drift_mean` |
> | 4 | **三阶段 PRB 映射干扰** | PPO-MLP 有 Critical Rescue + Global Sweep 覆写分配，PPO-baseline 只有分配→切片内两步 |
> | 5 | Intra-slice 调度器 | 两者完全相同（RR/PF/MT），不是差距来源 |
>
> ### 11 条 Codebook 模板（PPO-MLP 使用）
>
> ```
> idx  0: [27,27,27,27,27]  ratio=[0.200, 0.200, 0.200, 0.200, 0.200]
> idx  1: [30,28,27,26,24]  ratio=[0.222, 0.207, 0.200, 0.193, 0.178]
> idx  2: [31,29,27,25,23]  ratio=[0.230, 0.215, 0.200, 0.185, 0.170]
> idx  3: [33,29,27,24,22]  ratio=[0.244, 0.215, 0.200, 0.178, 0.163]
> idx  4: [36,30,26,23,20]  ratio=[0.267, 0.222, 0.193, 0.170, 0.148]
> idx  5: [39,31,26,22,17]  ratio=[0.289, 0.230, 0.193, 0.163, 0.126]
> idx  6: [44,32,25,19,15]  ratio=[0.326, 0.237, 0.185, 0.141, 0.111]
> idx  7: [51,32,22,17,13]  ratio=[0.378, 0.237, 0.163, 0.126, 0.096]
> idx  8: [60,31,19,14,11]  ratio=[0.444, 0.230, 0.141, 0.104, 0.081]
> idx  9: [70,28,16,11,10]  ratio=[0.519, 0.207, 0.119, 0.081, 0.074]
> idx 10: [79,23,13,10,10]  ratio=[0.585, 0.170, 0.096, 0.074, 0.074]
> ```
>
> ### 你需要汇合的两组数据
>
> 下面会把两台服务器上的结果贴给你。请先阅读完所有数据，再开始分析。
>
> ---
>
> #### 数据 A：PPO-MLP 侧诊断结果（来自本机会话）
>
> 包含：
> - **A1**：各场景 codebook index 选择频率直方图
> - **A2**：排序动作稳定性 + `sort_slices` 启发式消融实验结果
> - **A3**：Intra-slice 调度器选择分布
> - **A4**：S2 step-level 违约溯源
>
> 【粘贴 PPO-MLP 诊断结果到此处】
>
> ---
>
> #### 数据 B：PPO-baseline 侧决策分布数据（来自远程服务器）
>
> 包含：
> - **B1**：各场景各切片的 PRB 分配比例统计（mean/std/min/max/median）
> - **B2**：S0 场景某 episode 的分配比例时序轨迹
> - **B3**：Intra-slice 调度器选择分布
> - **B4**：sort_slices 排序结果统计
> - **B5**：分配比例的 K-Means 聚类分析
>
> 【粘贴 PPO-baseline 决策分布数据到此处】
>
> ---
>
> ### 你需要完成的分析（按此顺序）
>
> #### 分析 1：Codebook 覆盖度定量评估
>
> 将 B1（PPO-baseline 实际使用的分配比例）与 11 条 codebook 模板逐一对比：
> - 对每个场景，找出 PPO-baseline 的 mean ratio 向量与哪个 codebook 模板最近（L2 距离或 L1 距离）
> - 计算最近模板与真实分布的偏差（每切片的差值）
> - 结合 B5（聚类分析），回答：**如果我们把 codebook 模板替换为 PPO-baseline 的 K-Means 聚类中心，理论上能缩小多少差距？**
>
> #### 分析 2：排序策略对比
>
> 对比 A2（PPO-MLP 学到的排序分布）与 B4（PPO-baseline 的 `sort_slices` 排序结果）：
> - PPO-MLP 学到的 Top-3 排序是否与 `sort_slices` 一致？
> - A2 的消融实验结果（用 `sort_slices` 替代学习排序后的 comb 变化）量化了排序学习的损失
>
> #### 分析 3：Intra 调度器对比
>
> 对比 A3 和 B3，看两者的 RR/PF/MT 偏好是否一致。如果一致，说明 intra 不是差距来源。
>
> #### 分析 4：分配比例动态性评估
>
> 从 B2（时序轨迹）判断：
> - PPO-baseline 的分配比例在 episode 内是静态、缓慢漂移、还是剧烈波动？
> - 如果几乎不变，说明一个好的静态 codebook + 排序就能逼近 PPO-baseline
> - 如果剧烈波动，说明逐步动态调整是关键，需要更根本的动作空间改造
>
> #### 分析 5：S2 违约根因（结合两侧数据）
>
> 结合 A4（PPO-MLP 在 S2 的 step-level 违约溯源）和 B1（PPO-baseline 在 S2 的分配比例），找出：
> - PPO-MLP 在 S2 上给各切片分了多少 PRB，PPO-baseline 给了多少
> - 违约切片是否恰好是 PPO-MLP 分配不足的那个
>
> ### 最终输出：改进方案（按可行性/收益排序）
>
> 基于以上 5 个分析的结论，输出一份排序后的改进方案清单：
>
> ```
> | 优先级 | 改进方案 | 预期收益 | 实施难度 | 依据 |
> |-------|---------|---------|---------|------|
> | 1     |         |         |         |      |
> | ...   |         |         |         |      |
> ```
>
> 每条改进方案需要包含：
> - 具体的代码改动位置和思路
> - 预期能把 Mean comb 从 0.9323 降低到多少（粗略估计即可）
> - 是否需要重新训练 PPO，还是仅修改推理/映射逻辑
>
> ### 约束
>
> - **不改变 PPO-MLP 的基本框架**（仍然是单 agent、离散动作空间、SB3）
> - 优先考虑不需要重训的改进（如替换 codebook、改排序启发式）
> - 需要重训的改进作为第二优先级