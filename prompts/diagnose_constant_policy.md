# Prompt：诊断 PPO 在 EnvV2 中常数策略问题

## 核心问题

在 EnvV2（`src/basic_apis/dt_v2/env_v2.py`）中，PPO 在所有条件下都收敛到**常数策略**（同一个固定动作应对所有状态）。这发生在：
- 单场景训练（S0-S4 每个都是常数）
- 跨场景训练（S0-S4 联合训练，所有场景同一个常数）
- 三种不同 reward 设计（violation-only / margin bonus / flat bonus）

**需要确定**：是 obs→action 映射本身不存在（动作空间或 obs 空间设计问题），还是映射存在但 PPO 学不出来（优化/探索问题）？

## 背景

### 动作空间
- `MultiDiscrete([11, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3])`，共 11 个子动作
- `action[0]`：codebook 索引（11 个 inter-slice quota pattern）
- `action[1:6]`：ordering（5 个值，通过 argsort 决定切片优先级）
- `action[6:11]`：intra-slice scheduler（每切片 3 种调度器）
- 组合总数 ≈ 835 万

### 观测空间（Dict）
- `inter_feat`: (5, 8) — 每切片 8 维特征（priority, drift, traffic, alloc_ratio, spectral_eff, demand_pressure, drop_rate, n_users）
- `intra_feat`: (25, 7) — 每用户 7 维特征（buffer, priority, drift, csi, hol_delay, spectral_eff, thr_ratio）
- `global_feat`: (2,) — timestep 归一化 + 总分配比

### 旧环境对比
旧环境（`HierarchicalSlicingEnv`）动作空间仅 `[11] + [3]*5 = 2673` 种组合，ordering 由 `urgency_rank` heuristic 自动计算。旧 PPO **也收敛到常数策略**。

### 关键参考
- 完整实验记录：`docs/migrations/20260331_1417_PPO常数策略诊断与reward实验.md`
- EnvV2 源码：`src/basic_apis/dt_v2/env_v2.py`
- 旧环境源码：`src/basic_apis/ppo/ppo_ha_weighted/hierarchical_slicing_env.py`
- 特征提取器：`src/basic_apis/ppo/ppo_ha_weighted/agent_hierarchical.py`（HierarchicalAttentionExtractor）
- PPO 训练脚本：`src/basic_apis/dt_v2/train_ppo_v2.py`

---

## 诊断实验（按顺序执行）

### 实验 A：obs 的时序变异性分析

**目标**：确认 obs 在一个 episode 内是否有足够的变化来支撑状态依赖策略。

**做法**：
1. 在 S0 上用随机策略跑 1 个 episode（1000 步），记录每步的完整 obs
2. 计算每个 obs 维度的时间序列统计量：均值、标准差、变异系数（CV = std/mean）
3. 对 `inter_feat` 的 8 个维度分别分析：哪些维度几乎不变（CV < 0.1），哪些有显著变化
4. 可视化 3-4 个关键维度的时间序列（如 drift, traffic, demand_pressure）

**预期结果**：
- 如果大部分维度 CV < 0.1 → obs 空间信息量不足，策略自然退化为常数
- 如果多个维度 CV > 0.3 → obs 有足够变化，问题在学习端

### 实验 B：最优动作是否真的随状态变化？

**目标**：确认对于同一场景的不同时间步，最优动作是否不同。

**做法**：
1. 在 S0 上用 oracle 方式测试：每个时间步 **穷举**（或采样）多个 codebook + ordering 组合
2. 用 1-step rollout 评估每个动作的即时 reward
3. 记录每步的最优动作和次优动作
4. 统计：最优 codebook 是否在不同时间步发生变化？ordering 呢？

注意：完整穷举 835 万种不可行，但可以：
- 固定 ordering 为 urgency_rank，只穷举 11 种 codebook × 3^5=243 种 intra = 2673 种
- 或固定 intra 为全 0，穷举 11 × 120（排列）= 1320 种

**预期结果**：
- 如果最优动作在时间步间几乎不变 → 常数策略接近最优，设计空间无法改善
- 如果最优动作显著变化 → PPO 确实没学出该学的映射

### 实验 C：特征提取器输出分析

**目标**：检查 HierarchicalAttentionExtractor 是否能区分不同状态。

**做法**：
1. 加载训好的模型（`dt_v2_flat_bonus/ppo_s0/final_model.zip` 或 `dt_v2_per_scenario/ppo_s0/best_model/best_model.zip`）
2. 对 1 个 episode 的连续 obs，提取 features_extractor 的输出（256 维）
3. 计算：(a) 不同时间步 feature 向量的余弦相似度分布 (b) PCA 降维后的轨迹可视化
4. 如果所有时间步的 feature 高度相似（cos_sim > 0.99）→ 提取器把信息压缩掉了

### 实验 D：PPO 超参消融（如果 A-C 表明映射存在且提取器正常）

尝试以下改动（每次只改一个）：
1. **增大 ent_coef**：0.01 → 0.05 或 0.1，强制探索
2. **增大 n_steps**：512 → 2048（每次 rollout 覆盖 2 个完整 episode，减少 on-policy 偏差）
3. **简化动作空间**：去掉 ordering（固定 urgency_rank），只学 codebook + intra（2673 种组合）
4. **简化 obs**：去掉 intra_feat，只用 inter_feat + global_feat（大幅减小输入维度）

---

## 决策树

```
实验 A: obs 变异性
├── CV < 0.1 大部分维度 → obs 信息不足
│   → 增加 obs 维度（场景 ID、历史 drift 变化率、channel 变化趋势）
│   → 或确认"常数策略即最优"
│
└── CV > 0.3 多个维度 → obs 有变化
    │
    实验 B: 最优动作是否变化
    ├── 最优动作几乎不变 → 常数策略接近最优
    │   → 动作空间设计无法带来 state-dependent 收益
    │   → 考虑更细粒度的动作（如连续动作空间）
    │
    └── 最优动作显著变化 → 存在 state-dependent 最优策略
        │
        实验 C: 特征提取器
        ├── feature 高度相似 → 提取器丢失信息
        │   → 简化 obs + 用 MLP 替代 Attention
        │
        └── feature 有区分度 → PPO 优化问题
            │
            实验 D: 超参消融
            → ent_coef / n_steps / action space 简化
```

## 成功标准

如果诊断完成后能回答以下问题，任务即成功：
1. 在 EnvV2 中，最优策略是否真的需要 state-dependent？（实验 B 回答）
2. 如果需要，瓶颈在哪一层？obs 信息 / 特征提取 / 策略优化？（实验 A/C/D 回答）
3. 推荐的下一步改进方向是什么？（基于诊断结论）
