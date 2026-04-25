# Research Plan v5: Evolutionary Reward Synthesis for Wireless RL

**目标期刊**：IEEE JSAC（优先，AI for Communications / O-RAN 专题）/ TWC（备选）
**预计周期**：22–24 周（与 MEC Offloading 并行）
**版本说明**：v4 + 开源代码复用路径 + 实现细节收紧 + baseline 措辞防御性修订。核心原则是**高 ROI 的实现折扣**（复用开源物理层/统计工具）+ **论文可防御性**（hybrid baseline 显式声明）。

---

## v5 相对 v4 的变更摘要

| 变更 | 来源 | 影响 |
|---|---|---|
| Power Control 仿真器**抽取 Nasir & Guo Asilomar 2020 代码的物理层**（不整仓合并，只要 `project_backend.py` + `get_benchmarks.py`） | 开源策略评审 | Phase 0 工作量 1 周 → 2–3 天 |
| **Baseline reward 重定位为 "Nasir backbone + QoS + fairness 混合式"**（而非 "directly cite"） | 代码审查暴露原文无 QoS/fairness | 论文 §2.2 + §II-A + §VI 措辞改写；**防 desk-reject 关键** |
| BO baselines 改为 **Optuna `TPESampler(constant_liar=True) + n_jobs=8`** + 独立 GPU slot queue | TPE async 退化为 random 的坑 | 实现细节，Phase 0 新增 1.5 天 |
| CV τ + CI 用 **SciPy `bootstrap(paired=True, vectorized=False)` + BCa + percentile fallback** | DegenerateData 的 small-n 坑 | 实现细节；Phase 1.1 测样数 12 → 15 |
| **拒绝引入 OmniSafe**，保留现有 `src/basic_apis/ppo/ppo_lagrangian/` | pyproject 硬锁依赖冲突 + 离散动作支持弱 | Phase 3 Lagrangian 基线不换框架 |
| 新增 **LagrangianPPO toy-env 正确性验证**（`CartPole-v1` + 自写 `CostWrapper`） | 防"自实现正确性"被 reviewer 质疑 | Phase 0 新增 0.5 天；supplementary 一节 |
| Phase 1.1 proxy-fidelity **GPU-h 预算从 20 修正为 ~100**（proxy + full 双向都要跑） | v4 估计严重偏低 | 算力账核准；wall time ≈ 19h（8 并发） |
| 新增 §十 "v5 实现注记"（开源代码借用清单 + 依赖声明） | 复现性 + 致谢完整性 | 论文复现性审核加分项 |

### v4 相对 v3 的变更（保留作为历史参考）

| 变更 | 来源 | 影响 |
|---|---|---|
| Proxy 场景交叉验证（两轮轮换） | 评估 P1（有效） | Phase 1 增加 1 周 |
| 约束处理机制显式形式化（新 4.3 节） | 评估 P0（有效） | 方法章节改写，无新实验 |
| 新增 **ERS-structure + BO-params** 对照实验 | 评估 P1（有效） | Phase 4 增加 ~20 GPU-h |
| Jain's fairness 阈值改为自适应 `0.5 + 0.5/K` | 评估 P2（有效） | 仿真器参数改动 |
| 种子数 3→5，补 95% 置信区间 + paired t-test | 评估 P1（有效） | 全程 GPU 时间 +40% |
| 新增 "简化假设声明" 小节 | 评估 P2（部分有效） | 论文 Discussion 改写 |
| **拒绝**：切换到连续 action space | 评估 P3（过度） | 与 v3 算法一致性冲突 |
| **拒绝**：换 OpenAirInterface 仿真器 | 评估 P2（过度） | 成本高，Sun 2018 TWC 先例充分 |
| **部分接受**：WMMSE/FP 作 Power Control 参考点（而非核心对比） | 评估 P0（高估） | Nasir & Guo 已隐含此对比 |

---

## 一、论文定位（基本保留 v3）

### 1.1 核心问题

Wireless RL 中的 reward function 设计面临 **"soft optimization objective + hard constraint satisfaction"** 的根本矛盾：既要最大化连续目标（throughput、sum-rate），又要满足离散硬约束（SLA compliance、per-user QoS）。reward 的函数结构（线性/指数惩罚、是否引入前瞻信号）和参数（权重、阈值）交织在一起，人工试错低效且难跨场景/跨任务复用。

**本文问题：能否用 LLM + 进化搜索自动合成 reward function 的完整代码，使其在结构和参数层面同时优于人工设计和参数优化，并在多个 wireless RL 任务上验证？**

### 1.2 两个正交贡献

**贡献 1（方法论）：Evolutionary Reward Synthesis (ERS)**

Task-agnostic 框架，输入为 reward skeleton + evaluator，输出为 evolved reward Python 代码。核心设计：
- 多场景**交叉验证** fitness 评估 → 跨场景泛化（v4 更新）
- LLM 生成 + 反思机制 → verbal gradient 指导进化
- 岛屿进化 → 种群多样性维护
- **显式约束处理**（4.3 节）→ 硬约束满足率进入 fitness（v4 新增）
- 两个异质 wireless RL 任务验证 → 非 task-specific

与 Eureka/ReEvo 的关键差异：
1. 多场景**交叉验证** joint fitness（Eureka 单任务单场景）
2. 显式 constraint-aware fitness 设计（Eureka 软约束隐式编码）
3. 三层 BO 对比 + **ERS-structure + BO-params 对照实验**定量证明结构创新（v4 新增）
4. 服务于 offline RL pipeline（vs 直接部署）
5. 两个异质 wireless task 验证（vs 单领域）

**贡献 2（领域知识发现）：Transferable Reward Mechanisms**

ERS 发现的 reward 机制可被提取为独立的、可复用的 shaping 组件，嵌入其他 DRL agent 后仍能提升性能。

**关键定位（防 wireless 审稿人攻击）：**
- **不说** "LLM 发现了人类忽略的机制"（审稿人会反驳 buffer awareness 是常识）
- **而说** "ERS 发现的是 **reward-level** 的 constraint awareness，与传统 **scheduling-level** 的 queue-aware design（Lyapunov/backpressure）是不同数学层次的操作"
- 实验证明：ERS-evolved reward + vanilla scheduler > vanilla reward + Lyapunov-style scheduler

### 1.3 叙事策略（三层）

**主线（自动化 + scalability）**：Manual reward design 即使对资深工程师也需数周试错，且难以多场景 joint optimize。ERS 在 X GPU-hours 内自动合成跨场景 robust 的 reward function，在两个异质 wireless RL 任务上均 outperform manual baseline 和 BO-optimized baseline。

**辅线 1（结构 > 参数）**：ERS 的优势来自 reward 结构创新，而非仅参数调优。即使给 BO 更大参数空间（extended template），LLM 进化仍显著更优；且当固定 ERS 发现的**结构**、仅用 BO 搜参数时，性能与完整 ERS 接近，证明核心贡献来自结构发现（v4 新增实验支撑）。

**辅线 2（可复用机制）**：ERS 发现的 reward-level constraint awareness 可提取为 reusable shaping 组件，zero-code-modification 嵌入其他 DRL agent 和其他任务后仍有增益。

### 1.4 差异化定位

| 维度 | FunSearch | ReEvo | Eureka | **本文 (ERS)** |
|---|---|---|---|---|
| 搜索对象 | Priority function | 启发式代码 | Reward function | **Reward function** |
| 应用领域 | 组合数学 | 组合优化 | 机器人控制 | **Wireless RL（两个任务）** |
| 场景泛化 | 单问题 | 单问题 | 单任务 | **多场景交叉验证** |
| 约束处理 | 无 | 无 | 软约束隐式 | **硬约束显式进 fitness** |
| 结构 vs 参数 | 未区分 | 未区分 | 未区分 | **BO baseline + structure-only 对照** |
| 下游用途 | 直接使用 | 直接使用 | RL policy | **Expert → Offline RL pipeline** |
| 机制提取 | 无 | 无 | 无 | **可复用 shaping 组件 + 跨算法 + 跨任务** |

**关于 ReEvo 的明确声明（v4 新增，回应评估关切）**：ReEvo 原始用于组合优化启发式（TSP/CVRP 等），我们借用其**开源代码框架**实现 LLM 进化循环，但核心方法设计（多场景交叉验证 fitness、约束 aware 评估、mechanism 提取协议）均为本文原创，在论文致谢和消融中明确区分。

---

## 二、两个验证任务（基本保留 v3，小改进）

### 2.1 Task 1：O-RAN Network Slicing（主战场）

复用 IDT 论文仿真环境。

- 问题结构：多 slice 资源调度，每 slice 有 SLA 目标
- Reward 难点：最大化 fulfillment margin（soft）+ 零 HP violation（hard）+ 多指标平衡
- 动作空间：离散（template + ordering + intra-slice scheduler）
- RL 算法：PPO-Discrete
- Baseline reward：IDT 论文公式(22)
- 评估指标：HP/NHP violation rate, intent distance, **constraint satisfaction rate**（v4 新增）
- 场景：5 训练 + 5 OOD 测试

### 2.2 Task 2：Multi-Cell Power Control with QoS & Fairness（stress test）

搭建 K-user interference channel 仿真器（**v5 路径**：基于 Nasir & Guo Asilomar 2020 开源代码的物理层，加我们的约束层）。

- 问题结构：K 对发射-接收端共享频谱，joint power allocation
- Reward 难点：最大化 sum-rate（soft）+ per-user minimum rate R_min（hard）+ Jain's fairness（inter-user coupling，RAN slicing 中无直接对应）
- 动作空间：离散功率等级（10 级），与 Task 1 一致
- RL 算法：PPO-Discrete
- 评估指标：average sum-rate, QoS violation rate, Jain's fairness index
- 场景：K ∈ {4, 8, 16} × SNR regime × fading 模型

**v4 改进（保留）**：Jain's fairness 阈值改为 `threshold(K) = 0.5 + 0.5/K`（评估指出固定 0.8 在 K=16 过严）。

**v5 关键修订：Manual Baseline Reward 重定位为 "Hybrid Extension"**

v4 原表述 "直接引用 Nasir & Guo 2019 的 reward" 在代码审查后发现**与事实不符**：
- Nasir & Guo 2019 (JSAC) 和 Asilomar 2020 的 reward 均为 **distributed externality-aware sum-rate**（对邻居造成的 rate 损失作为 penalty），**不含 per-user QoS 阈值**、**不含 fairness 项**
- 我们 v4 problem formulation 要求的 `rate ≥ R_min` 和 Jain's fairness 是**我们自己加的约束**，必须显式声明

**v5 正式表述**（论文 §II-B 和 §III-B 措辞）：

> Our manual baseline **extends** Nasir and Guo's distributed externality-aware reward with two explicit constraint terms:
>
> $$r_k^{\text{manual}} = \underbrace{\log_2(1+\mathrm{SINR}_k) - \sum_{n \in \mathcal{N}(k)} \Delta_n}_{\text{Nasir-Guo backbone}} \underbrace{- \alpha \cdot \mathbb{1}[R_k < R_{\min}]}_{\text{QoS term (ours)}} \underbrace{- \beta \cdot \max(0, \tau(K) - J)}_{\text{Fairness term (ours)}}$$
>
> where $\Delta_n = \log_2(1+\mathrm{SINR}_n^{(\text{if-}k\text{-silent})}) - \log_2(1+\mathrm{SINR}_n^{(\text{actual})})$, $\tau(K) = 0.5 + 0.5/K$, and $J$ is Jain's fairness index. We tune $(\alpha, \beta)$ by grid search on scenario $K=8$ and freeze them for all other $K$. This is a **hybrid baseline**: the Nasir-Guo backbone encodes domain knowledge about inter-link interference externalities; the QoS and fairness terms encode this work's explicit hard-constraint problem statement.

**对 ERS 叙事的连带强化**（防 reviewer 质疑"不公平对比"）：
- Manual baseline 已经**享有与 ERS 相同的约束信号访问权**——只是编码方式不同（手工硬编码 vs LLM 自由搜索结构）
- 对比变成 "**同等约束信号下的结构搜索 vs 手工工程**"，而非 "ERS 用了约束信号，baseline 没用"
- 这直接支撑 §6 风险 5 第 2 层防线（"显式 constraint-aware fitness 不是作弊，是等价条件下的方法对比"）

**WMMSE/FP 的定位（保留 v4）**：仅作为 performance upper bound reference 放在 Discussion，不进主结果表，由 Nasir 仓库 `get_benchmarks.py` 直接提供实现。

### 2.3 两个任务的同构性与差异性（v3 保留，论文 Section III 专门分析）

**同构性（soft-hard coupling）**：

| 维度 | RAN Slicing | Power Control |
|---|---|---|
| Soft objective | Fulfillment margin | Sum-rate |
| Hard constraint | HP violation = 0 | Per-user rate ≥ R_min |
| 耦合方式 | Margin 和 violation 由同一个 allocation 决定 | Rate 和 QoS 由同一个 power vector 决定 |

**差异性（防"两任务太相似"攻击）**：

| 维度 | RAN Slicing | Power Control |
|---|---|---|
| 动作类型 | 组合离散 | 单维离散 |
| 约束层级 | Slice 级（inter-slice 隔离） | User 级（inter-user fairness） |
| 额外难点 | 多指标平衡 | Fairness constraint |
| 物理耦合 | Slice 间通过 RB pool | User 间通过干扰 |

### 2.4 Power Control 仿真器（v5 修订：基于 Nasir 开源代码抽层）

**代码复用策略**（v5 新增）：不从零写，从 [sinannasir/Power-Control-asilomar](https://github.com/sinannasir/Power-Control-asilomar) 仓库**只抽物理层**，丢弃 TF1.x 算法层：

| 原仓库文件 | 处理 | 在本项目中的位置 |
|---|---|---|
| `project_backend.py`（26 KB）| **抽取** Jakes fading + shadowing + path loss + SINR/rate 计算，改写为纯 NumPy | `src/basic_apis/power_control/channel.py` |
| `random_deployment.py` | **抽取** K 对收发端随机部署逻辑 | `src/basic_apis/power_control/deployment.py` |
| `get_benchmarks.py` | **抽取** WMMSE / FP 迭代实现（Discussion 章的 upper bound reference） | `src/basic_apis/power_control/benchmarks.py` |
| `config/deployment/*.json` | **参考**参数取值（K, N, shadow, fd） → 写成 Hydra yaml | `conf/simulation/power_control/*.yaml` |
| `DDPG.py / DQN.py / trainXXX.py / testXXX.py` | **丢弃**，用我们自己的 PPO-Discrete + SB3 栈 | — |
| `simulations/sumrate/policy/*` | 丢弃（不需要预训练权重）| — |
| `requirements.txt`（TF1 + py3.6）| 丢弃 | — |

**为什么选 Asilomar 2020 而非 Spectrum-Power-Allocation**：后者额外引入 M 维子带选择，与我们 "单频段 K-link IC + 离散功率 10 级" 的 v4 建模正交，且会稀释 ERS 的可解释性（动作空间从 "10 级离散功率" 变成 "子带 × 功率 joint"）。Asilomar 2020 正好匹配（DQN 分支本来就是 10 级离散功率实现）。

**环境外壳**（v5 在 Nasir 物理层之上自写的 Gym Env）：

```python
class InterferenceChannelEnv(gym.Env):
    """K-user IC env with discrete power + QoS + Jain's fairness.

    物理层来自 Nasir & Guo Asilomar 2020（src/basic_apis/power_control/channel.py）。
    约束层（QoS threshold + adaptive Jain's fairness threshold）为本项目新增。
    """
    def __init__(self, K=8, P_max=1.0, R_min=1.0, n_power_levels=10):
        self.K = K
        self.P_max = P_max
        self.R_min = R_min
        self.fairness_threshold = 0.5 + 0.5 / K   # v4: 自适应
        self.power_levels = np.linspace(0, P_max, n_power_levels)
        # channel = NasirChannel(...)  # 复用 channel.py

    def step(self, actions):
        powers = self.power_levels[actions]
        sinr = self.channel.compute_sinr(powers)   # Nasir 物理层
        rates = np.log2(1 + sinr)
        sum_rate = float(np.sum(rates))
        qos_violations = float(np.mean(rates < self.R_min))   # ours
        jains_fairness = float(np.sum(rates)**2 / (self.K * np.sum(rates**2) + 1e-12))
        return rates, sum_rate, qos_violations, jains_fairness
```

**关于仿真器真实性（v4 新增声明，v5 保留）**：Sun et al. 2018 (TWC) 和 Nasir & Guo 2019 (JSAC) 均使用 synthetic interference channel；ERS 的贡献在 reward design 方法论层面，不 claim channel modeling 创新。在 Limitations 小节声明此简化假设，并提出 future work 迁移到 Sionna/DeepMIMO。

**v5 新增可信度论据**：复用 Nasir & Guo 官方开源实现的物理层，reviewer 对 "仿真器是否忠实还原原文设定" 的质疑面显著收窄（"我们的仿真器与 Nasir 原文共享同一套 Jakes fading + 随机部署 + SINR 计算代码"）。

---

## 三、方法详细设计

### 3.1 ERS 框架（Task-Agnostic）

```
输入（per task）：
  - reward_skeleton: 函数签名 + 输入变量含义/范围 + 输出范围约束
  - constraint_spec: 硬约束定义（类型、阈值、满足率要求）  # v4 新增
  - evaluator: reward code → RL training → constrained fitness score
  - seed_reward: 人工设计的初始 reward

进化循环：
  for iter in range(max_fe):
    1. 从 island database 采样 top-k reward 代码
    2. 构造 prompt = [问题描述 + skeleton + constraint_spec + top-k 代码
                     + 历史 (fitness, constraint_sat_rate) + 上一代反思]
    3. LLM 生成 pop_size 个新 reward 候选
    4. LLM 反思：分析上一代失败原因（verbal gradient）
    5. 对每个候选：sanity check + 多场景 constrained fitness 评估
    6. 更新 island database（选择 + diversity）

输出：best_reward code + 完整进化历史
```

### 3.2 Output Sanity Check（防 reward hacking）

```python
def safe_evaluate_reward(reward_fn, *args):
    try:
        r = reward_fn(*args)
        if np.isnan(r) or np.isinf(r):
            return None
        r = np.clip(r, -15.0, 15.0)
        return float(r)
    except Exception:
        return None
```

无效输出 → `fitness += 2 * worst_observed_fitness`（动态，而非 magic number）。

### 3.3 多场景**交叉验证** Fitness（v4 关键改动）

**问题**：v3 固定 proxy = [0, 2, 4] 存在 reporting bias。ERS 可能在这三个场景过拟合，在 OOD 场景失败。

**v4 方案：两折交叉验证的 proxy-validation split**：

```python
SCENARIO_SPLITS = [
    {"proxy": [0, 2, 4], "validation": [1, 3]},    # Fold A
    {"proxy": [1, 3, 5], "validation": [0, 2, 4]}, # Fold B
]

def compute_cv_fitness(reward_code, fold):
    proxy_fitness = []
    for scenario in fold["proxy"]:
        agent = train_ppo(reward_code, scenario, steps=proxy_steps)
        metrics = evaluate(agent, scenario, n_seeds=5)  # v4: 3→5

        perf = metrics['throughput'] - 0.5 * metrics['latency']
        sat_rate = metrics['hp_satisfaction_rate']
        # 约束 aware（见 3.4）
        fit = perf * sat_rate - 100.0 * max(0, 1 - sat_rate - 0.05)**2
        proxy_fitness.append(fit)
    return np.mean(proxy_fitness)
```

**执行策略**：主进化跑 Fold A；Phase 1 验证时用 Fold B 的 validation 场景 sanity-check Fold A top-K reward（proxy-to-validation correlation 计算 Kendall's τ，样本 n ≥ 10，报告置信区间）。如果 τ < 0.5，切换到 Fold B 重新进化（timeline 成本已纳入 buffer）。

**Proxy 场景选择（客观）**：

| Task | Fold A proxy | Fold B proxy |
|---|---|---|
| RAN Slicing | [0,2,4]（低-中-高 HP/NHP 比例） | [1,3,5]（覆盖 traffic 强度另一维度） |
| Power Control | [K=4, K=16] | [K=8] + [K=16 w/ different SNR] |

### 3.4 约束处理机制显式形式化（v4 新增章节，回应评估 P0）

**设计原则**：硬约束满足率显式进入 fitness，避免 LLM 过度优化 soft objective 导致 constraint violation。

**Constrained Fitness 形式**：

$$
\text{fitness} = \underbrace{\text{perf}(\pi)}_{\text{soft}} \cdot \underbrace{\text{sat}(\pi)}_{\text{硬约束满足率}} - \lambda \cdot \max(0, \tau^\star - \text{sat}(\pi))^2
$$

其中：
- `perf(π)` 是归一化到 [0, 1] 的性能分数
- `sat(π)` 是硬约束满足率（HP non-violation rate / QoS 满足率），范围 [0, 1]
- `τ★` 是目标满足率（RAN slicing: 0.95; Power Control: 0.90）
- `λ` 是越界惩罚（默认 100）

**与基线约束 RL 的对比（v4 新增小节）**：

| 方法 | 约束编码方式 | 优点 | 缺点 |
|---|---|---|---|
| 手工软惩罚 | reward 里加 `-w * violation` | 实现简单 | w 难调，无满足率保证 |
| CPO / Lagrangian | 约束项进 Lagrangian，λ 对偶更新 | 理论保证收敛 | 训练不稳定，λ 调参敏感 |
| **ERS (本文)** | Fitness 层显式 sat(π) × perf(π) | LLM 可自由生成结构，sat 直接进 fitness | 依赖仿真评估（我们用 proxy 降本） |

**实验（Phase 3 新增）**：额外对比 PPO + **Lagrangian constraint**（文献常用的约束 RL baseline），证明 ERS 的 **reward-level 约束编码** 是否优于 algorithm-level 约束编码。

### 3.5 BO Baselines + 结构对照实验（v4 关键扩展）

**BO-Standard**：固定公式(22)结构，搜 6 个权重（同 v3）。

**BO-Extended**：扩展参数化 template，12 个参数（同 v3），含 buffer/temporal switch。

**ERS-Structure + BO-Params（v4 新增，评估 P1 建议）**：

锁定 ERS 最终发现的**结构**（函数形式、组件关系、嵌套），仅用 BO 搜这些组件里的连续参数。

**四层对比逻辑（v4 升级）**：

- **BO-Standard vs Manual** → 参数优化的价值
- **BO-Extended vs BO-Standard** → 结构扩展（人工）的价值
- **ERS-Structure + BO-Params vs BO-Extended** → ERS 发现的结构相对人工扩展的价值
- **ERS-Full vs ERS-Structure + BO-Params** → LLM 自由结构搜索相对仅参数优化的价值

**若 ERS-Full ≈ ERS-Structure + BO-Params**：claim 降级为 "LLM 在**结构空间发现**上有独立价值，即使给人类充分参数优化仍有显著 gap" — 这仍然是强贡献。
**若 ERS-Full > ERS-Structure + BO-Params**：claim 变为 "LLM 的价值同时体现在结构和参数的 joint 搜索" — 顶级结果。

这个实验能正面排除 "ERS 只是优化算法效率高" 的 alternative explanation。

### 3.6 Mechanism Transfer 实验设计（v3 保留）

**Step 1**：从 best evolved reward 中提取独立函数模块。方法：AST 解析 + LLM-assisted 模块边界识别，人工 verify。

**Step 2**：嵌入其他 agent（同任务，跨算法）：

| 配置 | 说明 |
|---|---|
| Manual | 公式(22) |
| Manual + Buffer Component | 公式(22) + ERS buffer shaping |
| Manual + Temporal Component | 公式(22) + ERS temporal shaping |
| Manual + Both | 两者 |
| Full Evolved | ERS 完整 reward |

× 两种算法（PPO-Discrete + CQL）→ algorithm-agnostic 验证。

**Step 3**：跨任务 **code-level** transfer（不是概念类比）：

```python
# RAN slicing 的 buffer 前瞻组件：buffer_occ > 0.9 → exp penalty
# 适配 Power Control：rate_margin < 0.1 → exp penalty
#   rate_margin = (rate - R_min) / R_min，类比 buffer 占用的 violation 前兆
```

**Step 4**：**Reward-level vs Scheduler-level 正面对比**（核心防攻击实验）：

| 配置 | Reward | Scheduler |
|---|---|---|
| A | Manual | Vanilla |
| B | Manual | Lyapunov/backpressure |
| C | ERS-evolved | Vanilla |
| D | ERS-evolved | Lyapunov/backpressure |

预期结果：C > B → 证明 reward-level awareness 独立于 scheduler design；D ≥ C → 两种 awareness 可叠加，非 zero-sum。

---

## 四、完整实验矩阵

### Phase 0：基础设施（第 1–2 周，v5 修订）

**v5 核心变化**：复用开源代码 + 保留现有 LagrangianPPO 自实现，整体从"从零 15 天"压到"~5.25 天"。

| 任务 | 产出 | 耗时 | v5 落地细节 |
|---|---|---|---|
| **Power Control 仿真器（抽 Nasir 物理层）** | `src/basic_apis/power_control/{channel, deployment, benchmarks, env}.py` + 对应 hydra yaml | **2–3 天** | 只抽 `project_backend.py` + `random_deployment.py` + `get_benchmarks.py`；外壳层自写 QoS + 自适应 fairness |
| **BO-Standard + BO-Extended（Optuna）** | `src/basic_apis/bo/*` + 6/12 参数 yaml | **1.5 天** | `TPESampler(constant_liar=True, seed=42)` + `n_jobs=8` + **独立**的 GPU slot queue（不复用 ReEvo 队列；避免 TPE async 退化为 random） |
| **约束 aware fitness 评估器**（v4→v5 实现收紧） | `src/reevo/problems/*_reward/eval.py` 加 `sat × perf − λ·max(0, τ⋆−sat)²` | **1 天** | λ 初始 100，量纲归一到 perf ∈ [0,1]、sat ∈ [0,1]；当前 `eval.py:155` 的简单公式需同步更新 |
| **多场景 CV split 基础设施**（v4→v5 实现收紧） | Fold A/B 切换 + Kendall τ + BCa CI | **1 天** | SciPy `bootstrap((x,y), tau_stat, paired=True, vectorized=False, method='BCa')`；DegenerateData 时自动 fallback `method='percentile'`；`IDT_FOLD` 环境变量 + `FOLD_SPLITS` dict |
| **LagrangianPPO toy-env 正确性验证**（v5 新增） | `tests/lagrangian_pp_toy_cartpole.py` + 结果图进 supplementary | **0.5 天** | `CartPole-v1` + 20 行 `CostWrapper`（cost = `1{|x| > 1.0}`）；期望 λ 收敛、`E[cost] → cost_limit`；**零新依赖**，不引 safety-gymnasium |
| **论文措辞同步修订**（v5 新增） | v4 plan §2.2 + §II-A + §VI 改 "directly cite" → "hybrid extension" | **0.25 天** | 见 §2.2 hybrid baseline formula；防 desk-reject 关键 |
| Power Control 的 ReEvo 适配 | prompt + seed + eval | **2 天** | 仿 `src/reevo/problems/idt_reward/` 三文件模板（DrEureka 同构）；复用 IDT 的进化循环不改动 |
| **P0 总计** | | **~8.25 天**（其中 P0 阻塞项 5.25 天 + ReEvo 适配 2 天 + 约束 fitness 1 天）| Phase 0 的 1–2 周预算下还留 4 天余量 |

**v5 拒绝清单**（不做）：
- ❌ 引入 OmniSafe 换掉现有 LagrangianPPO（依赖硬锁冲突 + 离散动作支持弱 + 方法论无增益）
- ❌ 从零写 Power Control 物理层（Nasir 开源代码成熟，4–5 天节省）
- ❌ Optuna 复用 ReEvo slot queue（TPE sampler 依赖 sequential dependency，async 会退化为 random）

### Phase 1：验证性实验（第 2–5 周）— Go/No-Go 关卡（v4 加长 1 周，v5 预算修订）

| 实验 | 目的 | 方法 | Go 条件 |
|---|---|---|---|
| 1.1 **CV Proxy fidelity**（v5：n 扩大 + 实现细节收紧） | 两折 proxy-validation correlation | **15** 个 reward × Fold A/B × (proxy PPO + full PPO) × 5 seeds，Kendall's τ + BCa CI | τ > 0.5，95% CI 下界 > 0.3 |
| 1.2 Fitness 可区分度 | 多场景 fitness 能否区分 | 10 random + manual reward | 明显 spread |
| 1.3 约束编码有效性（v4） | 含/不含 sat(π) 的 fitness 对比 | 20 random reward, 观察 sat 分布 | sat>0.9 比例翻倍 |
| 1.4 Power Control pilot | ReEvo 能否适配 | 5 轮进化端到端 | 无报错 |

**v5 Phase 1.1 算力预算修订（v4 低估）**：

| 项 | 单次耗时（实测 fps=82）| 次数 | GPU-h |
|---|---|---|---|
| Full PPO（400K 步）| ~1.35 h | 15 reward × 5 场景 × 5 seed × 2 fold = 750 | 1012.5 |
| Proxy PPO（200K 步，`proxy_train_ratio=0.5`）| ~0.67 h | 同上 750 | 502.5 |
| **减损**：baseline reward 的 full-run 与 Phase 2.1 共享 | | | −~50 |
| **实际增量** | | | **~100 GPU-h** |
| **Wall time（8 并发，4 GPU × per_gpu=2）** | | | **~19 h（一整夜）** |

（v4 原估 20 GPU-h 严重偏低；v5 需在 §6 风险 8 的预算表里把此数字代入）

**v5 实现细节（直接写给 code-implementer）**：

```python
# Fold 切换：src/reevo/problems/*_reward/eval.py 新增 IDT_FOLD 环境变量
FOLD_SPLITS = {
    "A": {"proxy": [0, 2, 4], "validation": [1, 3]},
    "B": {"proxy": [1, 3, 5], "validation": [0, 2, 4]},
}

# Kendall τ + BCa CI（scripts/analyze_cv_fidelity.py）
from scipy.stats import kendalltau, bootstrap
def tau_statistic(x, y):
    return kendalltau(x, y).statistic

try:
    res = bootstrap(
        (proxy_scores, full_scores), tau_statistic,
        paired=True, vectorized=False,
        n_resamples=9999, method='BCa', rng=42,
    )
except Warning:   # DegenerateDataWarning at small n + extreme τ
    res = bootstrap(
        (proxy_scores, full_scores), tau_statistic,
        paired=True, vectorized=False,
        n_resamples=9999, method='percentile', rng=42,
    )
tau, ci_low, ci_high = tau_statistic(proxy_scores, full_scores), res.confidence_interval.low, res.confidence_interval.high
```

**v5 设计提醒**：15 个 reward 要**刻意选有方差的**（baseline、random 1–2、evolved top-3、evolved mid-3 等），避免全 fail / 全 rank 平手导致 BCa accelerator 退化。

**Fallback**（v4 保留）：
1. τ < 0.5 → proxy_train_ratio 0.5 → 0.7
2. 仍不够 → surrogate-assisted evolution
3. 时间影响：方案 1 +2 周，方案 2 +3 周

**第 5 周末决策点**：Phase 1 结果 → proxy 策略确认 → 修订 GPU 预算。

### Phase 2：核心进化搜索（第 5–9 周）

| 实验 | Task | 配置 | GPU 时间 |
|---|---|---|---|
| 2.1 多场景 CV 进化（Fold A） | RAN Slicing | max_fe=40, pop=4, proxy [0,2,4] | ~110h（种子 5） |
| 2.2 多场景 CV 进化（Fold A） | Power Control | max_fe=30, pop=4, proxy [K=4,16] | ~45h |
| 2.3 BO-Standard | RAN Slicing | 100 trials, TPE, 6 params | ~55h |
| 2.4 BO-Extended | RAN Slicing | 100 trials, TPE, 12 params | ~55h |
| 2.5 BO-Standard | Power Control | 80 trials | ~20h |
| 2.6 BO-Extended | Power Control | 80 trials | ~20h |
| 2.7 Random Search | RAN Slicing | 50 random | ~25h |
| **2.8 ERS-Structure + BO-Params**（v4） | RAN Slicing | 固定 ERS 结构，BO 搜参数 80 trials | ~20h |

**第 7 周末 fidelity recheck**：取 ERS top-3 + BO top-3 跑 Fold B validation，再次确认 proxy-full correlation。

### Phase 3：Full Pipeline 验证（第 9–12 周）

**RAN Slicing 主结果（v4 扩展含 CI 和 p-value）**：

| Method | Reward Source | HP Viol↓ (95% CI) | Constraint Sat↑ | p-value vs Manual |
|---|---|---|---|---|
| PPO-Discrete | Manual | ... | ... | — |
| PPO-Discrete | BO-Standard | ... | ... | paired t-test |
| PPO-Discrete | BO-Extended | ... | ... | paired t-test |
| PPO-Discrete | **ERS-Struct + BO-Params**（v4） | ... | ... | paired t-test |
| PPO-Discrete | **ERS-Full** | ... | ... | paired t-test |
| PPO-Discrete | **Lagrangian**（v4；v5：实现自 `src/basic_apis/ppo/ppo_lagrangian/`，supplementary 有 toy-env validation） | ... | ... | paired t-test |
| IDT | Manual / BO-Std / BO-Ext / ERS-Full | ... | ... | ... |
| CQL-Discrete | Manual / ERS-Full | ... | ... | ... |

**v5 关于 Lagrangian 实现的声明**（论文 §V-A 加一段）：

> We implement Lagrangian PPO following Zangooei et al. 2024 (JSAC) [Ref] on top of Stable-Baselines3 PPO, with dual-variable update $\lambda \leftarrow [\lambda + \eta(E[\text{cost}] - d)]_+$. Implementation correctness is validated on a toy constrained CartPole task (supplementary §X.Y), where $\lambda$ converges and $E[\text{cost}]$ stabilizes near the prescribed $\text{cost\_limit}$. We deliberately did not adopt OmniSafe [Ref] because (a) our action space is discrete whereas OmniSafe's benchmarks are primarily continuous, and (b) our SB3-based implementation is lightweight and fully integrated with our existing PPO training pipeline.

**Power Control 主结果**（类似结构，含 Sum-Rate / QoS Viol / Jain's Fairness 三列）。

**统计规范（v4 保留）**：所有实验 n=5 种子，报告 mean ± 95% CI（bootstrap 1000 resamples），成对对比用 Wilcoxon signed-rank 或 paired t-test，显著性 α=0.05。

### Phase 4：消融与深度分析（第 12–15 周）

| # | 实验 | 目的 | Task |
|---|---|---|---|
| 4.1 | 机制消融 | 各 evolved 组件独立贡献 | 两个 |
| 4.2 | Mechanism transfer（同任务跨算法） | PPO + CQL × Manual+组件 | RAN |
| 4.3 | Mechanism transfer（跨任务 code-level） | RAN 组件适配到 PC | 跨任务 |
| 4.4 | **Reward-level vs Scheduler-level**（核心防攻击） | 四组 2×2 | RAN |
| 4.5 | **Proxy CV 稳健性**（v4） | Fold A vs Fold B 最终性能对比 | RAN |
| 4.6 | 进化过程可视化 | 进化曲线、机制时间线 | 两个 |
| 4.7 | Sensitivity: LLM backend | DeepSeek V3 / GPT-4o / GPT-4o-mini | RAN |
| 4.8 | Sensitivity: 进化轮数 | max_fe=[10,20,30,40,50] | RAN |
| 4.9 | 计算开销 | GPU-h + API $ vs 人工时间 | 两个 |

### Phase 5：论文写作（第 15–22 周，7 周含 buffer）

| 周 | 任务 |
|---|---|
| 15–16 | 初稿：Intro / System Model / Method |
| 16–17 | 初稿：Experiments / Analysis |
| 17–18 | 内部审阅 + Related Work |
| 18–19 | 修改 + 图表精修 |
| 19–20 | 语言打磨 + 格式检查 |
| 20–22 | 最终审阅 + 投稿 |

---

## 五、论文结构（v3 基础 + v4 小调整）

### 标题

*Evolutionary Reward Synthesis for Wireless Reinforcement Learning: Constraint-Aware Design, Cross-Task Validation, and Transferable Mechanisms*

（v4 将 "Automatic" 换成 "Constraint-Aware"，突出约束处理创新点）

### 结构大纲（14–16 页双栏）

**I. Introduction（2 页）**
- Wireless RL 中 reward design 是被低估的性能瓶颈
- 三个子问题：耗时、多场景难 joint optimize、不可跨任务复用
- LLM + 进化搜索范式（FunSearch/ReEvo/Eureka）简介
- **本文三个技术创新**：多场景 CV fitness / 显式约束编码 / 结构-参数分离验证（v4 新增）
- 两个正交贡献 + 主要结果摘要

**II. Related Work（1.5 页）**
- A. Reward Design in Wireless RL（RAN slicing + power control，引 Lyapunov/backpressure 预防攻击）。**v5 新增一句**："Our Power Control baseline adopts Nasir & Guo's distributed externality-aware reward structure [Ref] and extends it with explicit QoS and fairness constraint terms, which are not addressed in the original sum-rate-only formulation"（关键：让 reviewer 在翻到 §III-B 看到公式之前，已经被告知 baseline 是 extension）
- B. Automated Reward Design（shaping 理论、IRL、Eureka）
- C. LLM for Algorithm Design（FunSearch, ReEvo）
- D. Constrained RL（CPO, Lagrangian methods）（v4 新增子节）
- E. 定位

**III. System Models and Problem Statement（2.5 页）**
- A. Task 1: O-RAN Network Slicing
- B. Task 2: Multi-Cell Power Control with QoS & Fairness
- C. 同构性与差异性分析（0.5 页，generalization claim 根基）
- D. 统一表述：Reward Function Design as Constrained Program Search（v4 更新）

**IV. Evolutionary Reward Synthesis Framework（3.5 页，v4 加 0.5 页）**
- A. Reward Search Space Definition（task-agnostic claim）
- B. **Constraint-Aware Multi-Scenario Fitness**（v4 重点扩写）
- C. **Cross-Validated Proxy Evaluation**（v4 新增）
- D. LLM-Guided Generation with Reflection
- E. Island-Based Evolutionary Search
- F. Mechanism Extraction and Transfer Protocol

**V. Experimental Setup（1.5 页）**
- Baselines: Manual（**hybrid**: Nasir backbone + our QoS/fairness, v5 改称）, BO-Std, BO-Ext, **ERS-Struct+BO-Params**, Random, **Lagrangian**（v4 补两项）
- Metrics, statistical protocol（n=5 seeds, 95% CI, paired tests）
- **Implementation sources（v5 新增 transparency paragraph）**:
  - Power Control simulator: physical layer extracted from Nasir & Guo's open-source Asilomar 2020 repository; constraint layer (QoS, adaptive Jain's fairness) added by us
  - ERS framework: built on top of ai4co/reevo (NeurIPS 2024); multi-scenario CV fitness, constraint-aware fitness, and mechanism extraction are our contributions
  - Lagrangian PPO: custom SB3-based implementation following Zangooei et al. 2024, validated on toy constrained CartPole (supplementary)
  - BO: Optuna TPESampler with `constant_liar=True` for async parallel
  - Statistical tools: SciPy `bootstrap(paired=True, method='BCa')` with percentile fallback

**VI. Results and Analysis（5 页）**
- A. Evolution Process
- B. Main Results（两任务主表，四层 BO 对比 + Lagrangian）
- C. Full Pipeline: Evolved Reward → IDT
- D. Mechanism Ablation
- E. Mechanism Transfer（跨算法 + 跨任务 code-level + reward vs scheduler）
- F. **CV Robustness**（v4 新增）
- G. Sensitivity and Cost

**VII. Discussion（0.75 页，v4 加长）**
- 局限性：synthetic channel、discrete action space、仿真器 fidelity 声明
- 与 Eureka/ReEvo 的差异总结
- Future work: 连续 action、标准 O-RAN 仿真器、实际部署

**VIII. Conclusion（0.25 页）**

---

## 六、风险评估与应对（v4 更新版）

### 风险 1：Evolved reward 在 full training 后优势消失
- 概率：中
- 检测：Phase 1 CV proxy fidelity + Phase 2 末 recheck
- 应对：proxy_train_ratio 提高 / surrogate-assisted evolution / 重心转 PPO 贡献

### 风险 2：ERS ≈ BO-Extended（结构无优势）
- 概率：中低
- 检测：Phase 2.8 的 ERS-Struct + BO-Params 对照
- 应对：如 ERS-Full ≈ ERS-Struct+BO-Params，claim 调整为 "LLM 在结构发现上的独立价值"，仍是强结果

### 风险 3：ERS-Struct + BO-Params ≈ ERS-Full（结构足够 = 参数不重要）
- 概率：低（新增于 v4）
- 意义：如成立，是有利结果 — 说明 ERS 的核心贡献是结构发现，且结构可被 BO 精调到最优
- 论文叙事：从 "LLM 自由搜索" 调整为 "LLM 发现结构 + BO 精调参数" 的混合范式

### 风险 4：Power Control 改进幅度有限
- 概率：中
- 检测：Phase 1.4 pilot
- 应对：如改进 <5%，诚实报告并分析原因（PC reward 搜索空间更平坦），保留为 negative-but-informative finding

### 风险 5：审稿人说 "只是 Eureka 的应用"
- 概率：高
- 应对（六层防线，v4 加一层；v5 加强第 2 层）：
  1. 多场景 **交叉验证** 联合 fitness（Eureka 单任务单场景）
  2. **显式 constraint-aware fitness**（Eureka 软约束）— **v5 关键加强**：manual baseline 也**享有 QoS + fairness 约束项**（hybrid baseline），所以 ERS vs Manual 的对比是"**同等约束信号下的结构搜索 vs 手工工程**"，不是"ERS 有约束信号、baseline 没有"的不公平对比
  3. 两个异质 wireless task（Eureka 只做机器人）
  4. 四层 BO 对比 + **ERS-Struct + BO-Params**（Eureka 未区分）
  5. Mechanism transfer 独立贡献（Eureka 没有）
  6. Reward-level vs scheduler-level 实验（Eureka 没有）

### 风险 6：Mechanism Transfer 被降级为 ablation
- 概率：中高
- 应对：
  1. 精确定位为 "reward-level constraint awareness"（非通用 "buffer awareness"）
  2. 跨任务 code-level transfer（非概念类比）
  3. Reward vs scheduler level 正面对比
  4. 跨算法验证（PPO + CQL）

### 风险 7：Proxy 选择 reporting bias
- 概率：**低（v4 降低）**
- 应对：两折 CV + Fold B validation sanity check + 论文 Phase 4.5 专门小节报告

### 风险 8：计算预算
- 预算估算（v5 更新，Phase 1.1 修正）：RAN ~340 GPU-h，PC ~130 GPU-h，BO ~70 GPU-h，**Phase 1.1 CV fidelity ~100 GPU-h（v4 原估 20，严重低估）**，API ~$400-600
- 应对：max_fe 40→30 / sensitivity 粒度减半 / BO trials 100→60
- **v5 新增风险**：若 LagrangianPPO toy-env validation 失败（λ 不收敛 / cost 不收敛到 cost_limit），则被迫回到 OmniSafe 路径，此时 Phase 0 多 3–5 天（依赖冲突解决 + CMDP wrapper）。概率低（现有实现已在 RAN env 产出合理结果），但需在 P0 第 0.5 天先跑通。

### 风险 9：LLM 版本不可复现
- 应对：论文明确记录 model name、provider、snapshot date、accessed date range

### 风险 10（v4 新增）：约束满足率不达标
- 概率：低（因 v4 已将 sat 显式进 fitness）
- 检测：每代 evolved reward 评估 sat(π)
- 应对：如某代全部候选 sat < τ★，增大 λ 惩罚，下一代 prompt 中强调约束

---

## 七、时间线（22 周，含 buffer）

| 周 | 任务 | 决策点 |
|---|---|---|
| 1–2 | Phase 0: 基础设施 | |
| 2–5 | Phase 1: 验证性实验（含 CV fidelity） | **第 5 周：go/no-go** |
| 5–7 | Phase 2a: 进化搜索（RAN）+ BO | |
| 7–9 | Phase 2b: PC + fidelity recheck + **ERS-Struct+BO-Params** | **第 9 周：确认主结果** |
| 9–12 | Phase 3: Full pipeline + Lagrangian | |
| 12–15 | Phase 4: 消融 + transfer + CV robustness | |
| 15–20 | Phase 5: 写作 | |
| 20–22 | 投稿准备 + buffer | **提交** |

**并行策略**：Phase 2 GPU-intensive 但人工干预少 → 同步推 MEC。Phase 4 分析与 Phase 5 写作可与 MEC 交替。

**关键里程碑**：
- **第 5 周末**：CV Proxy fidelity → proxy 策略确认
- **第 9 周末**：主结果出炉 → 投 TWC/JSAC 决策
- **第 15 周末**：全部实验完成 → 全力写作

---

## 八、TWC/JSAC 自检清单（v4 更新）

| 要求 | 对应 | 状态 |
|---|---|---|
| 新方法/框架 | ERS + constraint-aware fitness + CV proxy | ✓ |
| 正交第二贡献 | Transferable Mechanisms | ✓ |
| ≥ 2 验证任务 | RAN Slicing + Power Control | ✓ |
| SOTA 对比 | Manual + BO-Std + BO-Ext + **ERS-Struct+BO** + Random + **Lagrangian** + CQL + IDT | ✓（v4 扩展） |
| 结构 vs 参数定量区分 | 四层 BO 对比 + structure-only 对照 | ✓（v4 加强） |
| 消融 | 各机制独立消融 × 两任务 | ✓ |
| 约束处理 | **Constraint-aware fitness + Lagrangian 对比** | ✓（v4 加强） |
| 跨场景泛化 | **两折 CV** + 5 OOD (RAN) + 多配置 (PC) | ✓（v4 加强） |
| 跨任务泛化 | Code-level mechanism transfer | ✓ |
| 跨算法验证 | PPO + CQL | ✓ |
| 统计显著性 | **n=5 seeds + 95% CI + paired tests** | ✓（v4 加强） |
| 可解释性 | 进化过程 + 机制分析 + reward vs scheduler level | ✓ |
| 计算开销 | GPU + API vs 人工时间 | ✓ |
| Sensitivity | LLM 后端 + max_fe + proxy 场景数 | ✓ |
| Reproducibility | 超参 + LLM 版本 + 代码开源 | ✓ |
| **Limitations 声明**（v4） | Synthetic channel / discrete action / 仿真器 fidelity | ✓ |

---

## 九、对评估意见的响应态度总结

| 评估建议 | v4 响应 | 理由 |
|---|---|---|
| Proxy 交叉验证 | **完全采纳** | 批评精准，方案可行 |
| 显式约束处理机制 | **完全采纳** | 高价值改进，方法章节改写 |
| ERS-Struct + BO-Params 对照 | **完全采纳** | 排除 alternative explanation 关键 |
| 自适应 Jain's fairness 阈值 | **完全采纳** | 技术 catch 精准 |
| 多种子 + CI + p-value | **完全采纳** | 必要的统计规范 |
| WMMSE/FP 直接对比 | **部分采纳**（作 reference，不作核心 baseline） | Nasir & Guo 已隐含此对比，强做会转移焦点 |
| 连续 action space 实验 | **拒绝** | 与"两任务同算法"论证冲突，成本高；放 future work |
| 换 OpenAirInterface | **拒绝** | Sun 2018 TWC 先例充分；Limitations 中声明 |
| 提到 ReEvo 框架借用 | **采纳** | 致谢 + 消融中区分 |

**整体态度**：评估的诊断大部分准确，但处方有过度响应倾向。v4 选择吸收 ROI 最高的批评（proxy CV、约束形式化、结构对照实验、统计规范），避免 timeline 膨胀到 30+ 周。最终投稿潜力（按评估同样框架自评）预计从 3.29 提升到 3.85+（Weak Accept 区间偏上）。

---

## 十、v5 实现注记（开源代码借用 + 依赖清单）

本节给 code-implementer / code-verifier 和论文复现性审稿人一份明确的"代码出处图谱"。

### 10.1 借用的开源仓库

| 用途 | 仓库 | License | 借用方式 | 去向 |
|---|---|---|---|---|
| Power Control 物理层 | [sinannasir/Power-Control-asilomar](https://github.com/sinannasir/Power-Control-asilomar) | MIT（核实） | 仅抽 `project_backend.py` + `random_deployment.py` + `get_benchmarks.py`，**不拉整仓** | `src/basic_apis/power_control/{channel,deployment,benchmarks}.py` |
| ERS 进化骨架 | [ai4co/reevo](https://github.com/ai4co/reevo) | MIT | 已集成在 `src/reevo/`（上游版本锁定在 commit `<hash>`）| — |
| DrEureka 三文件模板（概念借鉴，不拷代码）| [eureka-research/DrEureka](https://github.com/eureka-research/DrEureka) | MIT | 仅参考 `envs/*.py` + `prompts/reward_signatures/*.txt` + `cfg/env/*.yaml` 的分层设计 | 体现在 `src/reevo/problems/<task>_reward/` 三文件结构 |
| BO 骨架 | Optuna `optuna-examples/rl/sb3_simple.py` | MIT | objective 函数模板改造 | `src/basic_apis/bo/objective.py` |
| Lagrangian PPO | 本项目自实现（Zangooei et al. 2024 JSAC 参考）| — | 已存在 `src/basic_apis/ppo/ppo_lagrangian/`；v5 仅加 toy-env validation | 无改动 |

### 10.2 依赖声明（需 pixi add）

```toml
# pixi.toml 需新增
optuna = ">=4.0,<5"               # BO
# scipy = ">=1.15.2,<2"             # 已存在，bootstrap(paired=True) 需 >=1.12
# scikit-learn = 已存在，不用于 KFold，仅用于基础统计
# NO safety-gymnasium（拒绝）
# NO omnisafe（拒绝）
```

### 10.3 新资产路径注册（`conf/paths/default.yaml`）

```yaml
power_control: ${paths.data_root}/power_control
power_control_models: ${paths.power_control}/models
power_control_results: ${paths.power_control}/results
bo_studies: ${paths.data_root}/bo_studies   # Optuna sqlite
```

### 10.4 新 Hydra mode 清单（需在 §scripts_governance 的注册表里登记）

| Mode | 路径 | 用途 |
|---|---|---|
| `power_control/train_ppo_discrete` | `conf/simulation/power_control/train_ppo_discrete.yaml` | K-user IC 单 reward 训练 |
| `power_control/test_ppo_discrete` | `conf/simulation/power_control/test_ppo_discrete.yaml` | 跨 K 评测 |
| `power_control/run_evo` | `conf/simulation/power_control/run_evo.yaml` | Power Control 上的 ERS |
| `bo/run_bo` | `conf/simulation/bo/run_bo.yaml` | Optuna TPE 跑 BO-Std/BO-Ext |
| `analysis/cv_fidelity` | `conf/simulation/analysis/cv_fidelity.yaml` | Kendall τ + BCa CI 计算 |

### 10.5 P0 交付物验收清单（交给 code-verifier）

- [ ] `src/basic_apis/power_control/env.py` 上能跑 `PPO("MlpPolicy", env).learn(10_000)` 无报错
- [ ] `src/basic_apis/power_control/benchmarks.py` 的 WMMSE 能复现 Nasir 原文 Fig. 2 的 avg sum-rate（±5%）
- [ ] `tests/lagrangian_pp_toy_cartpole.py` 通过：`λ` 最终 ∈ [0.1, 100]，`E[cost] ∈ [cost_limit − 0.05, cost_limit + 0.05]`
- [ ] `scripts/analyze_cv_fidelity.py` 在合成数据（τ=0.7, n=15）上给出 BCa CI，且 DegenerateData 时自动 fallback
- [ ] Optuna 并行 smoke test：`n_jobs=4, constant_liar=True, n_trials=8`，4 张卡均有进程（每卡 1 trial，非串行）
- [ ] 论文 draft §2.2 + §II-A + §V 均使用 "hybrid baseline / extend Nasir-Guo" 表述（grep `directly cite Nasir` 应为空）