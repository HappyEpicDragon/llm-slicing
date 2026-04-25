# Agentic IDT 研究计划：ReEvo 改造方案

## 文档目的

本文档是 Agentic IDT 项目的技术执行手册，详细规定了如何将 ReEvo 框架改造为适用于 IDT reward function 自动搜索的进化平台。文档覆盖：研究目标、系统架构、改造要点、Prompt 设计、评估协议、实验设计、以及论文包装策略。

---

## 一、研究目标

### 1.1 核心研究问题

> 在 IDT 框架的设计空间中，LLM 引导的进化搜索能否自动发现优于人工设计的 reward function，从而降低 O-RAN network slicing 的 SLA violation rate？

### 1.2 三种可发表结论

- **结论 A（最佳）**：搜索到的设计显著优于人工设计 → "自动发现了更优方案"
- **结论 B（可接受）**：持平 → "自动化达到专家水平，降低部署门槛"
- **结论 C（也可接受）**：不同场景下各有优劣 → "揭示了设计空间的结构性特征，为人工设计提供指导"

### 1.3 学术定位

- **叙事角度**：Design-time agentic workflow（区别于主流的 runtime agentic network）
- **范式对标**：FunSearch / ReEvo / Eureka 的 "LLM-as-generator + evolutionary search" 范式
- **领域贡献**：首次将该范式应用于 O-RAN network slicing 的设计空间探索
- **Conference 目标**：ICC / Globecom / WCNC（5-6 页）
- **可选扩展**：视实验结果决定是否扩展为 TWC/JSAC journal

---

## 二、系统架构总览

### 2.1 整体流程

```
┌─────────────────────────────────────────────────────┐
│                  ReEvo 进化搜索框架                    │
│                                                     │
│  ┌───────────┐    ┌───────────┐    ┌──────────────┐ │
│  │  LLM API  │───>│  gpt.py   │───>│   eval.py    │ │
│  │(DeepSeek) │    │(candidate │    │(PPO训练+评估) │ │
│  │           │    │ reward fn)│    │              │ │
│  └─────▲─────┘    └───────────┘    └──────┬───────┘ │
│        │                                  │         │
│        │         fitness (float)          │         │
│        └──────────────────────────────────┘         │
│                                                     │
│  Population管理 / Elite Archive / Reflection        │
└─────────────────────────────────────────────────────┘
                        │
                        ▼ (top-K candidates)
         ┌──────────────────────────────┐
         │   完整 Pipeline 验证          │
         │  PPO训练 → 数据收集 → IDT训练  │
         │  → 全场景评估                  │
         └──────────────────────────────┘
                        │
                        ▼
              论文结果 (HP/NHP violation rate)
```

### 2.2 关键设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 搜索框架 | Fork ReEvo | 开源成熟、prompt/reflection/archive 机制可直接复用 |
| LLM 后端 | DeepSeek API（主）| 成本低、代码生成能力强 |
| 搜索目标 | Reward function（主）| 接口最干净、对标 Eureka、影响力直接 |
| 评估方式 | PPO proxy（搜索阶段）+ 完整 pipeline（验证阶段）| 平衡搜索效率与结果可靠性 |
| 评估场景 | 单场景 proxy + 全场景验证 | 搜索阶段用最有区分度的场景 |

---

## 三、ReEvo 改造要点

### 3.1 ReEvo 原始架构回顾

ReEvo 的评估接口约定：

1. 框架将 LLM 生成的代码写入 `problems/{problem_name}/gpt.py`
2. 启动子进程：`python -u eval.py {problem_size} {root_dir} train`
3. `eval.py` 中 `import gpt` 读取生成的函数
4. 评估完成后，将 fitness 标量打印到 stdout
5. 框架从 stdout 倒数第二行解析 float 作为 `obj`

### 3.2 需要新建的文件

在 ReEvo 项目中创建以下目录和文件：

```
problems/
└── idt_reward/
    ├── eval.py           # 核心：评估一个 candidate reward function
    ├── eval_validate.py   # 可选：用完整 pipeline 做最终验证
    ├── baseline_reward.py # 当前人工设计的 reward（参考实现）
    └── prompt.md          # Problem description（供 ReEvo prompt 模板引用）
```

以及一个配置文件：

```
cfg/
└── idt_reward.yaml       # ReEvo 运行配置
```

### 3.3 `eval.py`：核心评估脚本

#### 3.3.1 职责

接收 LLM 生成的 reward function → 注入 PPO 训练 → 训练到收敛 → 评估 violation rate → 输出 fitness

#### 3.3.2 接口规范

```python
"""
eval.py for IDT reward function search

调用方式（由 ReEvo 框架自动执行）：
    python -u eval.py {problem_size} {root_dir} train

其中：
    problem_size: 未使用（保持接口兼容），可传 "0"
    root_dir:     ReEvo 项目根目录
    mood:         "train"（搜索阶段）或 "val"（最终验证）

输出：
    fitness 标量打印到 stdout（越小越好，对应 obj_type: min）
"""
import sys
import os

# ---- 解析命令行参数 ----
problem_size = sys.argv[1]
root_dir = sys.argv[2]
mood = sys.argv[3]  # "train" or "val"

# ---- 导入 LLM 生成的 reward function ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpt  # gpt.py 由 ReEvo 框架自动写入，包含 compute_reward()

# ---- Safety Gate：静态检查 ----
def safety_check(reward_fn):
    """
    验证生成的 reward function 是否合法：
    1. 函数可调用
    2. 在随机输入上返回有限标量
    3. 输出不是常数（对不同输入有区分度）
    4. 输出范围合理
    """
    import random
    test_results = []
    for _ in range(20):
        # 模拟不同的 slice margins 和 HP 配置
        n_slices = random.randint(2, 5)
        margins = [random.uniform(-1.0, 1.0) for _ in range(n_slices)]
        is_hp = [random.random() > 0.7 for _ in range(n_slices)]
        try:
            result = reward_fn(margins, is_hp, n_slices)
            if not isinstance(result, (int, float)):
                return False, "Output is not a scalar"
            if not math.isfinite(result):
                return False, "Output is not finite"
            test_results.append(result)
        except Exception as e:
            return False, f"Runtime error: {e}"
    
    # 检查是否为常数函数
    if len(set(test_results)) <= 1:
        return False, "Constant function"
    
    return True, "OK"

# ---- 执行安全检查 ----
import math
passed, msg = safety_check(gpt.compute_reward)
if not passed:
    print(f"Safety check failed: {msg}")
    print(float('inf'))  # 不合法的 candidate，fitness = inf（最差）
    sys.exit(0)

# ---- 训练 PPO 并评估 ----
def train_and_evaluate(reward_fn, mood):
    """
    核心评估逻辑：
    1. 将 reward_fn 注入 PPO 训练环境
    2. 在代表性场景上训练 PPO
    3. 评估 HP/NHP violation rate
    4. 返回 fitness（越小越好）
    """
    # TODO: 导入你的 PPO 训练代码
    # from your_idt_codebase import PPOTrainer, Environment
    
    # TODO: 选择评估场景
    # mood == "train": 用单个代表性场景（最有区分度的那个）
    # mood == "val":   用全部测试场景
    
    # TODO: 创建环境，注入 reward_fn
    # env = Environment(scenario=..., reward_fn=reward_fn)
    
    # TODO: 训练 PPO
    # trainer = PPOTrainer(env, ...)
    # trainer.train(num_steps=...)  # 搜索阶段可缩短
    
    # TODO: 评估
    # hp_viol, nhp_viol = trainer.evaluate()
    
    # TODO: 计算 fitness（越小越好）
    # fitness = w_hp * hp_viol + w_nhp * nhp_viol
    
    # return fitness
    pass

fitness = train_and_evaluate(gpt.compute_reward, mood)
print(fitness)
```

#### 3.3.3 关键实现细节

**Reward function 注入方式**：你需要确认当前 PPO 代码中 reward 计算在哪里。有两种注入策略：

- **方式 A（推荐）**：当前 reward 逻辑在环境的 `step()` 函数中 → 重构为独立函数 → `eval.py` 中用 `gpt.compute_reward` 替换
- **方式 B**：通过 monkey-patching 直接替换环境对象的 reward 方法

**训练步数控制**：搜索阶段（mood="train"）不需要训练到完全收敛。建议用正常训练步数的 1/2 到 1/3，只要 fitness 有区分度即可。这可以在配置中控制。

**Fitness 定义**：

```python
# 推荐的 fitness 定义（越小越好，对应 ReEvo 的 obj_type: min）
w_hp = 10.0    # HP violation 权重远大于 NHP
w_nhp = 1.0
fitness = w_hp * hp_violation_rate + w_nhp * nhp_violation_rate
```

### 3.4 `compute_reward` 函数骨架

#### 3.4.1 函数签名

```python
def compute_reward(
    slice_margins: list[float],
    is_hp: list[bool],
    num_active_slices: int
) -> float:
    """
    计算单个 TTI 的 scheduling reward。

    Args:
        slice_margins: 长度为 num_active_slices 的列表。
            每个元素是该 slice 的 mean user fulfillment margin，范围 [-1, 1]。
            正值表示 SLA 满足，负值表示 SLA 违约，绝对值表示程度。
        is_hp: 长度为 num_active_slices 的布尔列表。
            True 表示该 slice 为高优先级（HP），False 为非高优先级（NHP）。
        num_active_slices: 当前活跃的 slice 数量。

    Returns:
        scalar reward value (float)。
        正值鼓励当前行为，负值惩罚当前行为。
    """
```

#### 3.4.2 当前人工设计的参考实现

```python
def compute_reward(slice_margins, is_hp, num_active_slices):
    """当前人工设计的 baseline reward（公式 22）"""
    w_hp = 10.0
    w_nhp = 1.0
    delta_bonus = 0.1
    
    total_reward = 0.0
    for i in range(num_active_slices):
        w = w_hp if is_hp[i] else w_nhp
        margin = slice_margins[i]
        if margin < 0:
            total_reward += margin * w      # 线性惩罚
        else:
            total_reward += delta_bonus * w  # 固定 bonus
    
    return total_reward / num_active_slices
```

#### 3.4.3 签名设计的考量

这个签名有意做了**信息压缩**：把 per-user、per-metric 的细粒度信息聚合成了 per-slice 的 mean margin。原因是：

- LLM 生成的函数需要简洁可控，输入维度太高会导致生成质量下降
- Mean margin 已经包含了 throughput/latency/loss 三个 metric 的加权信息
- 当前人工设计的 reward 也是在这个聚合层级上工作的

如果后续想扩展搜索空间，可以给 LLM 更细粒度的输入（比如 per-metric margins），但**阶段一不建议这样做**。

### 3.5 ReEvo 配置文件

```yaml
# cfg/idt_reward.yaml

problem_name: "idt_reward"
func_name: "compute_reward"

# --- LLM 设置 ---
llm_backend: "deepseek"          # 或 "openai"
llm_model: "deepseek-chat"       # 具体模型名
temperature: 0.8                  # 生成多样性

# --- 进化参数 ---
population_size: 4                # 每代 candidate 数（= GPU 数量，可并行）
n_generations: 50                 # 总代数
elite_size: 2                     # elite archive 保留数量
timeout: 3600                     # 单次评估超时（秒），PPO 训练可能需要较长时间

# --- 评估设置 ---
obj_type: "min"                   # fitness 越小越好
problem_size: "0"                 # 未使用，保持接口兼容

# --- Reflection 设置（ReEvo 特有）---
use_short_term_reflection: true   # 当代 parent vs child 对比分析
use_long_term_reflection: true    # 跨代经验积累
```

**关键参数说明**：

- `population_size: 4` 对齐你的 4 张 3090，每代 4 个 candidate 可以完全并行评估
- `n_generations: 50` 意味着总共评估 200 个 candidate（50 × 4），这在 PPO proxy 的计算预算下可接受
- `timeout` 必须设足够大，因为 PPO 训练可能需要 20-60 分钟

### 3.6 Prompt 设计

#### 3.6.1 Problem Description（写入 prompt.md）

这是给 LLM 的系统背景，分三层组织：

```markdown
# Problem: Reward Function Design for O-RAN Network Slicing

## System Overview

You are designing a reward function for a reinforcement learning agent that 
performs radio resource scheduling in an O-RAN network slicing system.

The system has S network slices, each serving a group of User Equipments (UEs). 
Each slice declares QoS requirements (intents) on three metrics: throughput, 
buffer latency, and packet loss rate. The slices are classified as either 
High-Priority (HP) or Non-High-Priority (NHP).

At each Transmission Time Interval (TTI), the RL agent makes three decisions:
1. Select a resource allocation template (how many Resource Blocks each slice gets)
2. Determine slice priority ordering (which slice gets first pick of resources)
3. Choose an intra-slice scheduling algorithm (Round Robin / Proportional Fair / Max Throughput)

After the agent acts, a Hierarchical Execution Mechanism (HEM) translates these 
decisions into physical resource allocations:
- Phase 1 (Critical Rescue): Preemptively allocates resources to users at risk 
  of imminent SLA violation, prioritizing HP users
- Phase 2 (Quota Execution): Distributes each slice's allocated resources among 
  its users using the selected scheduling algorithm
- Phase 3 (Global Sweep): Assigns any unallocated residual resources via 
  Proportional Fair to avoid waste

UEs have finite buffers with Poisson traffic arrivals. Packets are dropped when 
the buffer overflows or the latency deadline expires.

## Fulfillment Margin

Each slice's SLA compliance is measured by a fulfillment margin in [-1, 1]:
- Positive margin: SLA is satisfied (higher = more over-provisioned)
- Negative margin: SLA is violated (lower = more severe violation)
- The margin is capped at ζ (over-fulfillment cap) to discourage waste

The margin is a weighted average of per-user margins across throughput, latency, 
and packet loss metrics.

## Reward Function Requirements

The reward function receives:
- `slice_margins`: list of floats in [-1, 1], one per active slice
- `is_hp`: list of booleans, whether each slice is high-priority
- `num_active_slices`: integer, number of active slices

It must return a single float reward value.

## Design Goals

1. **HP protection**: HP slice violations must be eliminated first (zero HP violations 
   is the primary objective)
2. **NHP compliance**: Minimize NHP violations as secondary objective
3. **Resource efficiency**: Discourage over-provisioning; resources allocated beyond 
   SLA requirements should not generate proportional reward
4. **Stability**: The reward should provide smooth, informative gradients for PPO 
   training (avoid sparse or discontinuous signals)

## Current Best Design (Baseline)

The current human-designed reward uses:
- Linear penalty proportional to violation severity for violated slices
- Fixed constant bonus for satisfied slices (discourages over-provisioning)
- HP weight (10.0) >> NHP weight (1.0) for priority differentiation
- Normalization by number of active slices

Current performance: HP violation rate = 0%, NHP violation rate = 1.36%.
The goal is to reduce NHP violation rate while maintaining zero HP violations.

## What You Can Explore

- Nonlinear penalty functions (quadratic, exponential, piecewise)
- Margin-dependent bonus instead of fixed bonus
- Different HP/NHP weight ratios or adaptive weighting
- Temporal components (e.g., penalty escalation for sustained violations)
- Per-violation-severity treatment (mild vs severe violations handled differently)
- Asymmetric treatment of different operating regions
```

#### 3.6.2 Prompt 模板的使用

ReEvo 框架会自动将 problem description、函数签名、parent 代码、evaluation feedback、和 reflection 组合成完整的 generation prompt。你只需要提供 problem description 和函数签名，其余由框架处理。

### 3.7 不需要改动的 ReEvo 组件

以下组件直接使用 ReEvo 原版，不做修改：

- **Population 管理**：elite selection、individual replacement 逻辑
- **Short-term reflection**：parent vs child 的对比分析
- **Long-term reflection**：跨代经验积累
- **LLM 调用接口**：generation、mutation、crossover 的 prompt 组装
- **日志和记录**：每个 candidate 的代码、fitness、generation 信息

---

## 四、评估协议

### 4.1 两阶段评估

#### 阶段一：PPO Proxy（进化搜索阶段使用）

| 项目 | 设置 |
|------|------|
| 评估内容 | 训练 PPO expert + 评估 violation rate |
| 训练场景 | 单个代表性场景（选 violation 最高的场景） |
| 训练步数 | 正常步数的 1/2（加速评估） |
| 评估指标 | `fitness = 10 * HP_viol + 1 * NHP_viol`（越小越好）|
| 单次耗时 | 预估 15-30 分钟/candidate/GPU |
| 并行度 | 4（= 4 张 3090） |

#### 阶段二：完整 Pipeline（最终验证使用）

| 项目 | 设置 |
|------|------|
| 评估内容 | PPO训练 → ε-greedy数据收集 → IDT训练 → 全场景评估 |
| 训练场景 | 全部 5 个训练场景 |
| 训练步数 | 正常步数（完整训练） |
| 评估指标 | HP violation rate, NHP violation rate, Intent Distance |
| 候选数量 | Top-5 candidates from 阶段一 |
| 单次耗时 | 预估 1-2 小时/candidate |

### 4.2 Safety Gate（eval.py 中的静态检查）

在调用 PPO 训练之前，对 LLM 生成的 reward function 执行以下检查：

1. **可执行性**：函数可以被调用，不抛出异常
2. **类型正确**：返回值为 float 且有限（非 NaN/Inf）
3. **非常数**：对不同输入产生不同输出
4. **范围合理**：输出值不超过预设范围（防止数值溢出影响 PPO 训练）

不通过检查的 candidate 直接赋 `fitness = inf`，跳过 PPO 训练。

### 4.3 评估稳定性

PPO 训练有随机性。为控制 fitness 的方差，建议：

- 固定随机种子（相同 candidate 在相同 seed 下结果可复现）
- 如果方差仍然过大，对每个 candidate 用 2-3 个 seed 取平均（但这会成倍增加计算量，仅在必要时启用）

---

## 五、实验设计

### 5.1 对比组（全部必须）

| 组别 | 描述 | 评估预算 | 目的 |
|------|------|----------|------|
| **Human-Designed** | 第一篇论文的现有结果 | 0（已有） | Baseline |
| **Random Search** | LLM 独立生成 200 个 reward function（无进化反馈），选最优 | 200 次评估 | 证明进化优于盲搜 |
| **LLM Single-Shot** | LLM 一次性生成 10 个 candidate（无迭代），选最优 | 10 次评估 | 证明迭代的必要性 |
| **LLM-Guided Evolution** | 本文方法（ReEvo，50 代 × 4 candidate） | 200 次评估 | 主要实验组 |

**关键**：Random Search 的评估预算必须与 LLM-Guided Evolution 相同（都是 200 次），才是公平比较。

### 5.2 分析内容

#### 5.2.1 核心结果表

在全部测试场景上报告：

- HP Violation Rate (%)
- NHP Violation Rate (%)
- Intent Distance
- Weighted Violation Score (fitness)

#### 5.2.2 进化曲线

- X 轴：evaluation count（0 到 200）
- Y 轴：当前最优 fitness
- 对比：LLM-Guided Evolution vs Random Search
- 期望观察：LLM-Guided 收敛更快

#### 5.2.3 搜索到的 Reward 结构分析

对 top-5 candidates 进行人工解读：

- LLM 发明了什么新结构？（非线性惩罚？自适应权重？分段处理？）
- 和人工设计有什么异同？
- 多个 top candidates 之间有没有共同模式？

#### 5.2.4 跨场景泛化

- 在场景 A 上搜索到的最优 reward，在场景 B-E 上表现如何？
- 是否存在"通用好 reward"还是场景敏感？

### 5.3 可选消融实验（时间允许时）

| 消融 | 对比 | 验证什么 |
|------|------|----------|
| 去掉 reflection | 有 reflection vs 无 reflection | Reflection 的贡献 |
| 不同 LLM 后端 | DeepSeek vs GPT-4o vs 本地小模型 | LLM 能力对搜索质量的影响 |
| 不同 population size | 2 vs 4 vs 8 | 种群规模的影响 |

---

## 六、执行步骤

### Step 0：环境准备（0.5 天）

- [ ] Clone ReEvo 仓库
- [ ] 安装依赖，确认框架能跑通（用 ReEvo 自带的 TSP 问题测试）
- [ ] 配置 DeepSeek API key
- [ ] 确认 IDT 代码 pipeline 可跑通（train PPO → collect data → train IDT → evaluate）

### Step 1：重构 Reward Function（0.5-1 天）

- [ ] 定位当前 PPO 代码中 reward 计算的具体位置
- [ ] 将 reward 逻辑重构为独立的 `compute_reward(slice_margins, is_hp, num_active_slices)` 函数
- [ ] 验证重构后 PPO 训练结果与重构前一致（确认没有引入 bug）
- [ ] 编写 `baseline_reward.py`（当前人工设计的 reward，作为参考）

### Step 2：编写 eval.py（1 天）

- [ ] 实现 safety gate（静态检查）
- [ ] 实现 reward function 注入逻辑（替换 PPO 环境中的 reward 计算）
- [ ] 实现 PPO 训练调用（单场景、可配置训练步数）
- [ ] 实现 violation rate 评估
- [ ] 实现 fitness 计算和 stdout 输出
- [ ] 手动测试：用 baseline reward 手动写入 gpt.py，运行 eval.py，确认输出正确

### Step 3：编写配置和 Prompt（0.5 天）

- [ ] 编写 `cfg/idt_reward.yaml`
- [ ] 编写 `problems/idt_reward/prompt.md`（problem description）
- [ ] 确认 ReEvo 的 prompt 模板能正确加载你的 problem description

### Step 4：端到端测试（0.5-1 天）

- [ ] 用最小规模运行 ReEvo（2 代，每代 2 个 candidate）
- [ ] 验证完整链路：LLM 生成 → 写入 gpt.py → eval.py 训练 PPO → fitness 返回 → 下一代生成
- [ ] 检查生成的 reward function 质量（是否合法、是否有多样性）
- [ ] 检查 reflection 是否正常工作（LLM 是否基于上一轮反馈改进）
- [ ] 调整 timeout、训练步数等参数

### Step 5：预实验（1 天）

- [ ] 运行 10 代（40 个 candidate），观察 fitness 趋势
- [ ] 评估关键指标：candidate 合法率（通过 safety gate 的比例）、fitness 方差、代际改进幅度
- [ ] 根据预实验结果调整参数（如 temperature、训练步数、fitness 定义）

### Step 6：正式实验（3-5 天）

- [ ] 运行完整进化搜索（50 代，200 个 candidate）
- [ ] 运行 Random Search baseline（200 个独立 candidate）
- [ ] 运行 LLM Single-Shot baseline（10 个独立 candidate）
- [ ] 所有 candidate 的代码和 fitness 完整记录

### Step 7：完整 Pipeline 验证（2-3 天）

- [ ] 选取 top-5 candidates
- [ ] 每个 candidate 跑完整 pipeline：PPO 训练（全场景）→ 数据收集 → IDT 训练 → 全场景评估
- [ ] 记录 HP violation rate, NHP violation rate, Intent Distance

### Step 8：分析与可视化（1-2 天）

- [ ] 制作进化曲线图
- [ ] 人工分析 top-5 reward function 的结构特征
- [ ] 跨场景泛化测试
- [ ] 整理实验结果表格

### Step 9：论文写作（3-5 天）

- [ ] 复用 IDT 第一篇的 system model（精简）
- [ ] 撰写 method section（搜索框架 + 评估协议）
- [ ] 撰写 experiments section
- [ ] 撰写 introduction 和 related work（design-time agent 叙事 + 参考文献）

---

## 七、风险清单与应对

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|----------|
| LLM 生成的 reward 大部分不合法 | 中 | 搜索效率低 | 改进 prompt（加入更多约束和示例）；调整 temperature |
| PPO 在新 reward 下不收敛 | 中 | candidate 无法评估 | 增加训练步数；设置 fitness = inf 并反馈"training diverged"给 LLM |
| 搜索到的 reward 不优于人工设计 | 中 | 结论变为 B/C | 实验设计已覆盖此情况（进化曲线 + 结构分析仍可发表）|
| PPO proxy 与完整 pipeline 结果不一致 | 低-中 | Top candidates 在 IDT 上表现差 | 增加 top-K 的 K 值（验证更多 candidates）；在 discussion 中分析原因 |
| DeepSeek API 不稳定或调用限制 | 低 | 搜索中断 | 准备备选 LLM（GPT-4o-mini）；或本地部署小模型 |
| ReEvo 框架 bug 或兼容性问题 | 低 | 搭建延迟 | 如果改造成本过高，退回 Eureka 风格的简单闭环（自建 200 行脚本）|

---

## 八、参考文献层次

### 范式奠基（方法论来源）

- **FunSearch**: Romera-Paredes et al., "Mathematical discoveries from program search with large language models," Nature, 2024
- **ReEvo**: Ye et al., "ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution," NeurIPS, 2024
- **Eureka**: Ma et al., "Eureka: Human-Level Reward Design via Coding Large Language Models," ICLR, 2024

### 理论框架（包装模板）

- **Li et al.**, "Optimization Problem Solving Can Transition to Evolutionary Agentic Workflows," arXiv 2505.04354, 2025

### 2025-2026 方法论进展（Related Work 引用）

- **EvoTune**: Surina et al., 2025 — RL + 进化搜索结合
- **EvoPH**: Yi-hong et al., 2025 — Prompt-Heuristic 共同进化
- **MEoH**: Yao et al., AAAI 2025 — 多目标进化启发式
- **ChipSeek-R1**: Chen et al., 2025 — Hierarchical reward 驱动硬件设计

### 领域文献（应用场景）

- IDT 第一篇论文的全部引用（O-RAN、network slicing、DRL scheduling）
- Agentic network survey 中的代表性论文（论证 runtime agent 的局限性）

---

## 九、Journal 扩展路径（Phase 2，待定）

如果 conference 版本的实验结果达到"结论 A"水平，可扩展为 TWC/JSAC，需追加：

1. **多维联合搜索**：同时搜索 reward + risk scoring + fulfillment margin
2. **维度交互分析**：2×2 或 2×2×2 消融表
3. **与传统 AutoML 方法对比**：Bayesian optimization、grid search
4. **面向高成本评估的搜索框架改进**：surrogate-assisted evolution 或 curriculum evaluation
5. **更多测试场景 + 统计显著性检验**
6. **设计空间结构的深度分析**：fitness landscape 可视化、代码聚类