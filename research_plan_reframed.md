# Research Plan (Reframed)

## 0. 一句话定义这篇工作

本文聚焦 **intent-driven RAN slicing** 这一主系统模型，研究一个此前常被当作“实现细节”但实际上决定性能上限的核心问题：**优化目标如何定义**。在给定系统模型、状态/动作接口以及通用求解器（如 PPO、DT）的前提下，我们使用 **LLM-guided objective synthesis** 自动生成可执行的 reward/objective 代码，使学习到的 slicing policy 在多场景下同时实现更好的系统效能与约束满足。

---

## 1. 这篇论文打算讲什么

### 1.1 论文主题

我们不把本文定义为“LLM 同时解决多个无线资源分配任务”的工作，也不把它定义为“自然语言直接自动建模并自动求解网络问题”的 agentic demo。  

本文的主题更聚焦，也更适合通信社区的叙事方式：

> **LLM for objective design in learning-based RAN slicing**

更具体地说，本文研究：

- 在 **RAN slicing** 中，人工设计 reward/objective 往往需要大量试错；
- 目标函数既要体现系统效能（如 fulfillment margin、resource efficiency），又要体现硬约束（如 SLA violation 约束、HP protection）；
- 这类目标函数设计高度依赖经验，而且在跨场景时容易失效；
- 因此，我们希望让 LLM 不去替代求解器，而是去 **自动设计优化目标的可执行表达**。

### 1.2 本文不做什么

为了让工作边界清晰，本文明确 **不** 研究以下问题：

- 不让 LLM 自动设计动作空间；
- 不让 LLM 自动发现新的 RL 求解器；
- 不做“自然语言到完整优化建模与求解”的全闭环 agentic network；
- 不把多个相互独立的无线任务强行捆绑成一篇文章的双主线。

因此，本文的边界是：

- **system model 固定**
- **决策变量/动作接口固定**
- **通用求解器固定**
- **LLM 负责搜索 objective/reward**

---

## 2. System Model：我们打算采用什么舞台

### 2.1 主舞台：Intent-Driven RAN Slicing

本文采用 **intent-driven RAN slicing** 作为唯一主系统模型。

选择这一 system model 的原因有三点：

1. **通信意义强**：RAN slicing 本身就是一个明确的无线系统问题，而不是抽象 benchmark；
2. **代码基础现成**：我们已有可复用的 slicing 仿真代码与前期工作积累；
3. **问题张力合适**：它天然包含“soft objective + hard constraints”的冲突，非常适合讨论 objective design。

### 2.2 系统元素

系统由一个共享无线资源池和多个 slice 构成。每个 slice 对应不同业务类型与服务意图（intent），例如：

- high-priority / latency-sensitive slice
- eMBB-like throughput-oriented slice
- best-effort / non-HP slice

调度器需要在每个决策周期为不同 slice 分配资源，并决定具体的资源分配行为。不同场景下，系统负载、用户需求、流量分布与信道统计会发生变化，因此同一个 objective 在不同 scenario 下可能表现显著不同。

### 2.3 为什么这个 system model 足够支撑整篇论文

RAN slicing 已经同时具备以下三个要素：

- **明确的物理/网络语义**：资源分配、slice isolation、intent satisfaction；
- **非平凡的约束结构**：HP violation、SLA 满足率、不同 slice 间权衡；
- **学习求解的现实困难**：reward 稍有偏差，就会导致“性能看起来不错但违反约束”或者“满足约束但资源利用极差”。

因此，单独一个 RAN slicing system model 就足以支撑“LLM 设计 objective”这一研究主题，不需要再额外引入 power control 作为第二主舞台。

---

## 3. Problem：我们要解决什么问题

### 3.1 核心科学问题

在给定：

- 固定的 RAN slicing 系统模型；
- 固定的状态/动作接口；
- 固定的通用求解器（如 PPO、DT）；

的前提下，**如何自动构造一个更好的优化目标表达**，使得求解器学到的 policy 在多个场景下：

- 取得更好的系统效能；
- 更稳定地满足硬约束；
- 比人工设计的目标函数具有更好的泛化性。

### 3.2 该问题的本质

这个问题本质上不是“RL 算法设计”问题，而是一个 **objective design problem**：

- 从优化视角看，是在搜索一个更合理的 objective / utility / penalty structure；
- 从强化学习视角看，是在搜索一个可执行的 reward function；
- 二者是同一层问题的不同表述。

因此，本文采用这样的统一理解：

> **上位概念是 objective synthesis；在 RL 实现中，这个 objective 被落地为 reward code。**

### 3.3 为什么这是一个值得单独研究的问题

在很多 learning-based wireless control 工作中，reward 往往被当作附属组件人工指定。但在 RAN slicing 中，objective 设计本身决定了：

- 哪些系统行为被鼓励；
- 哪些约束违反被强惩罚；
- 学到的 policy 是否具有跨场景鲁棒性；
- 求解器是否会利用 reward loopholes 产生“看似高分、实际不可部署”的策略。

因此，objective engineering 不是实现细节，而是问题本身的重要组成部分。

---

## 4. Method：我们打算用什么方法来解

### 4.1 方法总述

我们提出一个 **LLM-guided objective synthesis** 框架。该框架不直接输出资源分配动作，而是自动生成 reward/objective 的 Python 代码，再通过通用求解器训练 policy，并根据训练后的多场景表现反过来评价该 objective 的好坏。

其基本闭环为：

1. 给定 slicing 环境、变量接口与 reward skeleton；
2. LLM 生成候选 objective/reward 代码；
3. 使用固定求解器（如 PPO）在多个训练场景上训练 policy；
4. 在 held-out scenarios 上评估该 reward 对应 policy 的性能与约束满足；
5. 将评估结果反馈给 LLM / evolutionary search，继续改写 objective。

### 4.2 方法定位

该方法的关键定位是：

- **LLM 不是 solver**
- **LLM 是 objective designer**
- **PPO / DT 等通用算法仍然是 solver**

因此本文的方法与“LLM directly solves optimization”不同，也与“LLM 自动发现新 RL 算法”不同。

### 4.3 目标搜索对象

我们允许 LLM 搜索的不只是几个 reward 权重，而是 reward 的 **结构与参数**，例如：

- 是否引入分段惩罚；
- 是否使用 margin 型 shaping；
- 是否使用显式 violation gates；
- 不同指标之间是线性叠加、乘性耦合还是条件触发；
- 惩罚项与奖励项的尺度、阈值和归一化方式。

因此，该方法是 **objective structure + parameter co-design**，而不是简单 HPO。

### 4.4 外层评估原则：Constraint-Aware Evaluation

本文不依赖“reward 里自己写了惩罚项，所以系统就会自动学会守约束”这一假设。  
相反，我们把硬约束满足率显式放进外层 evaluator 中。

外层 evaluator 至少包含两部分：

- **performance term**：系统效能，如 fulfillment margin、intent distance 改善、资源利用等；
- **constraint term**：HP violation rate、SLA satisfaction rate、hard constraint compliance。

这意味着我们评价的不是“reward 本身写得像不像人类工程师”，而是：

> 这个 objective 是否真的诱导出了更好的、可约束部署的策略。

### 4.5 多场景训练与验证

为了避免 objective 只在单一 scenario 上过拟合，本文采用多场景训练/验证协议：

- 训练阶段使用多个 slicing scenarios；
- 评估阶段使用 seen / unseen scenarios；
- objective 的 fitness 由跨场景综合表现决定，而不是单场景分数。

因此，本文追求的不是单场景最优 reward，而是 **cross-scenario robust objective**。

### 4.6 通用求解器角色

在当前计划中，求解器不作为贡献点，而作为受控变量存在：

- 在线/训练型 solver：PPO
- 离线/序列建模式 solver：DT（如需要与前序工作衔接）

我们不 claim 新 solver，而是利用这些通用求解器验证：

- 当 objective 更合理时，通用求解器也能学到更好的策略；
- 问题的提升来自 objective 设计，而不是求解器技巧。

---

## 5. 本文的主要 Contribution

### Contribution 1: 把 RAN slicing 中的 reward engineering 上升为 objective synthesis 问题

现有 learning-based wireless control 工作通常默认 objective 由人给定，而本文明确提出：

- 在 intent-driven RAN slicing 中，objective design 是影响性能与守约束的核心瓶颈；
- 该问题可以被形式化为一个独立研究问题，而不是实现细节。

这一定义本身就是本文的重要贡献，因为它重新界定了“LLM 在 network control 中该放在哪一层”。

### Contribution 2: 提出 LLM-guided objective synthesis for RAN slicing

本文提出一个以 LLM 为目标设计器、以通用 RL solver 为执行器的闭环框架：

- LLM 生成 reward/objective 代码；
- 固定 solver 训练策略；
- 多场景 evaluator 反馈 objective 优劣；
- 通过迭代搜索得到更优 objective。

该框架的特点是：**不改 system model，不改 solver，专注改 objective**。

### Contribution 3: 提出显式的 constraint-aware objective evaluation

本文不把硬约束仅仅作为 reward 中的软惩罚项，而是在外层评价中显式考察 constraint satisfaction。  
这使得 objective 搜索的目标与实际系统需求一致，即：

- 不只追求高 utility
- 还追求可部署性与守约束性

这也是本文区别于一般“LLM 帮忙调 reward”式工作的关键点。

### Contribution 4: 证明收益来自 objective structure，而不只是权重调参

本文会通过消融和对照实验区分：

- 手工 reward
- 手工 template + 参数搜索
- LLM 搜索出的结构 + 参数

从而回答一个关键问题：

> LLM 带来的提升，到底来自更好的结构发现，还是只是更激进的参数调优？

如果这个问题被清楚回答，论文的方法论价值会显著高于“又一个 reward tuning work”。

### Contribution 5: 面向跨场景鲁棒性的 objective 设计

本文不是只在单个 slicing scenario 上寻找高分 reward，而是明确面向 cross-scenario robustness。  
这与我们前序关于 slicing 泛化性的工作能够自然衔接，也使本文不只是“自动化”，而是“自动化 + 泛化导向”。

---

## 6. 实验设计的基本思路

### 6.1 主实验对象

主实验全部围绕 `RAN slicing` 展开。

实验将比较：

- manual objective / reward
- parameter-tuned manual reward
- LLM-synthesized objective / reward

### 6.2 评估维度

重点关注以下指标：

- intent fulfillment / intent distance
- HP violation rate
- SLA / constraint satisfaction rate
- aggregate efficiency metrics
- seen-to-unseen scenario generalization

### 6.3 消融重点

本文建议至少做以下消融：

- **structure vs parameter**：结构重要还是参数重要；
- **single-scenario vs multi-scenario fitness**：多场景评价是否真的提升泛化；
- **constraint-aware evaluator on/off**：显式约束评价是否必要；
- **PPO / DT under same objective**：同一 objective 是否能被不同通用 solver 复用。

---

## 7. 关于 Power Control 的位置

在新的研究计划中，**power control 不再作为主任务**。

原因很简单：

- 它不是当前论文主舞台所必需；
- 它会显著稀释论文的 system model；
- 它会让审稿人把注意力转移到“为什么拼两个任务”上。

因此，当前建议是：

- 主文只保留 RAN slicing；
- power control 最多作为 future work，或后续独立论文方向；
- 如果未来一定要保留，也只能作为 very light secondary validation，而不能再作为双主任务。

---

## 8. 论文最终希望建立的主张

本文希望向通信与网络智能社区传达的核心信息不是：

- “LLM 什么网络问题都能做”

而是：

> 在 learning-based RAN slicing 中，真正困难且关键的一层，是 objective design。  
> LLM 最合适的角色，不是替代求解器，而是作为 objective designer，帮助自动构造更鲁棒、更守约束的优化目标表达。

这一定义既保留了 LLM 的方法创新，也保留了通信论文最需要的 system-level clarity。

---

## 9. 当前版本的题目方向建议

下面这些题目方向都比“LLM for network + 多任务拼接”更聚焦：

- **LLM-Guided Objective Synthesis for Intent-Driven RAN Slicing**
- **Automating Objective Design for Learning-Based RAN Slicing**
- **Constraint-Aware Objective Synthesis for Intent-Driven RAN Slicing**
- **Large Language Models as Objective Designers for RAN Slicing Control**

如果更强调 RL 实现，也可以用：

- **LLM-Guided Reward Synthesis for Intent-Driven RAN Slicing**

但从定位上，我更建议优先使用 **objective**，再在文中说明“在 RL 实现中 objective 落地为 reward code”。

---

## 10. 本研究计划的最终结论

本文的最佳组织方式不是：

- 一个通用 LLM 方法，验证于 slicing 和 power control 两个相对独立任务

而是：

- 一个明确的通信系统模型：**intent-driven RAN slicing**
- 一个明确的问题：**objective design for learning-based control**
- 一个明确的方法：**LLM-guided synthesis of executable reward/objective code**
- 一组明确的贡献：**objective 问题化、constraint-aware evaluation、结构搜索、跨场景鲁棒性**

这会让整篇论文从“任务拼接”变成“问题清楚、舞台稳定、方法有层次”的通信论文。
