# 迁移 Prompt — DT V2 多教师实验（接续上一会话）

请先检查 terminals 里各后台任务的状态，然后按下述步骤继续。

---

## 背景

网络切片资源调度论文实验，代码仓库 `/root/decision_transformer_slicing`。

### 实验架构
- **环境**：10个网络切片场景（scenario 0-9），HP/NHP切片，135个PRB，每场景100组信道数据
- **DT V2系统**（无urgency_rank，增加信道特征obs，显式排序动作，violation-based reward）
  - EnvV2：`src/basic_apis/dt_v2/env_v2.py`（inter_feat 8维，intra_feat 7维，action=[11]+[5]*5+[3]*5=51维）
  - Split B：训练 S0-6,S8 → 测试 S7,S9

---

## 当前实验阶段：多教师混合数据集

### 已完成

#### 1. 跨场景 PPO（norm版，`dt_v2_split_b_norm/ppo_teacher`）
- 训练：S0-6,S8，800K steps，reward normalization（`/= active_slices`）
- **best_model** 测试结果（20 episodes）：
  - S7: `comb=0.6493`（hp=0.000, nhp=0.649）
  - S9: `comb=0.4634`（hp=0.129, nhp=0.334）
- 跨场景PPO在训练场景上的表现（10 episodes，`dt_v2_split_b_norm/ppo_teacher/best_model`）：
  ```
  S0: comb=1.5854  S1: comb=0.6866  S2: comb=1.5090  S3: comb=0.8600
  S4: comb=0.9867  S5: comb=0.0578  S6: comb=0.4262  S8: comb=0.8630
  Mean comb: 0.8718
  ```
  → **S0、S2 跨场景PPO几乎失效**（comb>1.5），验证了单场景专家的必要性

#### 2. DT V2（基于跨场景PPO，`dt_v2_split_b_norm/dt_model`）
- 数据集：1600条，epsilon=0.1，8个训练场景各200条
- 训练：100 epochs，batch_size=3072，4 GPU DataParallel
- 测试结果（最佳 epoch_95，20 episodes）：
  - S7: `comb=0.6216`（hp=0.000, nhp=0.622）
  - S9: `comb=0.5112`（hp=0.142, nhp=0.370）
- **结论：DT被教师质量限制，S9差距大（oracle_safe=0.342，DT=0.511）**

#### 3. 对比基准
| 方法 | S7 comb | S9 comb |
|------|---------|---------|
| dt_oracle_safe | 0.5785 | 0.3416 |
| PPO norm best（教师） | 0.6493 | 0.4634 |
| DT V2 epoch_95（当前最好） | 0.6216 | 0.5112 |
| urgency_rank（原始baseline） | ~0.79 | ~0.001 |

#### 4. 单场景PPO专家训练（正在后台运行）
- **Batch 1（S0-S3，已完成）**：各自独立训练800K steps
  - 保存目录：`data/channel_generality/dt_v2_per_scenario/ppo_s{0,1,2,3}/`
- **Batch 2（S4,S5,S6,S8，正在运行）**：各在GPU 0-3上并行
  - 保存目录：`data/channel_generality/dt_v2_per_scenario/ppo_s{4,5,6,8}/`
  - 监控：`tail -5 /tmp/ppo_s{4,5,6,8}.log`

---

## 当前待做（按顺序执行）

### Step 1：确认单场景PPO训练完成

```bash
# 检查Batch 2是否完成
tail -5 /tmp/ppo_s4.log /tmp/ppo_s5.log /tmp/ppo_s6.log /tmp/ppo_s8.log

# 检查best_model是否存在
ls data/channel_generality/dt_v2_per_scenario/ppo_s*/best_model/
```

### Step 2：测试每个单场景PPO在自己场景上的表现

```python
# 对比跨场景PPO和单场景专家PPO在同一场景的comb指标
# 期望：单场景PPO在自己场景上明显优于跨场景PPO
```

### Step 3：收集混合数据集（1800条）

```bash
# 从8个单场景专家各收集200条
for s in 0 1 2 3 4 5 6 8; do
  .pixi/envs/default/bin/python -m src.basic_apis.dt_v2.collect_data_v2 \
    --model data/channel_generality/dt_v2_per_scenario/ppo_s${s}/best_model/best_model.zip \
    --episodes 200 --epsilon 0.1 --scenarios $s \
    --output data/channel_generality/dt_v2_multi_teacher/dataset_per_scenario \
    --parallel 1
done

# 从跨场景PPO再收集200条（已有1600条，可复用 dt_v2_split_b_norm/dataset）
# 或者只用单场景专家数据也可以先试试
```

**注意**：如果 `collect_data_v2.py` 不支持在不同目录追加写入，
可能需要先分别收集再合并，或者一次性启动多个场景：
```bash
# 方案A：单场景PPO各自收集，输出到同一目录
# collect_data_v2.py --output 目录 时，会按 scenario_id + episode_idx 命名pkl，
# 不同 --scenarios 调用不会覆盖，可以重复调用追加

# 方案B：也可以先看一下 collect_data_v2.py 是否支持多次调用同一output目录
```

### Step 4：训练混合数据集 DT

```bash
# 合并数据后（或直接用合并后的目录）
CUDA_VISIBLE_DEVICES=0,1,2,3 .pixi/envs/default/bin/python -m src.basic_apis.dt_v2.train_dt_v2 \
  --dataset data/channel_generality/dt_v2_multi_teacher/dataset_per_scenario \
  --save_dir data/channel_generality/dt_v2_multi_teacher/dt_model \
  --epochs 100 --batch_size 3072
```

**注意**：metadata.json 需要基于合并后的所有轨迹重新计算。
检查 `collect_data_v2.py` 的 `--output` 是否在每次调用后都会重写 metadata.json。
如果是，需要在所有数据收集完毕后，最后一次统一计算 metadata：

```python
# 可能需要单独跑 metadata 计算脚本
# 或者让 collect_data_v2.py 全部收集完再一起跑
```

### Step 5：测试 DT（多个 epoch checkpoint）

```bash
.pixi/envs/default/bin/python -m src.basic_apis.dt_v2.test_v2 \
  --model_path data/channel_generality/dt_v2_multi_teacher/dt_model/epoch_50.pth \
  --meta_path data/channel_generality/dt_v2_multi_teacher/dataset_per_scenario/metadata.json \
  --target_rtg 0.0 --scenarios 7 9 --n_episodes 20
```

---

## 成功标准

| 方法 | S7 target | S9 target |
|------|-----------|-----------|
| DT 多教师版 | < 0.60（优于跨场景DT的0.622） | < 0.40（大幅改善，接近oracle_safe的0.342） |

---

## 关键文件路径

| 文件 | 路径 |
|------|------|
| EnvV2 | `src/basic_apis/dt_v2/env_v2.py` |
| PPO训练 | `src/basic_apis/dt_v2/train_ppo_v2.py` |
| 数据收集 | `src/basic_apis/dt_v2/collect_data_v2.py` |
| DT训练 | `src/basic_apis/dt_v2/train_dt_v2.py` |
| DT测试 | `src/basic_apis/dt_v2/test_v2.py` |
| 跨场景PPO（norm） | `data/channel_generality/dt_v2_split_b_norm/ppo_teacher/` |
| 单场景PPO专家 | `data/channel_generality/dt_v2_per_scenario/ppo_s{0-8}/` |
| 旧DT数据集（跨场景） | `data/channel_generality/dt_v2_split_b_norm/dataset/` |
| 旧DT模型（跨场景） | `data/channel_generality/dt_v2_split_b_norm/dt_model/` |
| 新多教师数据集（目标） | `data/channel_generality/dt_v2_multi_teacher/dataset_per_scenario/` |
| 新DT模型（目标） | `data/channel_generality/dt_v2_multi_teacher/dt_model/` |
| oracle_safe基准（S7/S9） | `data/channel_generality/dt_oracle_safe/metric_json/` |

---

## 重要注意事项

1. **collect_data_v2.py 的 metadata 问题**：每次运行都会覆盖 metadata.json。
   建议：先把8个单场景PPO的数据都收集到同一目录，最后统一计算metadata。
   或者先检查 `collect_data_v2.py` 源码确认行为。

2. **单场景PPO测试**：收集前最好先快速测一下每个单场景PPO在自己场景上的表现，
   确认它们确实优于跨场景PPO，再决定是否收集数据。

3. **是否加入跨场景PPO数据**：可以先只用8个单场景专家数据（1600条），
   对比当前结果；如果不够好再加入跨场景PPO数据（变成1800条）。
