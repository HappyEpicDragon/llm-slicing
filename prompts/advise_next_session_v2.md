# 下一步探索的提示词（V2 版）

将以下内容粘贴到新会话中：

---

## 背景

我们在做网络切片资源调度的论文实验，代码仓库在 /root/decision_transformer_slicing。

### 实验架构
- **环境**：10个网络切片场景（scenario 0-9），每场景有3-5个切片（HP高优先级/NHP低优先级），135个PRB资源块，每场景有100组信道数据
- **分层动作空间（HA）**：Dim0 = codebook条目（11类，控制跨切片PRB分配的不均匀程度），Dims1-5 = intra-slice调度器（RR/PF/MT三选一）
- **关键环境机制**：`urgency_rank`动态映射（按需求排序后分配codebook模板）+ `critical_rescue`（紧急资源重分配）
- **教师策略**：PPO-HA-weighted，联合模型在训练场景上训练

### 已完成的关键发现（全在本会话中得出）

#### 1. Codebook 选择的信噪比极低（R²=0.034）
- 200个场景的Oracle分析证明：最优codebook在0-10上近乎均匀随机分布
- 原因：`urgency_rank`动态映射吸收了codebook差异，剩余差异由信道噪声主导
- 150-train/50-test的LogisticRegression/RandomForest分类器预测最优codebook，准确率不如baseline
- **结论：observation → best codebook 之间不存在可学习的泛化规律**

#### 2. Intra=RR 是跨场景通用的最优调度
- Oracle sweep（11 codebooks × 3 intra × 5 test scenarios）全面验证
- PF/MT 在所有场景上都不如 RR

#### 3. 切片排序的信噪比很高（Phase 0 验证通过）
- 去掉 urgency_rank 后，5!=120种排序的violation spread：S5=0.48, S6=0.64, S7=0.56, S8=0.19, S9=0.94
- 远大于codebook的spread（0.3-0.4）
- S7上urgency_rank排名#97/120（很差！）→ DT有超越空间

#### 4. DT V2 系统设计（已实现，代码在 `src/basic_apis/dt_v2/`）
- **EnvV2**（`env_v2.py`）：去掉urgency_rank，增加信道特征到observation（inter_feat 8维，intra_feat 7维），action空间加5维显式排序，violation-based reward（已加normalization）
- **PPO教师**（`train_ppo_v2.py`）：在EnvV2上训练，学习排序+codebook+intra
- **数据收集**（`collect_data_v2.py`）：epsilon-greedy收集PPO轨迹
- **DT训练**（`train_dt_v2.py`）：HierarchicalStateEncoderV2（8维inter，7维intra）
- **测试**（`test_v2.py`）：在EnvV2上评估DT

#### 5. 当前实验进展：Split B（Train S0-6,S8 → Test S7,S9）

**PPO教师（无reward normalization，已完成）**：
- 训练：S0-6,S8，800K timesteps
- PPO学到了动态排序（94%步切换率）、codebook集中在act2/5（安全区）、intra 93% RR
- 但PPO在S9上表现很差（0.779），因为简单场景主导了eval reward → best model偏向简单场景
- PPO在S7上表现尚可（0.673），但离Oracle best（0.427）有距离

**DT V2（epoch 20结果，基于上述PPO教师）**：
- S7: **0.648**（rank #33/120，**击败urgency_rank的0.792**）→ 亮点！
- S9: 0.516（rank #83/120，不如urgency_rank的0.001）→ 继承了PPO教师在S9上的弱点
- DT在S7上甚至超过了PPO教师（0.648 vs 0.673）— 学生超过老师

**PPO教师（reward normalization版，正在训练中）**：
- 修改：`reward /= active_slices`，消除场景难度差异
- 保存在 `data/channel_generality/dt_v2_split_b_norm/ppo_teacher/`
- 训练命令：
  ```
  .pixi/envs/default/bin/python -m src.basic_apis.dt_v2.train_ppo_v2 \
    --timesteps 800000 --scenarios 0 1 2 3 4 5 6 8 \
    --save_dir data/channel_generality/dt_v2_split_b_norm/ppo_teacher
  ```

### 当前待做

1. **PPO norm版训练完成后**：测试best/final model在S7、S9上的表现
   ```
   # 可参考之前的测试代码，在新会话中直接运行
   ```

2. **如果S9改善**：收集轨迹(1600条) → 训练DT(100 epochs) → 测试S7,S9
   ```
   .pixi/envs/default/bin/python -m src.basic_apis.dt_v2.collect_data_v2 \
     --model data/channel_generality/dt_v2_split_b_norm/ppo_teacher/best_model/best_model.zip \
     --episodes 200 --epsilon 0.1 --scenarios 0 1 2 3 4 5 6 8 \
     --output data/channel_generality/dt_v2_split_b_norm/dataset --parallel 8

   CUDA_VISIBLE_DEVICES=0,1,2,3 .pixi/envs/default/bin/python -m src.basic_apis.dt_v2.train_dt_v2 \
     --dataset data/channel_generality/dt_v2_split_b_norm/dataset \
     --save_dir data/channel_generality/dt_v2_split_b_norm/dt_model \
     --epochs 100 --batch_size 3072

   .pixi/envs/default/bin/python -m src.basic_apis.dt_v2.test_v2 \
     --model_path data/channel_generality/dt_v2_split_b_norm/dt_model/epoch_50.pth \
     --meta_path data/channel_generality/dt_v2_split_b_norm/dataset/metadata.json \
     --target_rtg 0.0 --scenarios 7 9 --n_episodes 20
   ```

3. **如果S9仍然差**：需要更深层的修复，可能的方向：
   - per-scenario PPO（每个场景单独训一个PPO，然后混合收集轨迹）
   - 更复杂的reward shaping（基于排序的相对改善而非绝对violation）
   - 增加PPO训练步数或调小学习率

### 成功标准

DT V2在S7和S9上都要比urgency_rank好（或至少接近），才能声称"DT学到的排序泛化到未见场景"：
- S7 target: < 0.65（当前DT已做到0.648 ✓）
- S9 target: < 0.3（当前DT 0.516，urgency_rank 0.001，gap很大）

### 关键文件路径
- V2环境：`src/basic_apis/dt_v2/env_v2.py`
- V2 PPO训练：`src/basic_apis/dt_v2/train_ppo_v2.py`
- V2 数据收集：`src/basic_apis/dt_v2/collect_data_v2.py`
- V2 DT训练：`src/basic_apis/dt_v2/train_dt_v2.py`
- V2 DT测试：`src/basic_apis/dt_v2/test_v2.py`
- V2 模型定义：`src/basic_apis/dt_v2/model_v2.py`
- Phase 0排序sweep：`scripts/phase0_ordering_sweep.py`
- Oracle 200场景sweep：`scripts/oracle_sweep_200.py`
- 实验数据：`data/channel_generality/dt_v2_split_b*/`、`data/channel_generality/oracle_200/`、`data/channel_generality/phase0_ordering/`

请先检查PPO norm版训练是否完成（看terminals里的输出），然后按上述步骤继续。
