import os
import torch
import numpy as np
from tqdm import tqdm

from src.basic_apis.dt_utils.model_ha_dt import HierarchicalStateEncoder, DecisionTransformer
from src.basic_apis.network_slicing_business.path_manager import PathManager
from src.basic_apis.general_utils import pad_stack_tensor
from src.basic_apis.explainable_ai_utils.xai_utils import ACTION_DIMS, make_env_for_scenario, resolve_project_path


def extract_dt_embedding(model, obs, device):
    """
    核心手术刀：只运行 DT 的 State Encoder 部分，提取 Embedding
    这对应 PPO 的 features_extractor 输出
    """
    # 准备输入 Tensor
    t_inter = torch.from_numpy(obs['inter_feat']).float().to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, 5, 4]
    t_intra = torch.from_numpy(obs['intra_feat']).float().to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, 25, 5]
    t_glob = torch.from_numpy(obs['global_feat']).float().to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, 2]

    with torch.no_grad():
        # 调用 model.state_encoder
        # 输出形状 [1, 1, 256] -> squeeze -> [256]
        embedding = model.state_encoder(t_inter, t_intra, t_glob).squeeze(0).squeeze(0)

    return embedding.cpu().numpy()


def run_dt_collection(cfg, path_manager):
    """
    语义流形对齐实验 - GRRM-DT 数据采集入口
    """
    print(f"🧠 [DT] Starting Data Collection for Experiment")
    print(f"    Model Path: {cfg.model_paths.dt}")

    # 1. 准备工作
    save_dir = resolve_project_path(path_manager, cfg.asset_dir)
    os.makedirs(save_dir, exist_ok=True)

    base_env_conf = cfg.environment

    # 2. 初始化 DT 模型
    action_dims = ACTION_DIMS
    state_encoder = HierarchicalStateEncoder(embed_dim=512)
    model = DecisionTransformer(
        state_encoder=state_encoder,
        action_dims=action_dims,
        hidden_size=512,
        max_length=20  # context_len，与训练一致
    ).to(cfg.device)

    # 加载权重
    try:
        model.load_state_dict(torch.load(cfg.model_paths.dt, map_location=cfg.device))
        model.eval()
        print("✅ DT Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading DT Model: {e}")
        return

    # 3. 采集参数设置
    context_len = 20
    # 设定一个较高的目标回报，诱导 DT 发挥最佳水平
    # 注意：这里使用的是 Raw Value，之后会做 SymLog
    target_rtg = -50000.0
    rtg_candidates = np.linspace(-300000, -20000, num=10)
    rtg_scale = float(getattr(cfg, "rtg_scale", 100000.0))  # 与主链路 symlog+scale 对齐



    collected_data = []

    # 4. 遍历场景
    for key, scen_conf in cfg.target_scenarios.items():
        print(f"👉 Processing {scen_conf.name} (Scenario {scen_conf.scenario_id})...")

        env = make_env_for_scenario(base_env_conf.env_settings, scen_conf, path_manager)
        obs, _ = env.reset()

        # === 初始化推理上下文 Buffer ===
        # Dummy action (全0) 用于 t=0 时刻的输入
        dummy_action = torch.zeros(len(action_dims), dtype=torch.long, device=cfg.device)

        history = {
            "inter": [], "intra": [], "global": [],
            "actions": [dummy_action],  # a_{-1}
            "rtg": [],  # R_0 ...
            "timesteps": []
        }

        cumulative_reward = 0.0
        curr_step = 0

        # 采集循环
        # 我们采集 samples_per_scenario 个步数
        # 如果中途 done 了，会自动 reset 并延续 buffer (为了简化实现，done 后我们可以清空 buffer 重新开始，这更符合 DT 的逻辑)

        pbar = tqdm(total=cfg.samples_per_scenario, desc=f"Collecting {key}")
        collected_count = 0

        # [修改点] 针对不同场景设定不同的 RTG 诱导范围
        # 目的是覆盖该场景下的 "Low performance" 到 "High performance"
        if scen_conf.scenario_id == 0:  # S2 (Source) - 容易
            # S2 可能的分数范围: -50k (优) ~ -200k (差)
            current_candidates = np.linspace(-200000, -20000, num=10)
        else:  # S9 (Target) - 困难
            # S9 可能的分数范围: -150k (优) ~ -400k (差)
            # 我们需要给更低的 Prompt 才能让它觉得"不用拼命"，从而输出保守动作
            current_candidates = np.linspace(-400000, -100000, num=10)

        # 采集循环
        while collected_count < cfg.samples_per_scenario:

            # [Step 1] 随机选择一个适配当前场景的 RTG
            if curr_step == 0:
                target_rtg = np.random.choice(current_candidates)

        # while collected_count < cfg.samples_per_scenario:
        #     if curr_step == 0:
        #         target_rtg = np.random.choice(rtg_candidates)
            # --- A. 提取特征 (The Brain Scan) ---
            # 这是我们最关心的 XAI 数据：Transformer "看到" 的世界
            # 此时的 obs 是 raw numpy dict
            embedding = extract_dt_embedding(model, obs, cfg.device)

            # --- B. 维护上下文 (Context Management) ---
            # 1. Process Obs to Tensor
            t_inter = torch.from_numpy(obs['inter_feat']).float().to(cfg.device)
            t_intra = torch.from_numpy(obs['intra_feat']).float().to(cfg.device)
            t_glob = torch.from_numpy(obs['global_feat']).float().to(cfg.device)

            history["inter"].append(t_inter)
            history["intra"].append(t_intra)
            history["global"].append(t_glob)

            # 2. Calculate & Transform RTG
            # RTG_input = SymLog(Target - Cumulative)
            current_rtg_raw = target_rtg - cumulative_reward
            rtg_val = np.sign(current_rtg_raw) * np.log1p(np.abs(current_rtg_raw))
            # 归一化
            t_rtg = torch.tensor([rtg_val / rtg_scale], dtype=torch.float32, device=cfg.device)
            history["rtg"].append(t_rtg)

            # 3. Timestep
            t_step = torch.tensor([curr_step], dtype=torch.long, device=cfg.device)
            history["timesteps"].append(t_step)

            # 4. Window Sliding (Pop old)
            if len(history["rtg"]) > context_len:
                for k in ["inter", "intra", "global", "rtg", "timesteps", "actions"]:
                    history[k].pop(0)

            # --- C. 构造模型输入 (Padding & Stacking) ---
            in_inter = pad_stack_tensor(history["inter"], context_len, None, cfg.device)
            in_intra = pad_stack_tensor(history["intra"], context_len, None, cfg.device)
            in_global = pad_stack_tensor(history["global"], context_len, 2, cfg.device)
            in_rtg = pad_stack_tensor(history["rtg"], context_len, 1, cfg.device)
            in_steps = pad_stack_tensor(history["timesteps"], context_len, 1, cfg.device).squeeze(-1).long()

            # Actions 需要特殊处理 stack
            act_seq = torch.stack(history["actions"])
            if act_seq.shape[0] < context_len:
                pad_len = context_len - act_seq.shape[0]
                pad = torch.zeros((pad_len, len(action_dims)), dtype=torch.long, device=cfg.device)
                act_seq = torch.cat([pad, act_seq], dim=0)
            in_actions = act_seq.unsqueeze(0)  # [1, K, 6]

            states_in = {'inter': in_inter, 'intra': in_intra, 'global': in_global}

            # Attention Mask
            real_len = min(len(history["rtg"]), context_len)
            mask = torch.zeros((1, context_len), device=cfg.device)
            mask[0, -real_len:] = 1.0

            with torch.no_grad():
                context_embedding = model.get_context_embedding(
                    states=states_in,
                    actions=in_actions,
                    returns=in_rtg,
                    timesteps=in_steps,
                    attention_mask=mask
                )
                embedding_np = context_embedding.squeeze().cpu().numpy()

                action_preds = model(
                    states=states_in, actions=in_actions,
                    returns=in_rtg, timesteps=in_steps, attention_mask=mask
                )
                last_logits = action_preds[0, -1, :]

                pred_actions = []
                start_idx = 0
                for dim_size in action_dims:
                    end_idx = start_idx + dim_size
                    dim_logit = last_logits[start_idx:end_idx]
                    dim_act = torch.argmax(dim_logit).item()
                    pred_actions.append(dim_act)
                    start_idx = end_idx

                action = np.array(pred_actions, dtype=np.int32)

            state_intensity = np.linalg.norm(obs['inter_feat'])
            record = {
                "embedding": embedding_np,
                "scenario_label": scen_conf.label,
                "semantic_action": int(action[0]),
                "target_rtg": float(target_rtg),
                "timestep": int(curr_step),
                "state_intensity": float(state_intensity),
                "model_type": "DT"
            }
            collected_data.append(record)

            next_obs, reward, done, _, _ = env.step(action)
            act_tensor = torch.tensor(action, dtype=torch.long, device=cfg.device)
            history["actions"].append(act_tensor)
            cumulative_reward += reward

            obs = next_obs
            curr_step += 1
            collected_count += 1
            pbar.update(1)

            if done:
                obs, _ = env.reset()
                cumulative_reward = 0.0
                curr_step = 0
                history = {
                    "inter": [], "intra": [], "global": [],
                    "actions": [dummy_action],
                    "rtg": [], "timesteps": []
                }

        pbar.close()
        env.close()

    # 5. 保存数据
    save_path = os.path.join(save_dir, "dt_data.npy")
    np.save(save_path, collected_data)
    print(f"💾 [DT] Data collection complete. Saved {len(collected_data)} samples to:")
    print(f"   -> {save_path}")