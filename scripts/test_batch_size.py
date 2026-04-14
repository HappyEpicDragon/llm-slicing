"""
快速测试不同 batch_size 是否 OOM（不加载完整数据集）

用法: CUDA_VISIBLE_DEVICES=0,1,2,3 .pixi/envs/default/bin/python scripts/test_batch_size.py
"""
import torch
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.basic_apis.dt_utils.model_ha_dt import HierarchicalStateEncoder, build_decision_model
from omegaconf import OmegaConf

device = torch.device("cuda")
n_gpus = torch.cuda.device_count()
print(f"GPUs: {n_gpus}")

model_cfg = OmegaConf.create({
    "context_len": 20, "embed_dim": 512, "n_layer": 12,
    "n_head": 16, "activation": "relu", "dropout": 0.1, "act_dim": 30,
})
action_dims = [11, 3, 3, 3, 3, 3]
state_encoder = HierarchicalStateEncoder(embed_dim=512)
model = build_decision_model(model_cfg=model_cfg, state_encoder=state_encoder, action_dims=action_dims).to(device)

if n_gpus > 1:
    model = torch.nn.DataParallel(model)
    print(f"DataParallel on {n_gpus} GPUs")

K = 20  # context_len

for bs in [2048, 4096, 6144, 8192, 10240]:
    torch.cuda.empty_cache()
    try:
        states = {
            'inter': torch.randn(bs, K, 5, 4, device=device),
            'intra': torch.randn(bs, K, 25, 5, device=device),
            'global': torch.randn(bs, K, 2, device=device),
        }
        actions = torch.randint(0, 3, (bs, K, 6), device=device)
        rtg = torch.randn(bs, K, 1, device=device)
        timesteps = torch.arange(K, device=device).unsqueeze(0).expand(bs, -1)
        mask = torch.ones(bs, K, device=device)

        with torch.cuda.amp.autocast():
            out = model(states=states, actions=actions, returns=rtg, timesteps=timesteps, attention_mask=mask)
            loss = out.sum()
            loss.backward()

        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  bs={bs:>6}: OK  (peak {mem:.1f} GB)")
        del states, actions, rtg, timesteps, mask, out, loss
        model.zero_grad()
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  bs={bs:>6}: OOM")
            torch.cuda.empty_cache()
        else:
            raise
