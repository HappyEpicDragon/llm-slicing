import numpy as np
import torch


class AssocDataAnalyzer:
    def __init__(self, npz_data):
        self.npz_data = npz_data

    def get_keys(self):
        for key in self.npz_data.keys():
            print(key)

    def get_values(self):
        pass


def analyze_assoc_data(cfg):
    base_dir = cfg['base_dir']
    eps_list = cfg['eps_list']
    import os
    for ep in eps_list:
        data_path = os.path.join(base_dir, f"ep_{ep}.npz")
        data = np.load(data_path, allow_pickle=True)
        analyzer = AssocDataAnalyzer(data)
        analyzer.get_keys()

def deep_update(original, updates):
    """递归更新嵌套字典"""
    for key, value in updates.items():
        if key in original and isinstance(original[key], dict) and isinstance(value, dict):
            deep_update(original[key], value)
        else:
            original[key] = value
    return original


def pad_stack_tensor(tensor_list, context_len, feat_dim, device):
    """
    将 List[Tensor] 左侧补零并截断到固定长度 K，返回 [1, K, F]。

    Args:
        tensor_list: 每个元素形状为 [F] 或 [1, F]
        context_len: 目标序列长度 K
        feat_dim: 特征维度 F（列表为空时用于构造零张量）
        device: 目标设备
    Returns:
        Tensor of shape [1, K, F]
    """
    if len(tensor_list) == 0:
        return torch.zeros((1, context_len, feat_dim if feat_dim else 1), device=device)

    tensors = [t.squeeze(0) if t.ndim > 1 else t for t in tensor_list]
    seq = torch.stack(tensors)  # [T, F] or [T]

    curr_len = seq.shape[0]
    if curr_len < context_len:
        pad_len = context_len - curr_len
        shape = list(seq.shape)
        shape[0] = pad_len
        pad = torch.zeros(shape, device=device, dtype=seq.dtype)
        seq = torch.cat([pad, seq], dim=0)
    else:
        seq = seq[-context_len:]

    if seq.ndim == 1:
        seq = seq.unsqueeze(-1)

    return seq.unsqueeze(0)  # [1, K, F]

