import torch


def pad_stack_tensor(tensor_list, context_len, feat_dim, device):
    """
    Left-pad and truncate a list of tensors to fixed length K.

    Args:
        tensor_list: elements shaped [F] or [1, F]
        context_len: target sequence length K
        feat_dim: feature dimension F, used when the list is empty
        device: target device
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
