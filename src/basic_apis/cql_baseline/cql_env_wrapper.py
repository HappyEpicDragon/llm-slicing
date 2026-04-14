"""
CQL 环境适配器。
将 HierarchicalSlicingEnv 的字典观测降维为 flat per-slice 向量（dim=45），
并将 CQL 的连续 fraction 动作映射回离散环境动作。
"""
import numpy as np
from src.basic_apis.codebook_utils import build_dirichlet_inter_quota_codebook


# 与 HierarchicalSlicingEnv 保持一致的 inter-quota template 定义
def _build_inter_quota_patterns(num_prbs=135):
    """构建与环境一致的 Dirichlet template codebook。"""
    return build_dirichlet_inter_quota_codebook(
        num_slices=5,
        num_prbs=num_prbs,
        c_min=10,
        num_concentration_levels=10,
        num_mc_samples=10_000,
        beta_min=0.1,
        beta_max=50.0,
        seed=2025,
    )


INTER_QUOTA_PATTERNS = _build_inter_quota_patterns(135)


class CQLObsWrapper:
    """将 HierarchicalSlicingEnv 的字典观测降维为 flat 向量（dim=45）"""

    OBS_DIM = 45  # 5*4 (inter) + 5*5 (intra_per_slice)

    def transform_obs(self, obs_dict):
        """
        支持两种格式：
          - dict: {'inter_feat': [5,4], 'intra_feat': [25,5], ...}  (expert/dt 数据集)
          - ndarray shape (145,): inter(20) + intra(125)            (raw/dt_baseline 数据集)
        返回: flat numpy array, shape=(45,)
        """
        if isinstance(obs_dict, np.ndarray):
            # flat layout: inter(5×4=20) + intra(25×5=125)
            inter = obs_dict[:20].reshape(5, 4).astype(np.float32)
            intra = obs_dict[20:145].reshape(25, 5).astype(np.float32)
        else:
            inter = np.array(obs_dict['inter_feat'], dtype=np.float32)    # [5, 4]
            intra = np.array(obs_dict['intra_feat'], dtype=np.float32)    # [25, 5]

        intra_per_slice = intra.reshape(5, 5, 5).mean(axis=1)         # [5, 5]
        flat = np.concatenate([inter.flatten(), intra_per_slice.flatten()])
        return flat  # dim = 20 + 25 = 45

    def transform_action(self, continuous_fracs):
        """
        将 CQL 输出的连续 resource fraction 向量（长度 5）映射回环境所需的离散动作。
        动作格式：[template_id, j1, j2, j3, j4, j5]，intra-slice 固定为 PF (scheduler_id=1)
        """
        template_id = self._match_template(continuous_fracs)
        return np.array([template_id, 1, 1, 1, 1, 1], dtype=np.int32)

    def _match_template(self, continuous_fracs):
        """找最接近的 template（按 L2 距离）"""
        fracs = np.array(continuous_fracs, dtype=np.float32)
        total = fracs.sum()
        if total > 1e-8:
            fracs = fracs / total
        best_id = 0
        best_dist = float('inf')
        for tid, pattern in enumerate(INTER_QUOTA_PATTERNS):
            normalized = pattern / (pattern.sum() + 1e-8)
            dist = np.sum((fracs - normalized) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_id = tid
        return best_id
