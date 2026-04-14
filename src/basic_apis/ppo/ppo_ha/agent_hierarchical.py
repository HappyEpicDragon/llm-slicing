# ppo_ha_weighted 是带 return_attn 支持的超集版本，直接复用其实现。
from src.basic_apis.ppo.ppo_ha_weighted.agent_hierarchical import (
    HierarchicalAttentionExtractor,
    HierarchicalSmartPolicy,
)

__all__ = ["HierarchicalAttentionExtractor", "HierarchicalSmartPolicy"]
