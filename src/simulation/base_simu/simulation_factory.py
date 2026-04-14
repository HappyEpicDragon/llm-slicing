from typing import Dict, Type, Callable
from .base_simulation import BaseSimulation
from omegaconf import DictConfig

class SimulationFactory:
    """带自动注册功能的仿真工厂。"""
    _simulations: Dict[str, Type[BaseSimulation]] = {}

    @classmethod
    def register(cls, sim_type: str) -> Callable:
        """用于注册仿真类的装饰器。"""
        def decorator(sim_class: Type[BaseSimulation]):
            cls._simulations[sim_type] = sim_class
            # 可选：在注册时打印日志，方便调试
            # print(f"✅ Registered '{sim_type}' -> {sim_class.__name__}")
            return sim_class
        return decorator

    @classmethod
    def create(cls, sim_type: str, cfg: DictConfig) -> BaseSimulation:
        """根据类型创建仿真实例。"""
        if sim_type not in cls._simulations:
            available = list(cls._simulations.keys())
            raise ValueError(f"未注册的仿真类型: '{sim_type}'. 可用类型: {available}")
        sim_class = cls._simulations[sim_type]
        return sim_class(cfg)

    @classmethod
    def list_available(cls) -> list:
        """列出所有已注册的仿真。"""
        return list(cls._simulations.keys())