from abc import ABC, abstractmethod
from omegaconf import DictConfig
import logging

class BaseSimulation(ABC):
    """所有仿真的基类，定义了必须遵守的接口。"""
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def run(self) -> None:
        """执行仿真的核心逻辑。"""
        pass