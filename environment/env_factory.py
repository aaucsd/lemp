from abc import ABC, abstractmethod
import numpy as np


def EnvFactory(ABC):

    @abstractmethod
    def create_env():
        """
        Return an environment
        """          
        raise NotImplementedError