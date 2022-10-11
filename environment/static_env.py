from abc import ABC, abstractmethod
from environment.abstract_env import AbstractEnv


class StaticEnv(AbstractEnv):
    
    # Initialize env
    def __init__(self):
        pass

    @abstractmethod
    def _reset(self):
        self.finished = False

    @abstractmethod
    def render(self):
        pass
    
    @abstractmethod
    def check_collision(self):
        pass