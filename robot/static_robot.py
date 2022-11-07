from abc import ABC, abstractmethod
import numpy as np
from robot.abstract_robot import AbstractRobot

class StaticRobot(AbstractRobot):
    
    # Initialize env
    def __init__(self, limits_low, limits_high):
        
        super().__init__(limits_low, limits_high)
    
    # =====================internal collision check module=======================

    @abstractmethod
    def _edge_fp(self, env: StaticEnv, state, new_state):
        pass

    @abstractmethod
    def _state_fp(self, env: StaticEnv, state):
        pass