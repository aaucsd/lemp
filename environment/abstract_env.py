from abc import ABC, abstractmethod
import numpy as np

class AbstractEnv(ABC):
    
    # Initialize env
    def __init__(self, limits_low, limits_high):
        self.config_dim = len(limits_low)
        self.limits_low = np.array(limits_low)
        self.limits_high = np.array(limits_high)

    @abstractmethod
    def _reset(self):
        self.finished = False

    @abstractmethod
    def render(self):
        """
        Return a snapshot of the current environment
        :abstract
        """        
        pass

    def sample_n_points(self, n, need_negative=False):
        positive = []
        negative = []
        for i in range(n):
            while True:
                state = self.uniform_sample()
                if self._state_fp(state):
                    positive.append(state)
                    break
                else:
                    negative.append(state)
        if not need_negative:
            return positive
        else:
            return positive, negative

    def sample_free_config(self):
        while True:
            state = self.uniform_sample()
            if self._state_fp(state):
                return state

    def set_random_init_goal(self):
        while True:
            init, goal = self.sample_free_config(), self.sample_free_config()
            if np.sum(np.abs(init - goal)) != 0:
                break
        self.init_state, self.goal_state = init, goal

    def uniform_sample(self, n=1):
        '''
        Uniformlly sample in the configuration space
        '''
        sample = np.random.uniform(self.limits_low.reshape(1, -1), self.limits_high.reshape(1, -1), (n, self.config_dim))
        if n==1:
            return sample.reshape(-1)
        else:
            return sample
    
    # =====================internal collision check module=======================

    @abstractmethod
    def _edge_fp(self, state, new_state):
        pass

    @abstractmethod
    def _state_fp(self, state):
        pass
