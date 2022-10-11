from abc import ABC, abstractmethod
import numpy as np

class AbstractEnv(ABC):
    
    # Initialize env
    def __init__(self, dim, limits_low, limits_high, robots, obstacles, starts, goals):
        self.dim = dim
        self.limits_low = limits_low
        self.limits_high = limits_high
        self.robots = robots
        self.obstacles = obstacles
        self.starts = starts
        self.goals = goals

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
        if need_negative:
            negative = []
        samples = []
        for i in range(n):
            while True:
                sample = self.uniform_sample()
                if (self.dim==2 and self._state_fp(sample)):
                    samples.append(sample)
                    break
                elif need_negative:
                    negative.append(sample)
        if not need_negative:
            return samples
        else:
            return samples, negative

    def sample_empty_points(self):
        while True:
            point = self.uniform_sample()
            if self.dim == 2:
                if self._point_in_free_space(point):
                    return point
            if self.dim == 3:
                if self._stick_in_free_space(point):
                    return point

    def set_random_init_goal(self):
        while True:
            init, goal = self.sample_empty_points(), self.sample_empty_points()
            if np.sum(np.abs(init - goal)) != 0:
                break
        self.init_state, self.goal_state = init, goal

    def uniform_sample(self, n=1):
        '''
        Uniformlly sample in the configuration space
        '''
        sample = np.random.uniform(self.limits_low, self.limits_high, (n, self.dim))
        if n==1:
            return sample.reshape(-1)
        else:
            return sample        

    def get_problem(self):
        problem = {
            "map": self.map,
            "init_state": self.init_state,
            "goal_state": self.goal_state
        }
        return problem    
    
    # =====================internal collision check module=======================

    @abstractmethod
    def _edge_fp(self):
        pass

    @abstractmethod
    def _state_fp(self):
        pass

    @abstractmethod
