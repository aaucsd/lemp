from abc import ABC, abstractmethod
import numpy as np
from environment.abstract_env import AbstractEnv
from objects.dynamic_object import MovableObject, DynamicObject
from objects.trajectory import AbstractTrajectory
import pybullet as p

class AbstractRobot(MovableObject, ABC):

    # An abstract robot    
    def __init__(self, limits_low, limits_high, **kwargs):
        self.limits_low = limits_low
        self.limits_high = limits_high
        self.config_dim = len(self.limits_low)
        super(AbstractRobot, self).__init__(**kwargs)

    # =====================pybullet module=======================
    
    @abstractmethod
    def set_config(self, config, item_id=None):
        '''
        set a configuration
        '''        
        raise NotImplementedError        
        
    # =====================sampling module=======================        
        
    def uniform_sample(self, n=1):
        sample = np.random.uniform(self.limits_low.reshape(1, -1), self.limits_high.reshape(1, -1), (n, self.config_dim))
        if n==1:
            return sample.reshape(-1)
        else:
            return sample

    def sample_free_config(self):
        while True:
            state = self.uniform_sample()
            if self._state_fp(state):
                return state        
        
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

    def sample_random_init_goal(self):
        while True:
            init, goal = self.sample_free_config(), self.sample_free_config()
            if np.sum(np.abs(init - goal)) != 0:
                break
        return init, goal
    
    # =====================internal collision check module=======================
    
    @abstractmethod
    def no_collision(self):
        '''
        Perform the collision detection
        '''
        raise NotImplementedError      
    
    def _valid_state(self, state):
        return (state >= np.array(self.limits_low)).all() and \
               (state <= np.array(self.limits_high)).all()      
    
    def _state_fp(self, state):
        if not self._valid_state(state):
            return False

        self.set_config(state)
        return self.no_collision()
    
    def _iterative_check_segment(self, left, right):
        if np.sum(np.abs(left - left)) > 0.1:
            mid = (left + right) / 2.0
            if not self._state_fp(mid):
                return False
            return self._iterative_check_segment(left, mid) and self._iterative_check_segment(mid, right)

        return True 
    
    def _edge_fp(self, state, new_state):
        assert state.size == new_state.size

        if not self._valid_state(state) or not self._valid_state(new_state):
            return False
        if not self._state_fp(state) or not self._state_fp(new_state):
            return False

        disp = new_state - state

        d = self.distance(state, new_state)
        K = int(d / self.collision_eps)
        for k in range(0, K):
            c = state + k * 1. / K * disp
            if not self._state_fp(c):
                return False
        return True
    
    def distance(self, from_state, to_state):
        '''
        Distance metric
        '''
        to_state = np.maximum(to_state, np.array(self.limits_low))
        to_state = np.minimum(to_state, np.array(self.limits_high))
        diff = np.abs(to_state - from_state)

        return np.sqrt(np.sum(diff ** 2, axis=-1))   
    
    # =====================internal collision check module for dynamic environment=======================    
    
    def _state_fp_dynamic(self, env, state, t):
        for each_object in env.objects:
            if isinstance(each_object, DynamicObject):
                each_object.set_config_at_time(t)
        return self._state_fp(state)
    
    def _edge_fp_dynamic(self, env, state, new_state, t_start, t_end):
        assert state.size == new_state.size

        if not self._valid_state(state) or not self._valid_state(new_state):
            return False
        if not self._state_fp_dynamic(env, state, t_start) or not self._state_fp_dynamic(env, new_state, t_end):
            return False

        disp = new_state - state
        t_disp = t_end - t_start

        d = self.distance(state, new_state)
        K = int(d / self.collision_eps)
        for k in range(0, K):
            c = state + k * 1. / K * disp
            t_c = t_start + k * 1. / K * t_disp
            if not self._state_fp_dynamic(env, c, t_c):
                return False
        return True             


class DynamicRobotFactory:
    
    @staticmethod
    def create_dynamic_robot_class(Robot):
        class DynamicRobot(Robot, DynamicObject):
            def __init__(self, trajectory: AbstractTrajectory, **kwargs):
                super(DynamicRobot, self).__init__(item=self, trajectory=trajectory, **kwargs)
        return DynamicRobot