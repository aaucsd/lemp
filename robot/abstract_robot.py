from abc import ABC, abstractmethod
import numpy as np
from environment.abstract_env import AbstractEnv

class AbstractRobot(ABC):
    
    # Initialize env
    def __init__(self, limits_low, limits_high, joints, collision_eps):
        self.limits_low = limits_low
        self.limits_high = limits_high
        self.joints = joints
        self.collision_eps = collision_eps

    # =====================pybullet module=======================
    
    def load(self, config):
        robot_id = self.load2pybullet()
        self.collision_check_count = 0
        self.robot_id = robot_id
        self.set_config(config)
        
    @abstractmethod
    def load2pybullet(self):
        '''
        load into PyBullet and return the id of robot
        '''        
        return 0
    
    def set_config(self, config, robot_id=None):
        if robot_id is None:
            robot_id = self.robot_id
        for i, c in zip(self.joints, config):
            p.resetJointState(robot_id, i, c)
        p.performCollisionDetection()
        
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
    
    def _valid_state(self, state):
        return (state >= np.array(self.limits_low)).all() and \
               (state <= np.array(self.limits_high)).all()      
    
    def _state_fp(self, state):
        if not self._valid_state(state):
            return False

        self.set_config(state)
        p.performCollisionDetection()
        if len(p.getContactPoints(self.robot_id)) == 0:
            self.collision_check_count += 1
            return True
        else:
            self.collision_check_count += 1
            return False
    
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