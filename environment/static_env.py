from abc import ABC, abstractmethod
from environment.abstract_env import AbstractEnv
import pybullet as p


class StaticEnv(AbstractEnv):
    
    # Initialize env
    def __init__(self, robot_ids, dim, limits_low, limits_high, collision_eps):
        self.robot_ids = robot_ids
        self.collision_eps = collision_eps
        super().__init__(limits_low, limits_high)

    @abstractmethod
    def _reset(self):
        self.finished = False

    @abstractmethod
    def render(self):
        pass
    
    # =====================internal collision check module=======================
    
    def _valid_state(self, state):
        return (state >= np.array(self.limits_low)).all() and \
               (state <= np.array(self.limits_high)).all()      
    
    def _state_fp(self, state):
        if not self._valid_state(state):
            return False

        self.set_config(state)
        p.performCollisionDetection()
        if np.all([len(p.getContactPoints(robot_id)) == 0 for robot_id in self.robot_ids]):
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