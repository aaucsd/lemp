from abc import ABC, abstractmethod
import numpy as np


class AbstractTrajectory(ABC):

    @abstractmethod
    def get_spec(self, t):
        raise NotImplementedError  
        
    @abstractmethod
    def set_spec(self, obstacle, t):
        raise NotImplementedError


class WaypointDiscreteTrajectory(AbstractTrajectory):
    '''
    following the waypoints moving in discrete time
    '''
    
    def __init__(self, waypoints):
        self.waypoints = waypoints

    def get_spec(self, t):
        assert isinstance(t, int)
        if t != -1:
            assert 0<=t<=(len(self.waypoints)-1)
        return self.waypoints[t]
        
    def set_spec(self, obstacle, spec):
        obstacle.set_config(spec)
        
        
class WaypointLinearTrajectory(AbstractTrajectory):
    '''
    following the waypoints moving in continuous time
    the motion is linear between adjacent timesteps
    '''
    
    def __init__(self, waypoints):
        self.waypoints = waypoints

    def get_spec(self, t):
        if t == -1 or t >= len(self.waypoints)-1:
            return self.waypoints[-1]
        # if t != -1:
        #     assert 0<=t<=(len(self.waypoints)-1)
        t_prev, t_next = int(np.floor(t)), int(np.ceil(t))
        spec_prev, spec_next = self.waypoints[t_prev], self.waypoints[t_next]
        return spec_prev + (spec_next-spec_prev)*(t-t_prev)

    def set_spec(self, obstacle, spec):
        obstacle.set_config(spec)        
        