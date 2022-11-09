from abc import ABC, abstractmethod
from objects.abstract_object import AbstractObject
from objects.trajectory import AbstractTrajectory


class MovableObject(AbstractObject, ABC):
    
    @abstractmethod
    def set_config(self, config):
        raise NotImplementedError


class DynamicObject(AbstractObject):

    def __init__(self, item: MovableObject, trajectory: AbstractTrajectory, **kwargs):
        self.item = item
        self.trajectory = trajectory
        super(DynamicObject, self).__init__(**kwargs)

    def set_config_at_time(self, t):
        spec = self.trajectory.get_spec(t)
        self.trajectory.set_spec(self.item, spec)