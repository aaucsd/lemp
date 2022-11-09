from abc import ABC, abstractmethod


class StaticObstacle(ABC):

    @abstractmethod
    def load2pybullet(self, **kwargs):
        raise NotImplementedError