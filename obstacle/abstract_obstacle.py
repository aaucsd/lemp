from abc import ABC, abstractmethod


class AbstractObstacle(ABC):

    @abstractmethod
    def load2pybullet(self, **kwargs):
        raise NotImplementedError