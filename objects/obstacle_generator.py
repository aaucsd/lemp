from abc import ABC, abstractmethod


class ObstacleGenerator(ABC):
    @abstractmethod
    def generate(self):
        pass