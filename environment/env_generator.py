from abc import ABC, abstractmethod


class EnvGenerator(ABC):
    @abstractmethod
    def generate(self):
        pass