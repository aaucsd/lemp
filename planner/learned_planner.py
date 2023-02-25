import torch

from planner.abstract_planner import AbstractPlanner

class LearnedPlanner(AbstractPlanner):
    def __init__(self, model, **kwargs):
        self.model = model
        self.loaded = False

    def save_model(self, path):
        if isinstance(self.model, list):
            if not isinstance(path, list):
                raise ValueError("paths should be a list of paths")
            for _, m in enumerate(self.model):
                torch.save(m.state_dict(), path[_])
        else:
            torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        if isinstance(self.model, list):
            if not isinstance(path, list):
                raise ValueError("paths should be a list of paths")
            for _, m in enumerate(self.model):
                if path[_] is not None:
                    m.load_state_dict(torch.load(path[_]))
        else:
           self.model.load_state_dict(torch.load(path))
        self.loaded = True


    def train(self, envs, starts, goals, timeout, **kwargs):
        raise NotImplementedError




