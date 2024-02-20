import torch

from planner.abstract_planner import AbstractPlanner
from torch_geometric.typing import Adj, OptTensor, PairTensor

class LearnedPlanner(AbstractPlanner):
    def __init__(self, model, **kwargs):
        self.model = model
        self.loaded = False

    def _num_node(self):
        ## return nodes on the graph
        raise NotImplementedError

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
            for _, p in enumerate(path):
                if self.model[_] is not None:
                    self.model[_].load_state_dict(torch.load(p, map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load(path))
        self.loaded = True


    def train(self, envs, starts, goals, timeout, **kwargs):
        raise NotImplementedError




