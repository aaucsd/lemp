import torch
from planner.learned_planner import LearnedPlanner
from utils.utils import seed_everything, create_dot_dict, to_np
from utils.graphs import knn_graph_from_points

from planner.learned.model.GNN_static import GNNet

from torch_sparse import coalesce
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data

from tqdm import tqdm as tqdm
import numpy as np


class GNNStaticPlanner(LearnedPlanner):
    def __init__(self, num_batch, model_args, k_neighbors=50, **kwargs):
        self.num_batch = num_batch
        self.model = GNNet(**model_args)
        self.k_neigbors = k_neighbors
        self.num_node = 0

        super(GNNStaticPlanner, self).__init__(self.model, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _num_node(self):
        return self.num_node

    def _plan(self, env, start, goal, timeout, seed=0, **kwargs):

        seed_everything(seed=seed)
        self.model.eval()
        path = self._explore(env, start, goal, self.model, timeout, k=self.k_neigbors, n_sample=self.num_batch)

        return create_dot_dict(solution=path if len(path) else None)

    def create_graph(self):
        graph_data = knn_graph_from_points(self.points, self.k_neighbors)
        self.edges = graph_data.edges
        self.edge_index = graph_data.edge_index
        self.edge_cost = graph_data.edge_cost

    def create_data(self, points, obstacles, edge_index=None, k=50):
        goal_index = 1
        data = Data(goal=torch.FloatTensor(points[goal_index]))
        data.v = torch.FloatTensor(np.array(points))
        data.obstacles = torch.FloatTensor()

        if edge_index is not None:
            data.edge_index = torch.tensor(edge_index.T).to(self.device)
        else:
            # k1 = int(np.ceil(k * np.log(len(points)) / np.log(100)))
            edge_index = knn_graph(torch.FloatTensor(data.v), k=k, loop=True)
            edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
            ### bi-directional graph
            data.edge_index, _ = coalesce(edge_index, None, len(data.v), len(data.v))

        # create labels
        labels = torch.zeros(len(data.v), 1)
        labels[goal_index, 0] = 1
        data.labels = labels

        return data
        
    @torch.no_grad()
    def _explore(self, env, start, goal, model_gnn, timeout, k, n_sample, loop=10):
        success = False
        path = []
        points = [start] + [goal] + env.robot.sample_n_free_points(n_sample) 

        explored = [0]
        explored_edges = [[0, 0]]
        prev = {0: 0}

        while not success:

            data = self.create_data(points, env.get_obstacles(), k=k)
            self.num_node = len(data.v)
            policy = model_gnn(**data.to(self.device).to_dict(), loop=loop)
            policy = policy.cpu()

            policy[torch.arange(len(data.v)), torch.arange(len(data.v))] = 0
            policy[:, explored] = 0
            policy[np.array(explored_edges).reshape(2, -1)] = 0

            while policy[explored, :].sum() != 0:

                agent = policy[np.array(explored)[torch.where(policy[explored, :] != 0)[0]], torch.where(policy[explored, :] != 0)[1]].argmax()
                end_a, end_b = torch.where(policy[explored, :] != 0)[0][agent], torch.where(policy[explored, :] != 0)[1][agent]
                end_a, end_b = int(end_a), int(end_b)
                end_a = explored[end_a]
                explored_edges.extend([[end_a, end_b], [end_b, end_a]])
                if env.edge_fp(to_np(data.v[end_a]), to_np(data.v[end_b])):
                    explored.append(end_b)
                    prev[end_b] = end_a

                    policy[:, end_b] = 0
                    if end_b==1:
                        success = True
                        path = [end_b]
                        node = end_b
                        while node != 0:
                            path.append(prev[node])
                            node = prev[node]
                        path.reverse()
                        break
                else:
                    policy[end_a, end_b] = 0
                    policy[end_b, end_a] = 0

                self.check_timeout(timeout)

            if not success:
                # ----------------------------------------resample----------------------------------------
                new_points = env.sample_n_points(n_sample, need_negative=True)
                points = points + list(new_points)
       
        return list(data.v[path].data.cpu().numpy())



