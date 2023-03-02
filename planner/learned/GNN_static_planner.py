import torch
from planner.learned.model.GNN_dynamic import GNNet, PolicyHead
from planner.learned.model.base_models import PositionalEncoder
from planner.learned_planner import LearnedPlanner
from utils.utils import seed_everything, create_dot_dict

from collections import OrderedDict, defaultdict
from torch_sparse import coalesce
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data

from tqdm import tqdm as tqdm
import numpy as np


class GNNDynamicPlanner(LearnedPlanner):
    def __init__(self, num_samples, max_num_samples, use_bctc, model_path=None, k_neighbors=50, num_samples_add=200, **kwargs):
        self.model = [GNNet(config_size=2, embed_size=32, obs_size=9, use_obstacles=True), PolicyHead(embed_size=32), PositionalEncoder(embed_size=32), PositionalEncoder(embed_size=9)]
        self.num_samples = num_samples
        self.num_samples_add = num_samples_add
        self.max_num_samples = max_num_samples
        self.use_bctc = use_bctc
        self.model_path = model_path
        self.k_neigbors = k_neighbors

        super(GNNDynamicPlanner, self).__init__(self.model, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model_ in self.model:
            model_.to(self.device)

    def _plan(self, env, start, goal, timeout, **kwargs):
        assert not self.loaded and self.model_path is None, "model not loaded and model_path is None"

        if not self.loaded and self.model_path is not None:
            self.load_model(self.model_path)

        seed_everything(seed=0)

        for model_ in self.model:
            model_.eval()


        model_gnn, model_head, model_pe_gnn, model_pe_head = self.model

        self.speed = 1/(len(env.object.trajectory.waypoints)-1)

        path = self._explore(env, start, goal,
                         model_gnn, model_head, model_pe_gnn, model_pe_head, t_max=self.max_num_samples, k=self.k_neigbors, n_sample=self.num_samples_add)


        return create_dot_dict(solution=path)

    def create_graph(self, points, k):
        edge_index = knn_graph(torch.FloatTensor(points), k=k, loop=True)
        edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
        edge_index_torch, _ = coalesce(edge_index, None, len(points), len(points))
        edge_index = edge_index_torch.data.cpu().numpy().T
        edge_cost = defaultdict(list)
        edges = defaultdict(list)
        for i, edge in enumerate(edge_index):
            edge_cost[edge[1]].append(np.linalg.norm(points[edge[1]] - points[edge[0]]))

            edges[edge[1]].append(edge[0])

        self.edges = edges
        self.edge_index = edge_index
        self.edge_cost = edge_cost

    def create_data(self, points, edge_index=None, k=50):
        goal_index = -1
        data = Data(goal=torch.FloatTensor(points[goal_index]))
        data.v = torch.FloatTensor(points)

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

    @staticmethod
    def to_np(tensor):
        return tensor.data.cpu().numpy()

    def _explore(self, env, start, goal, model_gnn, model_head, model_pe_gnn, model_pe_head, t_max, k, n_sample):
        points = [start] + env.robot.sample_n_free_points(self.num_samples) + [goal]
        self.create_graph(points, k)

        data = self.create_data(points, k=k)

        success = False
        path = []
        while not success and (len(points) - 2) <= t_max:

            x = torch.arange(len(env.obs_points))
            pe = model_pe_gnn(x).to(self.device)

            edge_feat, node_feat = model_gnn(**data.to(self.device).to_dict(),
                                             pos_enc=pe,
                                             obstacles=torch.FloatTensor(env.obs_points).to(self.device),
                                             loop=5)

            # explore using the head network
            cur_node = 0
            prev_node = -1
            time_tick = 0

            costs = {0: 0.}
            path = [(0, 0)]  # (node, time_cost)

            success = False
            stay_counter = 0
            ######### explore the graph ########
            while success == False:
                nonzero_indices = torch.where(edge_feat[cur_node, :, :] != 0)[0].unique()
                if nonzero_indices.size()[0] == 0:
                    print('hello')
                edges = edge_feat[cur_node, nonzero_indices, :]
                offsets = torch.LongTensor([-2, -1, 0, 1, 2])
                time_window = torch.clip(int(time_tick) + offsets, 0, env.obs_points.shape[0] - 1)
                obs = torch.FloatTensor(env.obs_points[time_window].flatten())[None, :].repeat(edges.shape[0], 1).to(
                    self.device)
                pos_enc_tw = model_pe_head(time_window).flatten()[None, :].repeat(edges.shape[0], 1).to(self.device)

                policy = model_head(edges, obs, pos_enc_tw).flatten()  # [N, 32*4]
                policy = policy.cpu()

                mask = torch.arange(len(policy)).tolist()
                candidates = nonzero_indices.tolist()

                success_one_step = False

                while len(mask) != 0:  # select non-collision edge with priority
                    idx = mask[policy[mask].argmax()]
                    next_node = candidates[idx]

                    if next_node == prev_node:
                        mask.remove(idx)
                        continue

                    ############ Take the step #########
                    if env._edge_fp(self.to_np(data.v[cur_node]), self.to_np(data.v[next_node]), time_tick):
                        # step forward
                        success_one_step = True

                        dist = np.linalg.norm(self.to_np(data.v[next_node]) - self.to_np(data.v[cur_node]))
                        if dist == 0:  # stay
                            if stay_counter > 0:
                                mask.remove(idx)
                                success_one_step = False
                                continue
                            stay_counter += 1
                            costs[next_node] = costs[cur_node] + self.speed
                            time_tick += 1
                            path.append((next_node, time_tick))
                        else:
                            costs[next_node] = costs[cur_node] + dist
                            time_tick += int(np.ceil(dist / self.speed))
                            path.append((next_node, time_tick))
                            ####### there is no way back ###########
                            edge_feat[:, cur_node, :] = 0
                            stay_counter = 0

                        # update node
                        prev_node = cur_node
                        cur_node = next_node
                        break
                    ############ Search for another feasible edge #########
                    else:
                        mask.remove(idx)

                if success_one_step == False:
                    success = False
                    break

                elif env.in_goal_region(self.to_np(data.v[cur_node])):
                    break

            if not success:
                # print('----------------------------------------resample----------------------------------------!')
                new_points = env.uniform_sample_mine(n_sample)
                original_points = data.v.cpu()
                points = torch.cat(
                    (original_points[:-1, :], torch.FloatTensor(new_points), original_points[[-1], :]), dim=0)
                data.v = points

                edge_index = knn_graph(torch.FloatTensor(data.v), k=k, loop=True)
                edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1).to(self.device)
                data.edge_index, _ = coalesce(edge_index, None, len(data.v), len(data.v))

                # create labels
                labels = torch.zeros(len(data.v), 1).to(self.device)
                labels[-1, 0] = 1
                data.labels = labels
        if not success:
            return None
        else:
            return path


