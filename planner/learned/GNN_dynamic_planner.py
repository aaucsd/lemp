import torch
from planner.learned.model.GNN_dynamic import GNNet, PolicyHead
from planner.learned.model.base_models import PositionalEncoder
from planner.learned_planner import LearnedPlanner
from utils.utils import seed_everything, create_dot_dict
from utils.graphs import knn_graph_from_points

from collections import OrderedDict, defaultdict
from torch_sparse import coalesce
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data
from torch_geometric.typing import Adj, OptTensor, PairTensor
from tqdm import tqdm as tqdm
import numpy as np

from utils.utils import DotDict

class GNNDynamicPlanner(LearnedPlanner):
    def __init__(self, num_samples, max_num_samples, model_args, use_bctc=False, model_path=None, k_neighbors=50, num_samples_add=200, **kwargs):
        model_args = DotDict(model_args)
        self.model = [
            GNNet(config_size=model_args.config_size, embed_size=model_args.embed_size, obs_size=model_args.obs_size, use_obstacles=True),
            PolicyHead(embed_size=model_args.embed_size),
            PositionalEncoder(embed_size=model_args.embed_size),
            PositionalEncoder(embed_size=model_args.obs_size)]

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

    def _num_node(self):
        ## return nodes on the graph
        return self.num_samples

    def _plan(self, env, start, goal, timeout, **kwargs):
        # assert not self.loaded and self.model_path is None, "model not loaded and model_path is None"

        if not self.loaded and self.model_path is not None:
            self.load_model(self.model_path)

        seed_everything(seed=0)

        for model_ in self.model:
            model_.eval()


        model_gnn, model_head, model_pe_gnn, model_pe_head = self.model

        num_objects = len(env.objects)
        self.len_traj = len(env.objects[0].trajectory.waypoints)


        obs_traj = []
        for i in range(self.len_traj):
            ## set config
            for j in range(num_objects):
                env.objects[j].item.set_config(env.objects[j].trajectory.waypoints[i])
            obs_traj.append(np.concatenate([np.array(env.objects[j].item.get_workspace_observation()) for j in range(num_objects)], axis=0)[:, np.newaxis])

        all_traj = np.concatenate(obs_traj, axis=1).transpose()
        self.obs_traj = all_traj

        self.speed = 1/(self.len_traj/2-1)

        path = self._explore(env, start, goal,
                         model_gnn, model_head, model_pe_gnn, model_pe_head, t_max=self.max_num_samples, k=self.k_neigbors, n_sample=self.num_samples_add)


        return create_dot_dict(solution=path)

    def create_graph(self, points, k):
        graph_data = knn_graph_from_points(points, k)
        self.edges = graph_data.edges
        self.edge_index = graph_data.edge_index
        self.edge_cost = graph_data.edge_cost

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

        data = self.create_data(points, edge_index=self.edge_index, k=k)

        success = False
        path = []
        while not success and (len(points) - 2) <= t_max:

            x = torch.arange(self.len_traj)
            pe = model_pe_gnn(x).to(self.device)

            edge_feat, node_feat = model_gnn(**data.to(self.device).to_dict(),
                                             pos_enc=pe,
                                             obstacles=torch.FloatTensor(self.obs_traj).to(self.device),
                                             loop=5)

            # explore using the head network
            cur_node = 0
            prev_node = -1
            time_tick = 0

            costs = {0: 0.}
            path = [(data.v[0].cpu().numpy(), 0)]  # (node, time_cost)

            success = False
            stay_counter = 0
            ######### explore the graph ########
            while success == False:
                nonzero_indices = torch.where(edge_feat[cur_node, :, :] != 0)[0].unique()
                if nonzero_indices.size()[0] == 0:
                    print('hello')
                edges = edge_feat[cur_node, nonzero_indices, :]
                offsets = torch.LongTensor([-2, -1, 0, 1, 2])
                time_window = torch.clip(int(time_tick) + offsets, 0, self.obs_traj.shape[0] - 1)
                obs = torch.FloatTensor(self.obs_traj[time_window].flatten())[None, :].repeat(edges.shape[0], 1).to(
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
                    dist = np.linalg.norm(self.to_np(data.v[next_node]) - self.to_np(data.v[cur_node]))
                    duration = int(np.ceil(dist / self.speed))
                    if env.edge_fp(self.to_np(data.v[cur_node]), self.to_np(data.v[next_node]), time_tick, time_tick+duration):
                        # step forward
                        success_one_step = True

                        if dist == 0:  # stay
                            if stay_counter > 0:
                                mask.remove(idx)
                                success_one_step = False
                                continue
                            stay_counter += 1
                            costs[next_node] = costs[cur_node] + self.speed
                            time_tick += 1
                            path.append((data.v[next_node].cpu().numpy(), time_tick))
                        else:
                            costs[next_node] = costs[cur_node] + dist
                            time_tick += int(np.ceil(dist / self.speed))
                            path.append((data.v[next_node].cpu().numpy(), time_tick))
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

                elif env.robot.in_goal_region(self.to_np(data.v[cur_node]), goal):
                    success = True
                    break

            if not success:
                # print('----------------------------------------resample----------------------------------------!')
                new_points = env.robot.sample_n_free_points(n_sample)
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
            # print(path)
            return self.generatePath(path)

    def generatePath(self, discrete_path):

        path = []
        for i in range(len(discrete_path)-1):
            path.append(discrete_path[i][0])
            prev_point, prev_t = discrete_path[i][0], discrete_path[i][1]
            next_point, next_t = discrete_path[i+1][0], discrete_path[i+1][1]
            for k in range(1, next_t - prev_t):
                path.append(prev_point + (next_point - prev_point)/np.linalg.norm(next_point - prev_point) * self.speed * k)
        path.append(discrete_path[-1][0])

        return path


