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


class GNNStaticPlanner(LearnedPlanner):
    def __init__(self, num_batch, model, k_neighbors=50, **kwargs):
        self.num_batch = num_batch
        self.k_neigbors = k_neighbors

        super(GNNStaticPlanner, self).__init__(self.model, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model_ in self.model:
            model_.to(self.device)

    def _plan(self, env, start, goal, timeout, seed=0, **kwargs):

        seed_everything(seed=seed)
        self.model.eval()
        path = self._explore(env, start, goal, self.model, timeout, k=self.k_neigbors, n_sample=self.num_samples_add)


        return create_dot_dict(solution=path)

    def create_graph(self):
        graph_data = knn_graph_from_points(self.points, self.k_neighbors)
        self.edges = graph_data.edges
        self.edge_index = edge_index.edge_index
        self.edge_cost = edge_cost.edge_cost

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

    def _explore(self, env, start, goal, model_gnn, timeout, k, n_sample):
        points = [start] + self.env.robot.sample_n_free_points(self.num_samples) + [goal]
        self.create_graph(points, k)
        data = self.create_data(points, edge_index=self.edge_index, k=k)

        success = False
        path = []
        while not success and (len(points) - 2) <= t_max:

            policy = model_gnn(**data.to(self.device).to_dict(),
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
        
        
        
    @torch.no_grad()
    def _explore(self, env, start, goal, model_gnn, timeout, k, n_sample):

        c0 = env.collision_check_count
        t0 = time()
        forward = 0

        success = False
        path, smooth_path = [], []
        n_batch = batch
        #     n_batch = min(batch, t_max)
        free, collided = env.sample_n_points(n_batch, need_negative=True)
        collided = collided[:len(free)]
        free = [env.init_state] + [env.goal_state] + list(free)

        explored = [0]
        explored_edges = [[0, 0]]
        costs = {0: 0.}
        prev = {0: 0}

        data = create_data(free, collided, env, k)

        # data.edge_index = radius_graph(data.v, radius(len(data.v)), loop=True)
        while not success and (len(free) - 2) <= t_max:

            t1 = time()
            policy = model(**data.to(device).to_dict(), **obs_data(env, free, collided), loop=loop)
            policy = policy.cpu()
            forward += time() - t1

            policy[torch.arange(len(data.v)), torch.arange(len(data.v))] = 0
            policy[:, explored] = 0
            policy[:, data.labels[:, 1] == 1] = 0
            policy[data.labels[:, 1] == 1, :] = 0
            policy[np.array(explored_edges).reshape(2, -1)] = 0
            success = False
            while policy[explored, :].sum() != 0:

                agent = policy[
                    np.array(explored)[torch.where(policy[explored, :] != 0)[0]], torch.where(policy[explored, :] != 0)[
                        1]].argmax()

                end_a, end_b = torch.where(policy[explored, :] != 0)[0][agent], torch.where(policy[explored, :] != 0)[1][
                    agent]
                end_a, end_b = int(end_a), int(end_b)
                end_a = explored[end_a]
                explored_edges.extend([[end_a, end_b], [end_b, end_a]])
                if env._edge_fp(to_np(data.v[end_a]), to_np(data.v[end_b])):
                    explored.append(end_b)
                    costs[end_b] = costs[end_a] + np.linalg.norm(to_np(data.v[end_a]) - to_np(data.v[end_b]))
                    prev[end_b] = end_a

                    policy[:, end_b] = 0
                    if env.in_goal_region(to_np(data.v[end_b])):
                        success = True
                        cost = costs[end_b]
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

            if not success:
                if not smooth:
                    return []

                if (n_batch + len(free) - 2) > t_max:
                    break
                # ----------------------------------------resample----------------------------------------
                new_free, new_collided = env.sample_n_points(n_batch, need_negative=True)
                free = free + list(new_free)
                collided = collided + list(new_collided)
                collided = collided[:len(free)]

                data = create_data(free, collided, env, k)

        c_explore = env.collision_check_count - c0
        c1 = env.collision_check_count
        t1 = time()
        if success and smooth:
            path = list(data.v[path].data.cpu().numpy())
            if smoother == 'model':
                smooth_path = model_smooth(model_s, free, collided, path, env)
            elif smoother == 'oracle':
                smooth_path = joint_smoother(path, env, iter=5)
            else:
                smooth_path = path
        c_smooth = env.collision_check_count - c1
        if smooth:
            total_time = time()
            return {'c_explore': c_explore,
                    'c_smooth': c_smooth,
                    'data': data,
                    'explored': explored,
                    'forward': forward,
                    'total': total_time - t0,
                    'total_explore': t1 - t0,
                    'success': success,
                    't0': t0,
                    'path': path,
                    'smooth_path': smooth_path,
                    'explored_edges': explored_edges}
        else:
            return list(data.v[path].data.cpu().numpy()), free, collided



