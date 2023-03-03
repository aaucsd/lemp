import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid
from torch_geometric.nn import MessagePassing

from planner.learned.model.base_models import Block


class MPNN(MessagePassing):
    def __init__(self, embed_size, aggr: str = 'max', **kwargs):
        super(MPNN, self).__init__(aggr=aggr, **kwargs)
        self.fx = Seq(Lin(embed_size * 4, embed_size), ReLU(), Lin(embed_size, embed_size))

    def forward(self, x, edge_index, edge_attr):
        """"""
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)
        return torch.max(x, out)

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_j - x_i, x_j, x_i, edge_attr], dim=-1)
        values = self.fx(z)
        return values

    def __repr__(self):
        return '{}({}, dim={})'.format(self.__class__.__name__, self.channels,
                                       self.dim)



class GNNet(torch.nn.Module):
    def __init__(self, config_size, embed_size, obs_size, use_obstacles=True):
        super(GNNet, self).__init__()

        self.config_size = config_size
        self.embed_size = embed_size
        self.obs_size = obs_size
        self.use_obstacles = use_obstacles

        # label:1 goal/source
        self.hx = Seq(Lin((config_size + 1) * 4, embed_size),
                      #                           BatchNorm1d(embed_size, track_running_stats=False),
                      ReLU(),
                      Lin(embed_size, embed_size))
        self.hy = Seq(Lin((config_size + 1) * 3, embed_size),
                      #                           BatchNorm1d(embed_size, track_running_stats=True),
                      ReLU(),
                      Lin(embed_size, embed_size))
        self.mpnn = MPNN(embed_size)

        self.obs_node_code = Seq(Lin(obs_size, embed_size), ReLU(), Lin(embed_size, embed_size))
        self.obs_edge_code = Seq(Lin(obs_size, embed_size), ReLU(), Lin(embed_size, embed_size))
        self.node_attentions = torch.nn.ModuleList([Block(embed_size) for _ in range(3)])
        self.edge_attentions = torch.nn.ModuleList([Block(embed_size) for _ in range(3)])

        self.fy = Seq(Lin(embed_size * 3, embed_size), ReLU(),
                      Lin(embed_size, embed_size))
        
        self.policy = Seq(Lin(embed_size, embed_size), ReLU(),
                          Lin(embed_size, 1, bias=False))

    def forward(self, v, labels, obstacles, edge_index, loop, **kwargs):

        # labels: ?goal, one-hot
        v = torch.cat((v, labels), dim=-1)
        goal = v[labels[:, 0] == 1].view(1, -1)
        x = self.hx(torch.cat((v, goal.repeat(len(v), 1), v - goal, (v - goal) ** 2), dim=-1))  # node
        
        vi, vj = v[edge_index[0, :]], v[edge_index[1, :]]
        y = self.hy(torch.cat((vj - vi, vj, vi), dim=-1))  # edge

        if self.use_obstacles:
            obs_node_code = self.obs_node_code(obstacles.view(-1, self.obs_size))
            obs_edge_code = self.obs_edge_code(obstacles.view(-1, self.obs_size))
            
            for na, ea in zip(self.node_attentions, self.edge_attentions):
                x = na(x, obs_node_code)
                y = ea(y, obs_edge_code)

        for i in range(loop):
            x = self.mpnn(x, edge_index, y)
            xi, xj = x[edge_index[0, :]], x[edge_index[1, :]]
            y = torch.max(y, self.fy(torch.cat((xj - xi, xj, xi), dim=-1)))

        policy = self.policy(y)
        policy_output = policy.new_zeros(len(v), len(v))
        policy_output[edge_index[1, :], edge_index[0, :]] = policy.squeeze()
        return policy_output
