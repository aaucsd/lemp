import numpy as np
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid



class Attention(torch.nn.Module):

    def __init__(self, embed_size, temperature):
        super(Attention, self).__init__()
        self.temperature = temperature
        self.embed_size = embed_size
        self.key = Lin(embed_size, embed_size, bias=False)
        self.query = Lin(embed_size, embed_size, bias=False)
        self.value = Lin(embed_size, embed_size, bias=False)
        self.layer_norm = torch.nn.LayerNorm(embed_size, eps=1e-6)

    def forward(self, v_code, obs_code):
        v_value = self.value(v_code)
        obs_value = self.value(obs_code)

        v_query = self.query(v_code)

        v_key = self.key(v_code)
        obs_key = self.key(obs_code)

        obs_attention = (v_query @ obs_key.T)
        self_attention = (v_query.reshape(-1) * v_key.reshape(-1)).reshape(-1, self.embed_size).sum(dim=-1)
        whole_attention = torch.cat((self_attention.unsqueeze(-1), obs_attention), dim=-1)
        whole_attention = (whole_attention / self.temperature).softmax(dim=-1)

        v_code_new = (whole_attention.unsqueeze(-1) *
                        torch.cat((v_value.unsqueeze(1), obs_value.unsqueeze(0).repeat(len(v_code), 1, 1)),
                                  dim=1)).sum(dim=1)

        return self.layer_norm(v_code_new + v_code)


class FeedForward(torch.nn.Module):

    def __init__(self, d_in, d_hid):
        super(FeedForward, self).__init__()
        self.w_1 = Lin(d_in, d_hid)  # position-wise
        self.w_2 = Lin(d_hid, d_in)  # position-wise
        self.layer_norm = torch.nn.LayerNorm(d_in, eps=1e-6)

    def forward(self, x):
        residual = x

        x = self.w_2((self.w_1(x)).relu())
        x += residual

        x = self.layer_norm(x)

        return x


class Block(torch.nn.Module):

    def __init__(self, embed_size):
        super(Block, self).__init__()
        self.attention = Attention(embed_size, embed_size ** 0.5)
        self.v_feed = FeedForward(embed_size, embed_size)

    def forward(self, v_code, obs_code):
        v_code = self.attention(v_code, obs_code)
        v_code = self.v_feed(v_code)

        return v_code

class PositionalEncoder(torch.nn.Module):
    def __init__(self, embed_size, min_freq=1e-4, max_seq_len=1024):
        super().__init__()
        self.embed_size = embed_size

        position = torch.arange(max_seq_len)
        freqs = min_freq ** (2 * (torch.arange(self.embed_size) // 2) / self.embed_size)
        pos_enc = position.reshape(-1, 1) * freqs.reshape(1, -1)
        pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
        pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
        ### [seq_len, hidden_size]
        # self.pos_enc = pos_enc
        self.register_buffer('pos_enc',pos_enc)

    def forward(self, time):
        ## time: [N]
        x = self.pos_enc[time, :]

        return x