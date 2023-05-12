import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalingSuper(nn.Module):
    def __init__(self, embed_dim_list):
        super(ScalingSuper, self).__init__()

        self.embed_dim_list = embed_dim_list

        # the largest embed dim
        self.super_embed_dim = max(embed_dim_list)

        # the current sampled embed dim
        self.sample_embed_dim = None

        self.scalings = nn.Parameter(1e-4 * torch.ones(len(embed_dim_list), self.super_embed_dim))

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.sample_idx = self.embed_dim_list.index(sample_embed_dim)

    def forward(self, x):
        return x * self.scalings[self.sample_idx][:self.sample_embed_dim]

    def calc_sampled_param_num(self):
        return 0  #self.scalings[self.sample_idx][:self.sample_embed_dim].numel()

    def get_complexity(self, sequence_length):
        return 0  #sequence_length * self.sample_embed_dim

