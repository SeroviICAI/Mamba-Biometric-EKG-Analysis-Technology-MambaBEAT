import torch
from einops import rearrange, repeat

import numpy as np


class MambaBlock(torch.nn.Module):
    def __init__(self, in_channels, latent_state_dim, expand, dt_rank, kernel_size, conv_bias, bias):
        super(MambaBlock, self).__init__()
        self.in_channels = in_channels
        self.latent_state_dim = latent_state_dim
        self.expand = expand
        self.dt_rank = dt_rank
        self.kernel_size = kernel_size

        self.expanded_dim = int(self.expand * self.d_model)
        self.in_proj = torch.nn.Linear(self.in_channels, self.expanded_dim * 2, bias=bias)

        self.conv1d = torch.nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            bias=conv_bias,
            kernel_size=kernel_size,
            groups=self.expanded_dim,
            padding=kernel_size - 1,
        )

        self.activation = torch.nn.SiLU()
        
        self.selection = torch.nn.Linear(self.expanded_dim, self.dt_rank + self.latent_state_dim * 2, bias=False)
        self.dt_proj = torch.nn.Linear(self.dt_rank, self.expanded_dim, bias=True)  # Broadcast

        # HiPPO-LegS initialization
        P = torch.sqrt(1 + 2 * torch.arange(self.expanded_dim))
        A = P.unsqueeze(1) * P.unsqueeze(0)
        A = torch.tril(A) - torch.diag(torch.arange(self.expanded_dim))
        self.A = torch.nn.Parameter(-A)

        self.D = torch.nn.Parameter(torch.ones(self.expanded_dim))

        self.out_proj = torch.nn.Linear(self.d_inner, self.d_model, bias=self.bias)

    def forward(self, x):
        pass
