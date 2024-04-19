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

        self.expanded_dim = int(self.expand * self.in_channels)
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

        self.selection = torch.nn.Linear(self.expanded_dim, self.latent_state_dim * 2 + self.dt_rank, bias=False)
        self.dt_proj = torch.nn.Linear(self.dt_rank, self.expanded_dim, bias=True)  # Broadcast

        # HiPPO-LegS initialization
        P = torch.sqrt(1 + 2 * torch.arange(self.expanded_dim))
        A = P.unsqueeze(1) * P.unsqueeze(0)
        A = torch.tril(A) - torch.diag(torch.arange(self.expanded_dim))
        self.A = torch.nn.Parameter(-A)

        self.D = torch.nn.Parameter(torch.ones(self.expanded_dim))

        self.out_proj = torch.nn.Linear(self.expanded_dim, self.in_channels, bias=bias)

    def forward(self, x):

        # proyect input x an residual connection z
        xz = self.in_proj(x)

        # Split expanded x and residual z
        x, z = xz.chunk(2, dim=-1)

        # pass input through the conv and the non_linearity
        x = self.activation(self.conv1d(x))

        # Get B, C and delta from self.selection
        B_C_delta = self.selection(x)

        # Split the matrix.
        B, C, delta = torch.split(B_C_delta, [self.expanded_dim, self.expanded_dim, self.dt_rank])

        # Broadcast delta with self.dt_proj
        delta = self.dt_proj(delta)
        # ####MAYBE A NON LINEARITY IS NEEDED

        # Ad, Bd = discretize(A, B, delta)

        # Compute ssm -> ssm(Ad, Bd, C, D) or ssm(A, B, C, D, delta)
        out = x

        # Activation of residual connection:
        z = self.activation(z)

        # multiply outputs and residual connection
        out *= z

        # and calculate output
        out = self.out_proj(out)

        return out
