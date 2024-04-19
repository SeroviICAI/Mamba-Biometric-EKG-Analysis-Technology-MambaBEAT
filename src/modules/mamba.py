import torch
from einops import rearrange, repeat
from ops.selective_scan import selective_scan
import numpy as np


class MambaBlock(torch.nn.Module):
    def __init__(self, in_channels, latent_state_dim, expand, dt_rank, kernel_size, conv_bias, bias, method):
        super(MambaBlock, self).__init__()
        self.in_channels = in_channels
        self.latent_state_dim = latent_state_dim
        self.expand = expand
        self.dt_rank = dt_rank
        self.kernel_size = kernel_size
        self.method = method

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
        # Project input x an residual connection z
        x_z = self.in_proj(x)

        # Split expanded x and residual z
        x, z = x_z.chunk(2, dim=-1)

        # pass input through the conv and the non_linearity
        x = self.activation(self.conv1d(x))

        # Compute ssm -> ssm(Ad, Bd, C, D) or ssm(A, B, C, D, dt)
        out = self.selective_ssm(x)

        # Activation of residual connection:
        z = self.activation(z)

        # multiply outputs and residual connection
        out *= z

        # and calculate output
        out = self.out_proj(out)

        return out

    def selective_ssm(self, x):
        # Get B, C and dt from self.selection
        B_C_dt = self.selection(x)

        # Split the matrix.
        B, C, dt = torch.split(B_C_dt, [self.expanded_dim, self.expanded_dim, self.dt_rank])

        # Broadcast dt with self.dt_proj
        dt = torch.nn.functional.softplus(self.dt_proj(dt))
        Ad, Bd = self.discretize(dt, self.A, B, self.method)

        hidden = selective_scan(Ad, Bd * x.unsqueeze(-1))

        out = hidden @ C.unsqueeze(-1)

        return out.squeeze(3) + self.D * x

    @staticmethod
    def discretize(dt, A, B, method):
        E = torch.eye(A.size(0), dtype=A.dtype, device=A.device)
        if method == "zoh":
            # Zero-Order Hold (ZOH) method
            Ad = torch.matrix_exp(dt * A)
            Bd = torch.inverse(dt * A) @ (Ad - E) @ (dt * B)
        elif method == "bilinear":
            # Bilinear Method (Tustin’s method)
            ...
        else:
            raise ValueError("Invalid method. Choose either 'zoh' or 'bilinear'.")

        return Ad, Bd
