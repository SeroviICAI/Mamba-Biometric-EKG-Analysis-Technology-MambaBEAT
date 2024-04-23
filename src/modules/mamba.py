import torch
import math
from src.ops.selective_scan import selective_scan


class MambaBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_state_dim: int,
        expand: int,
        dt_rank: int,
        kernel_size: int,
        conv_bias: bool,
        bias: bool,
        method: str,
    ):
        super(MambaBlock, self).__init__()
        self.in_channels = in_channels
        self.latent_state_dim = latent_state_dim
        self.expand = expand
        self.dt_rank = dt_rank
        self.kernel_size = kernel_size
        self.method = method

        self.expanded_dim = int(self.expand * self.in_channels)
        self.in_proj = torch.nn.Linear(
            self.in_channels, self.expanded_dim * 2, bias=bias
        )

        self.conv1d = torch.nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            bias=conv_bias,
            kernel_size=kernel_size,
            groups=self.expanded_dim,
            padding=kernel_size - 1,
        )

        self.activation = torch.nn.SiLU()

        self.selection = torch.nn.Linear(
            self.expanded_dim, self.latent_state_dim * 2 + self.dt_rank, bias=False
        )
        self.dt_proj = torch.nn.Linear(
            self.dt_rank, self.expanded_dim, bias=True
        )  # Broadcast

        # S4D Initialization
        A = torch.arange(1, self.latent_state_dim + 1, dtype=torch.double).repeat(
            self.expanded_dim, 1
        )
        self.A_log = torch.nn.Parameter(torch.log(A))
        self.D = torch.nn.Parameter(torch.ones(self.expanded_dim))

        self.out_proj = torch.nn.Linear(self.expanded_dim, self.in_channels, bias=bias)

    def forward(self, x):
        L = x.size(1)
        # Project input x an residual connection z
        x_z = self.in_proj(x)

        # Split expanded x and residual z
        x, z = x_z.chunk(2, dim=-1)

        # pass input through the conv and the non_linearity
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)

        x = self.activation(x)

        # Compute ssm -> ssm(Ad, Bd, C, D) or ssm(A, B, C, D, dt)
        out = self.selective_ssm(x)

        # Activation of residual connection:
        z = self.activation(z)

        # multiply outputs by residual connection
        out *= z

        # and calculate output
        out = self.out_proj(out)

        return out

    def selective_ssm(self, x):
        A = -torch.exp(self.A_log)

        # Get B, C and dt from self.selection
        B_C_dt = self.selection(x)

        # Split the matrix.
        B, C, dt = torch.split(
            B_C_dt, [self.latent_state_dim, self.latent_state_dim, self.dt_rank], dim=-1
        )

        # Broadcast dt with self.dt_proj
        dt = torch.nn.functional.softplus(self.dt_proj(dt))
        Ad, Bd = self.discretize(dt, A, B, self.method)
        hidden = selective_scan(Ad, Bd * x.unsqueeze(-1))

        out = hidden @ C.unsqueeze(-1)

        return out.squeeze(3) + self.D * x

    @staticmethod
    def discretize(dt, A, B, method):
        if method == "zoh":
            # Zero-Order Hold (ZOH) method
            Ad = torch.exp(dt.unsqueeze(-1) * A)
            Bd = dt.unsqueeze(-1) * B.unsqueeze(2)
        elif method == "bilinear":
            raise NotImplementedError
            # TODO: complete the method
            # E = torch.eye(A.size(0), dtype=A.dtype, device=A.device)
            # half_dt_A = 0.5 * dt.unsqueeze(-1) * A
            # Ad = torch.inverse(E - half_dt_A) @ (E + half_dt_A)
            # Bd = torch.inverse(E - half_dt_A) @ dt * B
        else:
            raise ValueError("Invalid method. Choose either 'zoh' or 'bilinear'.")

        return Ad, Bd


class MambaBEAT(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_layers: int = 1,
        latent_state_dim: int = 16,
        expand: int = 2,
        dt_rank: int = None,
        kernel_size: int = 4,
        conv_bias: bool = True,
        bias: bool = True,
        method: str = "zoh",
        dropout: float = 0,
    ):
        super().__init__()

        if dt_rank is None:
            dt_rank = math.ceil(in_channels / latent_state_dim)

        self.layers = torch.nn.Sequential(
            *[
                MambaBlock(
                    in_channels,
                    latent_state_dim,
                    expand,
                    dt_rank,
                    kernel_size,
                    conv_bias,
                    bias,
                    method,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = RMSNorm(in_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # Get last batch of labels
        x = self.layers(x)[:, -1]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return self.softmax(x)


class RMSNorm(torch.nn.Module):
    def __init__(self, size: int, epsilon: float = 1e-5, bias: bool = False):
        super().__init__()

        self.epsilon = epsilon
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.bias = torch.nn.Parameter(torch.zeros(size)) if bias else None

    def forward(self, x):
        normed_x = (
            x
            * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.epsilon)
            * self.weight
        )

        if self.bias is not None:
            return normed_x + self.bias

        return normed_x
