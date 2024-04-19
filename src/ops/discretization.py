import torch


class Discretize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dt, A, B, method="zoh"):
        ctx.save_for_backward(dt, A, B)
        ctx.method = method

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

    @staticmethod
    def backward(ctx, grad_Ad, grad_Bd):
        dt, A, B = ctx.saved_tensors
        method = ctx.method

        E = torch.eye(A.size(0), dtype=A.dtype, device=A.device)
        if method == "zoh":
            # Compute gradients for ZOH method
            ...

        elif method == "bilinear":
            # Compute gradients for Bilinear method
            ...
        else:
            raise ValueError("Invalid method. Choose either 'zoh' or 'bilinear'.")

        return grad_dt, grad_A, grad_B, None


def discretize(dt, A, B, method="zoh"):
    return Discretize.apply(dt, A, B, method)
