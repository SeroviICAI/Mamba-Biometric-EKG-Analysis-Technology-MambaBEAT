import torch
import torch.nn.functional as F
import numpy as np
import math

N = 5
A = torch.randint(0, 9, (N, N, N, N))
X = torch.randint(0, 9, (N, N, N, N))


class SelectiveScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_in, X_in):
        # Store the original sequence length
        original_L = A_in.size(1)

        # Calculate the next power of 2
        next_power_of_2 = 2 ** math.ceil(math.log2(original_L))

        # Extend the sequence lengths to the next power of 2 with zeros
        if original_L == next_power_of_2:
            A = A_in.clone()
            X = X_in.clone()
        else:
            A = F.pad(A_in, (0, 0, 0, 0, 0, next_power_of_2 - original_L), "constant", 0)
            X = F.pad(X_in, (0, 0, 0, 0, 0, next_power_of_2 - original_L), "constant", 0)
        L = A.size(1)
        
        A = A.transpose(2, 1)
        X = X.transpose(2, 1)

        # Store last elements
        last_elementA = A_in[:, :, -1]
        last_elementX = X_in[:, :, -1]

        # Calculate the number of iterations needed
        iterations = int(math.log2(L))

        # Perform the up-sweep operation
        for d in range(iterations):
            indices = torch.arange(0, L, 2 ** (d + 1))
            X[:, :, indices + 2 ** (d + 1) - 1] += (
                A[:, :, indices + 2 ** (d + 1) - 1] * X[:, :, indices + 2**d - 1]
            )
            A[:, :, indices + 2 ** (d + 1) - 1] *= A[:, :, indices + 2**d - 1]

        # Perform the down-sweep operation
        X[:, :, -1] = 0
        for d in range(iterations - 1, -1, -1):
            indices = torch.arange(0, X.size(2), 2 ** (d + 1))
            t = X[:, :, indices + 2**d - 1].clone()
            X[:, :, indices + 2**d - 1] = X[:, :, indices + 2 ** (d + 1) - 1]
            X[:, :, indices + 2 ** (d + 1) - 1] = (
                A[:, :, indices + 2**d - 1] * X[:, :, indices + 2 ** (d + 1) - 1] + t
            )
        
        # Remove the first zero elements and add the last elements
        X = torch.cat(
            (X[:, :, 1:L], X[:, :, L - 1:L] * last_elementA + last_elementX), dim=2
        )

        # Save tensors for backward pass
        ctx.save_for_backward(A_in, X)
        return X.transpose(2, 1)[:, :original_L]
    
    @staticmethod
    def backward(ctx, grad_output_in):  
        return


def main():
    print(SelectiveScan.apply(A.clone(), X.clone()))


if __name__ == "__main__":
    main()
