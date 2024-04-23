# deep learning libraries
import torch
import torch.nn.functional as F
import math


class SelectiveScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_in, X_in):
        # Store the original sequence length
        original_L = A_in.size(1)
        last_elementA = A_in.transpose(2, 1)[:, :, -1:]
        last_elementX = X_in.transpose(2, 1)[:, :, -1:]

        # Calculate the next power of 2
        next_power_of_2 = 2 ** math.ceil(math.log2(original_L))

        # Extend the sequence lengths to the next power of 2 with zeros
        A = F.pad(A_in, (0, 0, 0, 0, 0, next_power_of_2 - original_L), "constant", 0)
        X = F.pad(X_in, (0, 0, 0, 0, 0, next_power_of_2 - original_L), "constant", 0)
        L = A.size(1)

        # Transpose the input tensors for efficient memory access during computation
        A = A.transpose(2, 1)
        X = X.transpose(2, 1)

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
            (
                X[:, :, 1:original_L],
                last_elementA * X[:, :, original_L - 1:original_L] + last_elementX,
            ),
            dim=2,
        )
        X = X.transpose(2, 1)

        # Save tensors for backward pass
        ctx.save_for_backward(A_in, X)
        return X

    @staticmethod
    def backward(ctx, grad_output):
        A_in, X = ctx.saved_tensors

        # Store the original sequence length
        original_L = grad_output.size(1)

        # Calculate the next power of 2
        next_power_of_2 = 2 ** math.ceil(math.log2(original_L))

        # Extend the sequence lengths to the next power of 2 with zeros
        grad_output = F.pad(
            grad_output, (0, 0, 0, 0, 0, next_power_of_2 - original_L), "constant", 0
        )
        A = F.pad(A_in, (0, 0, 0, 0, 0, next_power_of_2 - original_L), "constant", 0)
        X = F.pad(X, (0, 0, 0, 0, 0, next_power_of_2 - original_L), "constant", 0)

        # Transpose the tensors for efficient memory access during computation
        grad_output = grad_output.transpose(2, 1)
        A = A.transpose(2, 1)
        X = X.transpose(2, 1)

        # Shift A one to the left
        A = F.pad(A[:, :, 1:], (0, 0, 0, 1))
        B, D, L, _ = A.size()

        # Calculate the number of iterations needed
        iterations = int(math.log2(L))

        # Perform the up-sweep operation
        for d in range(iterations):
            indices = torch.arange(0, L, 2 ** (d + 1))
            grad_output[:, :, indices] += (
                A[:, :, indices] * grad_output[:, :, indices + 2**d]
            )
            A[:, :, indices] *= A[:, :, indices + 2**d]

        # Perform the down-sweep operation
        Aa = A
        Xa = grad_output

        for d in range(iterations - 1, -1, -1):
            Aa = A[:, :, 0:L:2**d]
            Xa = grad_output[:, :, 0:L:2**d]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

        # # Perform the down-sweep operation
        # grad_output[:, :, -1] = 0
        # for d in range(iterations - 1, -1, -1):
        #     indices = torch.arange(0, L, 2 ** (d + 1))
        #     t = grad_output[:, :, indices + 2 ** d].clone()
        #     grad_output[:, :, indices + 2 ** d] = grad_output[:, :, indices]
        #     grad_output[:, :, indices] = (
        #         A[:, :, indices] * grad_output[:, :, indices + 2 ** d] + t
        #     )

        # Compute gradient with respect to A
        grad_A = torch.zeros_like(X)
        grad_A[:, :, 1:] = X[:, :, :-1] * grad_output[:, :, 1:]

        # Return back to original dimensions
        return (
            grad_A.transpose(2, 1)[:, :original_L],
            grad_output.transpose(2, 1)[:, :original_L],
        )


def selective_scan(A_in, X_in):
    return SelectiveScan.apply(A_in, X_in)
