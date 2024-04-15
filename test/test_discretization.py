import torch
from torch.autograd import gradcheck
from src.ops.discretization import discretize


class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.discretize = discretize

    def forward(self, dt, A, B):
        return self.discretize(dt, A, B)


if __name__ == "__main__":
    # Initialize the module
    module = TestModule()

    # Create some random input tensors
    dt = torch.randn((), dtype=torch.float64, requires_grad=True)
    A = torch.randn((3, 3), dtype=torch.float64, requires_grad=True)
    B = torch.randn((3, 1), dtype=torch.float64, requires_grad=True)

    # Check the gradients
    gradcheck(module, (dt, A, B), eps=1e-6, atol=1e-4)