# deep learning libraries
import torch.autograd.gradcheck

# own modules
from src.ops.selective_scan import selective_scan
from src.utils.torchutils import set_seed

set_seed(42)
A = torch.randn((6, 6, 6, 6), dtype=torch.double, requires_grad=True, device="cuda")
X = torch.randn((6, 6, 6, 6), dtype=torch.double, requires_grad=True, device="cuda")


if __name__ == "__main__":
    # Use gradcheck to verify the gradients
    test = torch.autograd.gradcheck(selective_scan, (A, X), eps=1e-6, atol=1e-4)
    print(test)  # If the result is True then your gradients are correct
