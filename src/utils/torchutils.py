# deep learning libraries
import torch
import numpy as np

# other libraries
import os
import random


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None


def save_model(model: torch.nn.Module, path: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model: pytorch model.
        path: path of the model
    """

    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    torch.save(model.to("cpu"), path)

    return None


def load_model(path: str) -> torch.nn.Module:
    """
    This function is to load a model from the 'models' folder.

    Args:
        path: path of the model to load.

    Returns:
        model in torchscript.
    """

    # define model
    model: torch.nn.Module = torch.load(path)

    return model
