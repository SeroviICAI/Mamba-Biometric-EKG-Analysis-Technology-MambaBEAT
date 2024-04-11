# deep learning libraries
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

# other libraries
import os
import random
import requests  # type: ignore
import zipfile


class EKGDataset(torch.utils.data.Dataset):
    pass


def load_ekg_data(
    path: str, batch_size: int = 128, num_workers: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    # download folders if they are not present
    if not os.path.isdir(f"{path}"):
        # create main dir
        os.makedirs(f"{path}")

        # download data
        download_data(path)

    return None, None, None


def download_data(path: str) -> None:
    """
    This function downloads the PTB-XL dataset from the internet.

    Args:
        path: path to save the data.
    """

    # define paths
    url: str = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    target_path: str = f"{path}/ptb-xl.zip"

    # download zip file
    response: requests.Response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, "wb") as f:
            f.write(response.raw.read())

    # extract zip file
    with zipfile.ZipFile(target_path, 'r') as zip_ref:
        zip_ref.extractall(path)

    # delete the zip file
    os.remove(target_path)

    return None


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
