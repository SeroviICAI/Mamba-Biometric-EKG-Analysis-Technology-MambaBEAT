# deep learning libraries
import torch

# own modules
from src.utils import (
    load_ekg_data,
    plot_ekg,
    set_seed,
)

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH: str = "./data/"


def main() -> None:
    """
    This function is the main program for the training.
    """
    # hyperparameters
    batch_size: int = 64
    train_data, val_data, _ = load_ekg_data(DATA_PATH, batch_size=batch_size)
    plot_ekg(train_data)


if __name__ == "__main__":
    main()
