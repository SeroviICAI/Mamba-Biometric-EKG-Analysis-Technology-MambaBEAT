# deep learning libraries
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


# other libraries
from tqdm.auto import tqdm

# own modules
from src.utils.train_functions import train_step, val_step, Accuracy

from src.utils.data import (
    load_ekg_data,
    plot_ekg,
)
from src.utils.torchutils import (
    set_seed,
)

from src.modules.mamba import MambaBEAT

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH: str = "./data/"
N_CLASSES: int = 5


def main() -> None:
    """
    This function is the main program for the training.
    """
    # hyperparameters
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 16
    accuracy: Accuracy = Accuracy()

    # Mamba Hyperparameters
    n_layers: int = 1
    latent_state_dim: int = 16
    expand: int = 2
    dt_rank: int = None
    kernel_size: int = 4
    conv_bias: bool = True
    bias: bool = True
    method: str = "zoh"
    # empty nohup file
    open("nohup.out", "w").close()

    # load data
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _ = load_ekg_data(DATA_PATH, batch_size=batch_size)

    # define name and writer
    name: str = f"model_MambaBEAT_lr_{lr}_bs_{batch_size}_layers_{n_layers}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")
    inputs: torch.Tensor = next(iter(train_data))[0]

    # define model
    model: torch.nn.Module = MambaBEAT(
        inputs.size(2),
        N_CLASSES,
        n_layers,
        latent_state_dim,
        expand,
        dt_rank,
        kernel_size,
        conv_bias,
        bias,
        method,
    ).to(device).to(torch.double)

    # define loss and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # compute the accuracy

    accuracy: Accuracy = Accuracy()

    # define an empty scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    # Train the model
    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
        # call train step
        train_step(
            model, train_data, 0, 1, loss, optimizer, writer, epoch, device, accuracy
        )

        # call val step
        val_step(model, val_data, 0, 1, loss, scheduler, writer, epoch, device)

        # clear the GPU cache
        torch.cuda.empty_cache()

    # save model
    torch.save(model.state_dict(), f"{name}.pth")

    return None


if __name__ == "__main__":
    main()
