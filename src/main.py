# deep learning libraries
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# other libraries
from tqdm.auto import tqdm

# own modules
from src.train_functions import (
    train_step,
    val_step,
)
from src.utils import (
    load_ekg_data,
    plot_ekg,
    set_seed,
)

from src.modules.resnet152 import ResNet152

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
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 64
    num_classes: int = 5  # replace with the actual number of classes

    # empty nohup file
    open("nohup.out", "w").close()

    # load data
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _ = load_ekg_data(DATA_PATH, batch_size=batch_size)

    # define name and writer
    name: str = f"model_resnet152_lr_{lr}_bs_{batch_size}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # define model
    model: torch.nn.Module = ResNet152(num_classes).to(device)

    # define loss and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # define an empty scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)

    # Train the model
    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
        # call train step
        train_step(model, train_data, 0, 1, loss, optimizer, writer, epoch, device)

        # call val step
        val_step(model, val_data, 0, 1, loss, scheduler, writer, epoch, device)

        # clear the GPU cache
        torch.cuda.empty_cache()

    # save model
    torch.save(model.state_dict(), f"{name}.pth")

    return None


if __name__ == "__main__":
    main()
