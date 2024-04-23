# deep learning libraries
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


# other libraries
from tqdm.auto import tqdm

# own modules
from src.utils.train_functions import train_step, val_step
from src.utils.metrics import BinaryAccuracy


from src.binary_classification.data import (
    load_ekg_data,
)
from src.utils.torchutils import set_seed, save_model

from src.modules.resnet import ResNet101

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
    epochs: int = 20
    lr: float = 5e-5
    batch_size: int = 128

    # empty nohup file
    open("nohup.out", "w").close()

    # load data
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _ = load_ekg_data(
        DATA_PATH, batch_size=batch_size, num_workers=4
    )

    # define name and writer
    name: str = "binary_resnet101"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # define model
    model: torch.nn.Module = ResNet101(N_CLASSES).to(device).double()

    # define loss and optimizer
    loss: torch.nn.Module = torch.nn.BCELoss()
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # compute the accuracy

    accuracy: BinaryAccuracy = BinaryAccuracy()

    # define an empty scheduler
    scheduler: torch.optim.lr_scheduler.LRScheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    )
    # Train the model
    try:
        for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
            # call train step
            train_step(
                model, train_data, loss, optimizer, writer, epoch, device, accuracy
            )

            # call val step
            val_step(model, val_data, loss, scheduler, writer, epoch, device, accuracy)

            # clear the GPU cache
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        pass

    save_model(model, f"./models/{name}.pth")
    return None


if __name__ == "__main__":
    main()
