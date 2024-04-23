# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from typing import Optional

from src.utils.metrics import Accuracy


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
    accuracy: Accuracy = Accuracy(),
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        train_data: dataloader of train data.
        loss: loss function.
        optimizer: optimizer.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """
    # Training
    model.train()
    losses = []

    for inputs, _, targets in train_data:
        inputs, targets = inputs.to(device), targets.to(device)

        # forward
        outputs = model(inputs)

        # Compute loss
        loss_value = loss(outputs, targets)
        losses.append(loss_value.item())

        optimizer.zero_grad()
        loss_value.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        accuracy.update(outputs, targets)

    # Write to tensorboard
    writer.add_scalar("train/loss", np.mean(losses), epoch)
    writer.add_scalar("train/accuracy", accuracy.compute(), epoch)
    accuracy.reset()


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
    accuracy: Accuracy = Accuracy(),
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        val_data: dataloader of validation data.
        loss: loss function.
        scheduler: scheduler.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """
    # Validation
    model.eval()
    losses = []

    with torch.no_grad():
        for inputs, _, targets in val_data:
            inputs, targets = inputs.to(device), targets.to(device)

            # forward
            outputs = model(inputs)

            # Compute loss
            loss_value = loss(outputs, targets)
            losses.append(loss_value.item())

            # Update accuracy
            accuracy.update(outputs, targets)

    if scheduler is not None:
        scheduler.step()

    # Write to tensorboard
    writer.add_scalar("val/loss", np.mean(losses), epoch)
    writer.add_scalar("val/accuracy", accuracy.compute(), epoch)
    accuracy.reset()
