import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Callable
from moxi.src.mlogger import DeviceLogger
import copy


class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


## ---- Utility Functions ---- ##


def create_dataloader(data, labels, batch_size=32, shuffle=False):
    dataset = SimpleDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def is_trained(model, initial_model, tol=1e-5):
    for p1, p2 in zip(model.parameters(), initial_model.parameters()):
        if not torch.allclose(p1.data, p2.data, atol=tol):
            return True
    return False


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    device: str | torch.device = "cpu",
    optimizer: optim.Optimizer | None = None,
    train: bool = True,
) -> float:
    """
    Run one epoch on a dataset.

    Args:
        model: The model to train/evaluate.
        dataloader: DataLoader with (inputs, labels).
        criterion: Loss function.
        device: Device to use ("cpu" or "cuda").
        optimizer: Optimizer (required if train=True).
        train: If True, run training mode; if False, evaluation mode.

    Returns:
        Average loss for the epoch.
    """
    model.to(device)
    model.train() if train else model.eval()

    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(inputs)
        if outputs.shape != labels.shape:
            labels = labels.view_as(outputs)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def simple_train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    criterion: Callable,
    optimizer_cls: type[optim.Optimizer],
    epochs: int = 5,
    device: str | torch.device = "cpu",
    val_dataloader: DataLoader | None = None,
    learning_rate: float = 0.01,
    logger: DeviceLogger | None = None,
):
    """
    Train a model with optional validation, logging metrics to DeviceLogger.
    """
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # ---- Training ----
        avg_train_loss = run_epoch(
            model, train_dataloader, criterion, device, optimizer, train=True
        )
        if logger:
            logger.log("train_loss", avg_train_loss, step=epoch)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}")


def simple_eval_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    device: str | torch.device = "cpu",
    logger: DeviceLogger | None = None,
    step: int | None = None,
) -> float:
    """
    Evaluate a model on a given dataloader.

    Args:
        model: PyTorch model.
        dataloader: DataLoader with (inputs, labels).
        criterion: Loss function.
        device: Device ("cpu" or "cuda").
        logger: Optional DeviceLogger for logging metrics.
        step: Optional step/epoch index for logging.

    Returns:
        Average evaluation loss.
    """
    avg_loss = run_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        optimizer=None,
        train=False,
    )

    if logger:
        logger.log("eval_loss", avg_loss, step=step)

    print(f"[Evaluation] - Loss: {avg_loss:.4f}")
    return avg_loss
