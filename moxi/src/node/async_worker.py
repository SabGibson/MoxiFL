from .base import AbstractMoxiWorker
from moxi.src.common.interfaces import MoxiNetworkProtocol
import asyncio
import random
import mlflow
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class AsyncMoxiPytorchWorker(AbstractMoxiWorker):
    def __init__(
        self,
        nw_ref,
        device_id,
        model,
        lr,
        n_epochs,
        optimizer,
        train_data=None,
        validation_data=None,
        criterion=nn.MSELoss,
        **kwargs,
    ):
        self.device_id = device_id
        self.n_epochs = n_epochs
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr)
        self._train_data = train_data
        self._validation_data = validation_data
        self.criterion = criterion()
        self.queue = asyncio.Queue()
        self.network_ref = nw_ref

    @property
    def train_data(self):
        return self._train_data

    @train_data.setter
    def train_data(self, new_data):
        self._train_data = new_data

    @property
    def validation_data(self):
        return self._validation_data

    @validation_data.setter
    def validation_data(self, new_data):
        self._validation_data = new_data

    @property
    def peers(self):
        return [
            self.network_ref._graph.nodes[x]["node"]
            for x in self.network_ref._graph.neighbors(self.device_id)
        ]

    async def train_local(self, epochs_per_round=1, grad_clip=1.0):
        """
        Trains locally for a given number of epochs (per round) and returns:
        - state_dict
        - list of per-epoch train losses
        """
        epoch_losses = []
        for epoch in range(epochs_per_round):
            batch_losses = []
            for X, y in self.train_data:
                pred = self.model(X)

                # Check if shapes match
                if pred.shape != y.shape:
                    if isinstance(
                        self.criterion,
                        (
                            torch.nn.MSELoss,
                            torch.nn.BCELoss,
                            torch.nn.BCEWithLogitsLoss,
                            torch.nn.CrossEntropyLoss,
                        ),
                    ):
                        if y.ndim == 1:
                            y = y.view(-1, 1)
                    else:
                        raise ValueError(
                            f"Shape mismatch: pred {pred.shape} vs y {y.shape} for criterion {type(self.criterion).__name__}"
                        )

                if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                    y = y.squeeze(1)
                loss = self.criterion(pred, y)
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                self.optimizer.step()
                batch_losses.append(loss.item())

            epoch_losses.append(sum(batch_losses) / len(batch_losses))
        return self.model.state_dict(), epoch_losses

    async def validate_local(self):
        """
        Computes validation loss for this device.
        """
        if self.validation_data is None:
            return None, None

        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for X, y in self.validation_data:
                pred = self.model(X)

                # Check for shape mismatch
                if pred.shape != y.shape:
                    if isinstance(
                        self.criterion,
                        (
                            torch.nn.MSELoss,
                            torch.nn.BCELoss,
                            torch.nn.BCEWithLogitsLoss,
                            torch.nn.CrossEntropyLoss,
                        ),
                    ):
                        if y.ndim == 1:
                            y = y.view(-1, 1)
                    else:
                        raise ValueError(
                            f"Shape mismatch: pred {pred.shape} vs y {y.shape} for criterion {type(self.criterion).__name__}"
                        )
                if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                    y = y.squeeze(1)
                loss = self.criterion(pred, y)
                val_losses.append(loss.item())

        self.model.train()
        avg_val_loss = sum(val_losses) / len(val_losses)
        return avg_val_loss, None  # Optionally add accuracy later

    async def send_updates(self, target_id, message):
        await self.network_ref._graph.nodes[target_id]["node"].queue.put(message)

    async def compute_updates(self, local_update, lam=0.1):
        peer_updates = []
        while not self.queue.empty():
            update = await self.queue.get()
            peer_updates.append(update)

        if peer_updates:
            penalized = {}
            for k in local_update:
                peer_mean = sum(u[k] for u in peer_updates) / len(peer_updates)
                penalized[k] = local_update[k] - lam * (local_update[k] - peer_mean)
            return penalized
        return local_update
