import os
from pathlib import Path
from logging import Logger
from datetime import datetime
from dataclasses import dataclass
from multiprocessing import Queue

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from tqdm import tqdm

from omega.transformer.typing import LRScheduler


@dataclass
class TrainerConfig:
    logging_frequency: int = 100  # Save every N batches
    gradient_clip: float = 1.0  # The maximum gradient clipping norm
    early_stopping_patience: int = 2000  # Number of batches to wait before early stopping
    early_stopping_min_delta: float = 0.01  # Minimum change to qualify as an improvement


class BestCheckpointTrainer:
    def __init__(self, logger: Logger, model: nn.Module, loss_fn: nn.Module,
                 optimizer: optim.Optimizer, scheduler: LRScheduler | None = None,
                 config: TrainerConfig | None = None, destination: Path | None = None):
        self.logger = logger
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or TrainerConfig()
        self.destination = destination
        self.lowest_loss = float("inf")
        self.steps_since_no_improvement = 0

        if self.destination is None:
            self.destination = Path()
            self.logger.info("Defaulting to current working directory for model checkpointing")
        self.destination.mkdir(parents=True, exist_ok=True)

    def save(self, loss: float) -> None:
        timestamp = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "timestamp": timestamp,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        checkpoint_path = self.destination / f"Checkpoint from {timestamp}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}. Loss: {loss:4f}")

    def load(self, destination: Path | None = None) -> None:
        if destination is None:
            destination = Path()
        if not destination.exists():
            self.logger.error(f"No model checkpoint exists at {destination}")

        try:
            checkpoint = torch.load(destination)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            loss = checkpoint["loss"]
            self.logger.info(f"Loaded checkpoint from {destination} with loss {loss:.4f}")
        except KeyError as e:
            self.logger.error(f"Checkpoint at {destination} is corrupt: {e}")

    def step(self, embeddings: Tensor) -> float:
        logits = self.model(embeddings)
        probabilities = torch.softmax(logits, dim=-1)
        loss = self.loss_fn(probabilities, embeddings)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.gradient_clip)

        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.scheduler:
            self.scheduler.step()
        if loss.item() < self.lowest_loss - self.config.early_stopping_min_delta:
            self.lowest_loss = loss.item()
            self.steps_since_no_improvement = 0
            self.save(self.lowest_loss)
        else:
            self.steps_since_no_improvement += 1
        return loss.item()

    def fit(self, queue: Queue[Tensor], epochs: int, batches_per_epoch: int) -> None:
        self.model.train()
        with tqdm(total=epochs, desc="Training " + os.environ.get("OMEGA_MODEL_NAME", "Omega")) as pbar:
            for epoch in range(epochs):
                for idx in range(batches_per_epoch):
                    if self.steps_since_no_improvement >= self.config.early_stopping_patience:
                        self.logger.warning(f"Stopped training at epoch {epoch} at batch {idx} by early stopping")
                    embeddings = queue.get()
                    loss = self.step(embeddings)
                    if idx % self.config.logging_frequency == 0:
                        self.logger.info(f"Epoch {epoch}, batch {idx}, loss: {loss:.4f}")
                    pbar.update(1)
                    pbar.set_postfix({"loss": loss})
                self.logger.info(f"Epoch {epoch} finished")
            self.logger.info(f"Training finished. Best loss: {self.lowest_loss:.4f}")
        self.model.eval()
