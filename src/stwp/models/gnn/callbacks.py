"""Training callbacks for GNN models."""

import numpy as np
import torch


class LRAdjustCallback:
    """Learning rate adjustment callback based on validation loss."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        patience: int = 7,
        epsilon: float = 1e-10,
        gamma: float = 0.5,
    ):
        """Initialize the callback.

        Args:
            optimizer: PyTorch optimizer
            patience: Number of epochs to wait before reducing LR
            epsilon: Minimum improvement threshold
            gamma: Factor to reduce LR by
        """
        self.optimizer = optimizer
        self.patience = patience
        self.epsilon = epsilon
        self.gamma = gamma
        self.counter = 0
        self.best_loss = np.inf

    def step(self, loss: float) -> None:
        """Update callback state based on current loss.

        Args:
            loss: Current validation loss
        """
        if loss + self.epsilon * loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\n[Callback] Adjusting lr. Counter: {self.counter}\n")
                self.adjust_learning_rate()
                self.counter = 0

    def adjust_learning_rate(self) -> None:
        """Reduce learning rate by gamma factor."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= self.gamma


class CkptCallback:
    """Checkpoint callback to save best model."""

    def __init__(self, model: torch.nn.Module, path: str = "model_state.pt"):
        """Initialize the callback.

        Args:
            model: PyTorch model to save
            path: Path to save checkpoint
        """
        self.best_loss = np.inf
        self.path = path
        self.model = model

    def step(self, loss: float) -> None:
        """Save model if loss improved.

        Args:
            loss: Current validation loss
        """
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(self.model.state_dict(), self.path)


class EarlyStoppingCallback:
    """Early stopping callback based on validation loss."""

    def __init__(self, patience: int = 40):
        """Initialize the callback.

        Args:
            patience: Number of epochs to wait before stopping
        """
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def step(self, val_loss: float) -> None:
        """Update callback state based on current loss.

        Args:
            val_loss: Current validation loss
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping ....")
