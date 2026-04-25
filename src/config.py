"""Project configuration defaults."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    seed: int = 42
    input_dim: int = 16
    hidden_dim: int = 64
    action_dim: int = 8
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 5
    gamma: float = 0.99
    alpha: float = 1.0
    conformal_alpha: float = 0.1
