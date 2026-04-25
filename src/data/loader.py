"""Data loading utilities for dummy and processed datasets."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


def make_dummy_dataloader(
    num_samples: int = 256,
    state_dim: int = 16,
    action_dim: int = 8,
    batch_size: int = 32,
) -> DataLoader:
    states = torch.randn(num_samples, state_dim)
    next_states = states + 0.1 * torch.randn_like(states)
    actions = torch.randint(low=0, high=action_dim, size=(num_samples,))
    rewards = torch.randn(num_samples)
    dataset = TensorDataset(states, actions, rewards, next_states)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
