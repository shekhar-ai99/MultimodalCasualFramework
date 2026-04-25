"""Training routines for TGN + CQL."""

from __future__ import annotations

import torch

from src.models.cql import cql_loss


def train_epoch(model, q_net, dataloader, optimizer, gamma: float = 0.99, alpha: float = 1.0) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    q_net.train()

    running_loss = 0.0
    batches = 0

    for batch in dataloader:
        states, actions, rewards, next_states = batch

        h = torch.zeros(states.size(0), model.gru.hidden_size, device=states.device)
        delta_t = torch.ones(states.size(0), device=states.device)
        state_embed = model(states, delta_t, h)
        next_state_embed = model(next_states, delta_t, h)

        q_values = q_net(state_embed)
        next_q_values = q_net(next_state_embed)

        loss = cql_loss(q_values, actions, rewards, next_q_values, gamma=gamma, alpha=alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batches += 1

    return running_loss / max(1, batches)


def train(model, q_net, dataloader, optimizer, epochs: int, gamma: float = 0.99, alpha: float = 1.0) -> list[float]:
    """Train for multiple epochs and return loss history."""
    loss_history: list[float] = []
    for _ in range(epochs):
        epoch_loss = train_epoch(model, q_net, dataloader, optimizer, gamma=gamma, alpha=alpha)
        loss_history.append(epoch_loss)
    return loss_history
