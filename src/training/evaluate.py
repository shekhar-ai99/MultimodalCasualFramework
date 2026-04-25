"""Evaluation routines for policy quality and uncertainty analysis."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.uncertainty.conformal import ConformalPredictor


def evaluate_policy(
    model: torch.nn.Module,
    q_net: torch.nn.Module,
    dataloader,
    conformal_alpha: float = 0.1,
) -> dict[str, Any]:
    """Evaluate policy metrics and conformal set statistics on a dataloader.

    Returns metrics needed for logging and visualization.
    """
    model.eval()
    q_net.eval()

    all_q_values: list[np.ndarray] = []
    all_max_q: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_rewards: list[np.ndarray] = []

    with torch.no_grad():
        for states, actions, rewards, _ in dataloader:
            h = torch.zeros(states.size(0), model.gru.hidden_size, device=states.device)
            delta_t = torch.ones(states.size(0), device=states.device)
            state_embed = model(states, delta_t, h)
            q_values = q_net(state_embed)

            all_q_values.append(q_values.cpu().numpy())
            all_max_q.append(q_values.max(dim=1).values.cpu().numpy())
            all_actions.append(actions.cpu().numpy())
            all_rewards.append(rewards.cpu().numpy())

    q_values_np = np.concatenate(all_q_values, axis=0)
    max_q_np = np.concatenate(all_max_q, axis=0)
    actions_np = np.concatenate(all_actions, axis=0)
    rewards_np = np.concatenate(all_rewards, axis=0)

    # Dummy binary outcome label for AUROC in scaffold setting.
    outcome_labels = (rewards_np > 0).astype(int)
    try:
        auroc = float(roc_auc_score(outcome_labels, max_q_np))
    except ValueError:
        auroc = float("nan")

    unique_actions, counts = np.unique(actions_np, return_counts=True)
    action_distribution = {int(a): int(c) for a, c in zip(unique_actions, counts)}

    # Build conformal sets from a calibration split of nonconformity scores.
    n = len(max_q_np)
    calib_n = max(1, int(0.2 * n))
    nonconformity = max_q_np - q_values_np[np.arange(n), actions_np]
    calibration_scores = nonconformity[:calib_n]

    conformal = ConformalPredictor(alpha=conformal_alpha)
    conformal.fit(calibration_scores)

    conformal_set_sizes: list[int] = []
    for row in q_values_np:
        cset = conformal.predict_set(row)
        conformal_set_sizes.append(len(cset))

    set_size_counter = Counter(conformal_set_sizes)

    return {
        "auroc": auroc,
        "mean_q_value": float(np.mean(q_values_np)),
        "q_value_variance": float(np.var(q_values_np)),
        "policy_value": float(np.mean(max_q_np)),
        "action_distribution": action_distribution,
        "conformal_q_hat": float(conformal.q_hat),
        "conformal_set_size_distribution": {int(k): int(v) for k, v in sorted(set_size_counter.items())},
        "conformal_set_sizes": conformal_set_sizes,
        "all_q_values": q_values_np.reshape(-1).tolist(),
    }
