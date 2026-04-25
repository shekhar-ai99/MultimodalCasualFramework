from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from src.config import TrainConfig
from src.data.loader import make_dummy_dataloader
from src.explainability.counterfactual import gated_explanation
from src.models.q_network import QNetwork
from src.models.tgn import TGN
from src.training.evaluate import evaluate_policy
from src.training.train import train
from src.utils.logger import get_logger
from src.utils.visualization import (
    plot_action_distribution,
    plot_conformal_set_sizes,
    plot_loss_curve,
    plot_patient_trajectory,
    plot_q_value_distribution,
)


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_output_dirs() -> tuple[Path, Path, Path]:
    root = Path("results")
    figures = root / "figures"
    logs = root / "logs"
    figures.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return root, figures, logs


def run() -> None:
    cfg = TrainConfig()
    logger = get_logger()

    set_seeds(cfg.seed)
    _, figures_dir, logs_dir = ensure_output_dirs()

    logger.info("Preparing dataloader...")
    dataloader = make_dummy_dataloader(
        num_samples=256,
        state_dim=cfg.input_dim,
        action_dim=cfg.action_dim,
        batch_size=cfg.batch_size,
    )

    logger.info("Initializing TGN + Q-network...")
    tgn = TGN(input_dim=cfg.input_dim, hidden_dim=cfg.hidden_dim)
    q_net = QNetwork(state_dim=cfg.hidden_dim, action_dim=cfg.action_dim)
    optimizer = torch.optim.Adam(list(tgn.parameters()) + list(q_net.parameters()), lr=cfg.lr)

    logger.info("Starting training...")
    loss_history = train(
        tgn,
        q_net,
        dataloader,
        optimizer,
        epochs=cfg.epochs,
        gamma=cfg.gamma,
        alpha=cfg.alpha,
    )
    for i, loss in enumerate(loss_history, start=1):
        logger.info("Epoch %d/%d - Loss: %.4f", i, cfg.epochs, loss)

    logger.info("Running evaluation + conformal analysis...")
    metrics = evaluate_policy(tgn, q_net, dataloader, conformal_alpha=cfg.conformal_alpha)

    # Explanation on a sample state.
    sample_state = torch.randn(1, cfg.input_dim)
    h = torch.zeros(1, cfg.hidden_dim)
    delta_t = torch.ones(1)
    state_embed = tgn(sample_state, delta_t, h)
    sample_q = q_net(state_embed).detach().cpu().numpy().squeeze(0)

    q_hat = metrics["conformal_q_hat"]
    max_q = float(np.max(sample_q))
    sample_cset = [i for i, q in enumerate(sample_q) if q >= max_q - q_hat]
    action, explanation = gated_explanation(sample_q, sample_cset, sample_state, q_net)

    metrics["sample_conformal_set"] = sample_cset
    metrics["sample_selected_action"] = None if action is None else int(action)
    metrics["sample_explanation"] = explanation

    logger.info("Evaluation metrics:")
    for key in [
        "auroc",
        "mean_q_value",
        "q_value_variance",
        "policy_value",
        "conformal_q_hat",
        "conformal_set_size_distribution",
        "action_distribution",
    ]:
        logger.info("  %s: %s", key, metrics[key])

    logger.info("Final explanation: %s", explanation)

    logger.info("Generating figures...")
    plot_loss_curve(loss_history, str(figures_dir / "loss_curve.png"))
    plot_q_value_distribution(metrics["all_q_values"], str(figures_dir / "q_value_distribution.png"))
    plot_action_distribution(metrics["action_distribution"], str(figures_dir / "action_distribution.png"))
    plot_conformal_set_sizes(metrics["conformal_set_sizes"], str(figures_dir / "conformal_set_sizes.png"))
    plot_patient_trajectory(str(figures_dir / "patient_trajectory.png"))

    metrics_path = logs_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    run()
