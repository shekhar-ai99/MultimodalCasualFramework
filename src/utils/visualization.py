"""Visualization helpers for training and evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(loss_history: list[float], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
    ax.set_title("Training Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_q_value_distribution(q_values: list[float], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(q_values, bins=30)
    ax.set_title("Q-value Distribution")
    ax.set_xlabel("Q-value")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_action_distribution(action_distribution: dict[int, int], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    actions = sorted(action_distribution.keys())
    counts = [action_distribution[a] for a in actions]
    ax.bar(actions, counts)
    ax.set_title("Action Distribution")
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_conformal_set_sizes(set_sizes: list[int], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    values, counts = np.unique(set_sizes, return_counts=True)
    ax.bar(values, counts)
    ax.set_title("Conformal Set Size Distribution")
    ax.set_xlabel("Set Size")
    ax.set_ylabel("Count")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_patient_trajectory(output_path: str, num_steps: int = 24) -> None:
    time = np.arange(num_steps)
    trajectory = np.sin(time / 4.0) + 0.1 * np.random.randn(num_steps)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(time, trajectory, marker="o")
    ax.set_title("Sample Patient Trajectory (Dummy)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Risk Score")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
