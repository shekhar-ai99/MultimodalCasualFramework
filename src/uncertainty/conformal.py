import numpy as np


class ConformalPredictor:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q_hat = None

    def fit(self, calibration_scores: np.ndarray) -> None:
        self.q_hat = np.quantile(calibration_scores, 1 - self.alpha)

    def predict_set(self, q_values: np.ndarray) -> list[int]:
        if self.q_hat is None:
            raise ValueError("Call fit before predict_set.")
        max_q = np.max(q_values)
        return [a for a, q in enumerate(q_values) if q >= max_q - self.q_hat]
