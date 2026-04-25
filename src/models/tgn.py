import torch
import torch.nn as nn


class TGN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.message_mlp = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, delta_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        delta_t = torch.log1p(delta_t)
        inp = torch.cat([x, delta_t.unsqueeze(-1)], dim=-1)

        m = self.message_mlp(inp)
        h = self.gru(m, h)

        return h
