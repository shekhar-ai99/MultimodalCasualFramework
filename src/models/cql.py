import torch


def cql_loss(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_q_values: torch.Tensor,
    gamma: float = 0.99,
    alpha: float = 1.0,
) -> torch.Tensor:
    q_sa = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    max_next_q = next_q_values.max(dim=1)[0]
    target = rewards + gamma * max_next_q

    bellman_loss = ((q_sa - target.detach()) ** 2).mean()

    logsumexp_q = torch.logsumexp(q_values, dim=1)
    dataset_q = q_sa

    cql_reg = (logsumexp_q - dataset_q).mean()

    return bellman_loss + alpha * cql_reg
