
from __future__ import annotations


def compute_gae(
    rewards: list[list[float]],
    values: list[list[float]],
    terminated: list[list[bool]],
    truncated: list[list[bool]],
    gamma: float | list[float],
    lam: float | list[float],
) -> list[list[float]]:
    T = len(rewards)
    if len(values) != T + 1:
        raise ValueError("values must have length T+1")
    if T == 0:
        return []

    def _as_schedule(x: float | list[float], name: str) -> list[float]:
        if isinstance(x, (int, float)):
            return [float(x)] * T
        if isinstance(x, list):
            if len(x) != T:
                raise ValueError(f"{name} must have length T")
            return [float(v) for v in x]
        raise TypeError(f"{name} must be float or list[float]")

    gammas = _as_schedule(gamma, "gamma")
    lams = _as_schedule(lam, "lam")

    B = len(rewards[0])
    advantages = [[0.0] * B for _ in range(T)]
    next_adv = [0.0] * B

    for t in range(T - 1, -1, -1):
        g = gammas[t]
        l = lams[t]
        for b in range(B):
            term = bool(terminated[t][b])
            trunc = bool(truncated[t][b])
            delta = rewards[t][b] + g * values[t + 1][b] * (0.0 if (term or trunc) else 1.0) - values[t][b]
            continue_mask = 0.0 if term else (0.5 if trunc else 1.0)
            next_adv[b] = delta + g * l * continue_mask * next_adv[b]
            advantages[t][b] = next_adv[b]

    return [row[:] for row in advantages]

