from __future__ import annotations
import numbers


def compute_gae(
    rewards: list[list[float]],
    values: list[list[float]],
    terminated: list[list[bool]],
    truncated: list[list[bool]],
    gamma: float | list[float],
    lam: float | list[float],
) -> list[list[float]]:
    # Validate input shapes and types
    T = len(rewards)
    if T == 0:
        return []
    
    B = len(rewards[0])
    
    # Extensive input validation
    def validate_input(arr, name, allow_bool=False):
        if not isinstance(arr, list):
            raise ValueError(f"{name} must be a list")
        if len(arr) == 0:
            raise ValueError(f"{name} cannot be empty")
        if len(arr) != T:
            raise ValueError(f"{name} must have length {T}")
        for row in arr:
            if not isinstance(row, list):
                raise ValueError(f"{name} must be a list of lists")
            if len(row) == 0:
                raise ValueError(f"{name} cannot contain empty lists")
            if len(row) != B:
                raise ValueError(f"{name} must have consistent inner list lengths")
            for val in row:
                if allow_bool:
                    if not (val in (0, 1, True, False)):
                        raise ValueError(f"{name} must contain only 0/1 or bool")
                else:
                    if not isinstance(val, numbers.Real):
                        raise ValueError(f"{name} must contain only real numbers")
                    if isinstance(val, bool):
                        raise ValueError(f"{name} cannot contain bool")

    validate_input(rewards, "rewards")
    validate_input(values, "values")
    validate_input(terminated, "terminated", allow_bool=True)
    validate_input(truncated, "truncated", allow_bool=True)

    if len(values) != T + 1:
        raise ValueError("values must have length T+1")

    # Convert gamma and lam to schedules with validation
    def _as_schedule(x: float | list[float], name: str) -> list[float]:
        if isinstance(x, numbers.Real):
            x = float(x)
            if not (0.0 <= x <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
            return [x] * T
        if isinstance(x, list):
            if len(x) != T:
                raise ValueError(f"{name} must have length T")
            converted = []
            for v in x:
                if not isinstance(v, numbers.Real):
                    raise ValueError(f"{name} must contain only real numbers")
                if isinstance(v, bool):
                    raise ValueError(f"{name} cannot contain bool")
                v_float = float(v)
                if not (0.0 <= v_float <= 1.0):
                    raise ValueError(f"{name} values must be in [0, 1]")
                converted.append(v_float)
            return converted
        raise TypeError(f"{name} must be float or list[float]")

    gammas = _as_schedule(gamma, "gamma")
    lams = _as_schedule(lam, "lam")

    # Compute generalized advantage estimation
    advantages = [[0.0] * B for _ in range(T)]
    next_adv = [0.0] * B

    for t in range(T - 1, -1, -1):
        g = gammas[t]
        l = lams[t]
        for b in range(B):
            term = bool(terminated[t][b])
            trunc = bool(truncated[t][b])

            # Value bootstrapping depends only on termination
            bootstrap_value = values[t + 1][b] if not term else 0.0
            delta = rewards[t][b] + g * bootstrap_value - values[t][b]

            # Accumulation of advantage stops on termination
            if not term:
                delta += g * l * next_adv[b]

            next_adv[b] = delta
            advantages[t][b] = next_adv[b]

    return [row[:] for row in advantages]