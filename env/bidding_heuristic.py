import numpy as np


def bidding_heuristic(N: int, mu: float, alpha=0.09) -> int:
    k = np.arange(N + 1)
    probs = np.exp(-alpha * (k - mu) ** 2)
    probs /= probs.sum()
    return np.random.choice(k, p=probs).item()
