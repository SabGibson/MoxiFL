import random
import numpy.random as rnd


def star_network_dict(n: int) -> dict[str, list[str]]:
    """
    Create adjacency dict for a star network with n devices.
    Device_0 is the center.
    """
    if n < 2:
        raise ValueError("Star network requires at least 2 nodes")

    devices = [f"Device_{i}" for i in range(n)]
    adj: dict[str, list[str]] = {d: [] for d in devices}

    center = devices[0]
    for d in devices[1:]:
        adj[center].append(d)
        adj[d].append(center)

    return adj


def random_network_dict(n: int, p: float = 0.3) -> dict[str, list[str]]:
    """
    Create adjacency dict for a random network with n devices.
    Each edge exists with probability p.
    """
    rnd.seed(1000)
    if not (0 <= p <= 1):
        raise ValueError("Probability p must be between 0 and 1")

    devices = [f"Device_{i}" for i in range(n)]
    adj: dict[str, list[str]] = {d: [] for d in devices}

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[devices[i]].append(devices[j])
                adj[devices[j]].append(devices[i])

    return adj
