
import numpy as np

def clip_grad_norm(g: np.ndarray, tau: float) -> np.ndarray:
    g = np.asarray(g, dtype=float)
    norm = np.linalg.norm(g)

    if norm > tau:
        return g * (tau / norm)

    return g
