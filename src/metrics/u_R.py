import numpy as np

from src.metrics.embeddings import _get_embedder


def compute_u_R(traces: list[str], calibration_max: float = 1.0) -> float | None:
    if len(traces) < 2:
        return None
    embedder = _get_embedder()
    embeddings = np.array(embedder.encode(traces))
    distances = []
    for k in range(1, len(embeddings)):
        dist = np.linalg.norm(embeddings[k] - embeddings[k - 1])
        distances.append(dist)
    mean_dist = np.mean(distances) if distances else 0.0
    if calibration_max > 0:
        return min(mean_dist / calibration_max, 1.0)
    return min(mean_dist, 1.0)
