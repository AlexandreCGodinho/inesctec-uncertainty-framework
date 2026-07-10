from sentence_transformers import SentenceTransformer
import numpy as np

_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def compute_u_R(traces: list[str], calibration_max: float = 1.0) -> float:
    if len(traces) < 2:
        return None
    embedder = _get_embedder()
    embeddings = embedder.encode(traces)
    distances = []
    for k in range(1, len(embeddings)):
        dist = np.linalg.norm(embeddings[k] - embeddings[k - 1])
        distances.append(dist)
    mean_dist = np.mean(distances) if distances else 0.0
    if calibration_max > 0:
        return min(mean_dist / calibration_max, 1.0)
    return min(mean_dist, 1.0)
