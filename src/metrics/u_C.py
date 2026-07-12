import numpy as np

from src.metrics.embeddings import _get_embedder


def compute_u_C(agent_answers: list[str]) -> float | None:
    if len(agent_answers) < 2:
        return None
    embedder = _get_embedder()
    embeddings = embedder.encode(agent_answers)
    K = len(embeddings)
    similarities = []
    for i in range(K):
        for j in range(i + 1, K):
            norm_i = np.linalg.norm(embeddings[i])
            norm_j = np.linalg.norm(embeddings[j])
            if norm_i == 0.0 or norm_j == 0.0:
                sim = 0.0
            else:
                sim = np.dot(embeddings[i], embeddings[j]) / (norm_i * norm_j)
            similarities.append(sim)
    mean_sim = np.mean(similarities) if similarities else 0.0
    return max(0.0, min(1.0, 1.0 - mean_sim))
