from sentence_transformers import SentenceTransformer
import numpy as np

_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def compute_u_C(agent_answers: list[str]) -> float:
    if len(agent_answers) < 2:
        return None
    embedder = _get_embedder()
    embeddings = embedder.encode(agent_answers)
    K = len(embeddings)
    similarities = []
    for i in range(K):
        for j in range(i + 1, K):
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            similarities.append(sim)
    mean_sim = np.mean(similarities) if similarities else 0.0
    return 1.0 - mean_sim
