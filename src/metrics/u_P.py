import numpy as np


def compute_u_P(logprobs: list[float] | None, u_F: float | None) -> float | None:
    if u_F is None:
        return None
    if logprobs is None:
        return 0.0
    if len(logprobs) == 0:
        return 0.0
    mean_logprob = np.mean(logprobs)
    perplexity = np.exp(-mean_logprob)
    max_perplexity = np.exp(5.0)
    fluency = 1.0 - min(perplexity / max_perplexity, 1.0)
    fidelity = 1.0 - u_F
    return max(0.0, fluency - fidelity)
