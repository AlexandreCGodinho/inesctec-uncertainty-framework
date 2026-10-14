import yaml
import os

# Maps the flat YAML key prefix to its pattern_id, since Pedro's format
# uses pattern1_governed / pattern2_role_pipeline / ... instead of a
# pattern_id field.
_PATTERN_NAME_TO_ID = {
    "pattern1_governed": 1,
    "pattern2_role_pipeline": 2,
    "pattern3_reflexive": 3,
    "pattern4_deliberation": 4,
}


def _load_activation_matrix(path: str | None = None):
    if path is None:
        path = os.path.join(
            os.path.dirname(__file__), "../../config/activation_matrix.yaml"
        )
    path = os.path.normpath(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    return {
        _PATTERN_NAME_TO_ID[name]: weights
        for name, weights in data.items()
        if name in _PATTERN_NAME_TO_ID
    }


def compute_composite(
    pattern_id: int,
    u_L: float | None,
    u_F: float | None,
    u_P: float | None,
    u_R: float | None,
    u_C: float | None,
    dataset: str | None = None,
    matrix_path: str | None = None,
) -> float | None:
    raw_weights = _load_activation_matrix(matrix_path).get(pattern_id)
    if raw_weights is None:
        return None

    component_values = {
        "w_L": u_L,
        "w_F": u_F,
        "w_P": u_P,
        "w_R": u_R,
        "w_C": u_C,
    }

    # Only include components with non-zero nominal weight in the active matrix
    active_weights = {k: w for k, w in raw_weights.items() if w > 0}

    # Conditional deactivation: on FinQA and PopQA, u_L is structurally N/A by task design
    if dataset and dataset.lower() in ("finqa", "popqa"):
        active_weights.pop("w_L", None)

    if not active_weights:
        return None

    # Strict abstention: if ANY active component is missing, return None (no zero imputation)
    for key in active_weights:
        if component_values.get(key) is None:
            return None

    # Re-normalize remaining active weights to sum exactly to 1.0
    total_w = sum(active_weights.values())
    if total_w <= 0:
        return None

    return sum((w / total_w) * component_values[k] for k, w in active_weights.items())


def compute_surface(u_L: float | None, u_F: float | None, u_P: float | None) -> float | None:
    vals = [v for v in [u_L, u_F, u_P] if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)