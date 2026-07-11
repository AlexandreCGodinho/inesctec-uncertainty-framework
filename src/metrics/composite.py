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
    matrix_path: str | None = None,
) -> float | None:
    weights = _load_activation_matrix(matrix_path).get(pattern_id)
    if weights is None:
        return None
    active = []
    for key, w in weights.items():
        value = {
            "w_L": u_L,
            "w_F": u_F,
            "w_P": u_P,
            "w_R": u_R,
            "w_C": u_C,
        }.get(key)
        if value is not None:
            active.append(w * value)
    if not active:
        return None
    return sum(active)


def compute_surface(u_L: float | None, u_F: float | None, u_P: float | None) -> float | None:
    vals = [v for v in [u_L, u_F, u_P] if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)