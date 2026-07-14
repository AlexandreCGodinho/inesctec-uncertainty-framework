from src.metrics import compute_u_L, compute_u_C, compute_u_R, compute_u_P, compute_surface


def test_u_L_no_constraints():
    assert compute_u_L("Any answer") == 0.0


def test_u_L_with_constraints():
    constraints = [{"type": "min_length", "value": 10}]
    assert compute_u_L("Short", constraints) > 0.0
    assert compute_u_L("Long enough answer here", constraints) == 0.0


def test_u_C_single_answer():
    assert compute_u_C(["only one"]) is None


def test_u_C_two_identical():
    u = compute_u_C(["Hello world", "Hello world"])
    assert u is not None
    assert 0.0 <= u <= 1.0


def test_u_R_single_trace():
    assert compute_u_R(["only one"]) is None


def test_u_P_no_logprobs():
    assert compute_u_P(None, 0.5) == 0.0


def test_u_P_empty_logprobs():
    assert compute_u_P([], 0.3) == 0.0


def test_u_P_low_fluency():
    u = compute_u_P([-10.0, -12.0], u_F=0.1)
    assert u >= 0.0


def test_surface_all_none():
    assert compute_surface(None, None, None) is None


def test_surface_partial():
    s = compute_surface(0.1, 0.2, None)
    assert abs(s - 0.15) < 1e-10
