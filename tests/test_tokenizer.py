import torch

from model.config import MotionLMConfig
from model.motion_tokenizer import MotionTokenizer


def _make_tokenizer() -> MotionTokenizer:
    return MotionTokenizer(MotionLMConfig())


def test_paper_spec_values():
    tok = _make_tokenizer()
    # MotionLM paper Appendix A defaults
    assert tok.V == 169
    assert tok.C == 6
    assert abs(tok.bin_step - 36.0 / 128.0) < 1e-6, f"bin_step should be 36/128 = 0.28125, got {tok.bin_step}"


def test_pack_unpack():
    tok = _make_tokenizer()
    C = tok.C
    for ox in range(-C, C + 1):
        for oy in range(-C, C + 1):
            t = tok.pack(torch.tensor(ox), torch.tensor(oy))
            assert 0 <= int(t) < tok.V
            ox2, oy2 = tok.unpack(t)
            assert int(ox2) == ox and int(oy2) == oy


def test_zero_action_means_repeat_bin():
    """Paper: 'a zero action indicates that the same delta index should be used as the previous step.'
    A sequence of the token encoding offset (0, 0) with init_bin (bx, 0) should produce a straight
    line at speed bx * bin_step per step."""
    tok = _make_tokenizer()
    zero_tok = tok.pack(torch.tensor(0), torch.tensor(0))
    tokens = torch.full((3, 16), int(zero_tok), dtype=torch.long)

    # With zero init: positions should stay at origin.
    pos = tok.decode(tokens)
    assert pos.abs().max() < 1e-5, "zero tokens + zero init_bin ⇒ stay at origin"

    # With init_bin = (5, 0): constant velocity of 5 bins / step.
    init_bin = torch.tensor([5, 0], dtype=torch.long)
    pos = tok.decode(tokens, init_bin=init_bin)
    expected_step = 5 * tok.bin_step
    for t in range(16):
        assert abs(pos[0, t, 0].item() - (t + 1) * expected_step) < 1e-3
        assert abs(pos[0, t, 1].item()) < 1e-3


def test_round_trip_zero_init_slow_trajectories():
    """With init_bin=0, the encoder has to ramp up — only slow trajectories are accurate.

    Max delta-of-delta per step is C·bin_step ≈ 1.69 m. So trajectories with per-step Δ
    ≤ 1.69 m reconstruct within ~bin_step/2 ≈ 0.14 m.
    """
    tok = _make_tokenizer()
    T = 16
    # Δ = 1 m/step is well within the one-step ramp-up budget.
    pos = torch.stack([torch.tensor([1.0 * (t + 1), 0.0]) for t in range(T)])
    tokens = tok.encode(pos.unsqueeze(0)).squeeze(0)
    rec = tok.decode(tokens.unsqueeze(0)).squeeze(0)
    err = torch.linalg.vector_norm(pos - rec, dim=-1)
    assert err.max() < 3 * tok.bin_step, f"slow-trajectory zero-init error too large: {err.max().item():.3f}"


def test_round_trip_with_init_bin_tracks_any_speed():
    """Seeding init_bin from the observed velocity at t=0 recovers the quantization floor
    (~bin_step/2 ≈ 0.14 m) across walking, city, and highway speeds and curves.
    """
    tok = _make_tokenizer()
    T = 16

    def check(name: str, pos: torch.Tensor, init_vel_per_step: torch.Tensor, tol: float):
        init_bin = torch.round(init_vel_per_step / tok.bin_step).long().unsqueeze(0)
        tokens = tok.encode(pos.unsqueeze(0), init_bin=init_bin).squeeze(0)
        rec = tok.decode(tokens.unsqueeze(0), init_bin=init_bin.squeeze(0)).squeeze(0)
        err = torch.linalg.vector_norm(pos - rec, dim=-1).max().item()
        assert err < tol, f"{name}: err={err:.3f} exceeds tol={tol}"

    for name, step_delta in [("walk 2 m/step", 2.0), ("city 5 m/step", 5.0), ("highway 15 m/step", 15.0)]:
        pos = torch.stack([torch.tensor([step_delta * (t + 1), 0.0]) for t in range(T)])
        check(name, pos, torch.tensor([step_delta, 0.0]), tol=3 * tok.bin_step)

    # Smoothly curving (approximately constant acceleration) — Δ-of-Δ small per step.
    pos = torch.stack([torch.tensor([3.0 * (t + 1), 0.3 * (t + 1)]) for t in range(T)])
    check("gentle left curve (constant velocity)", pos, torch.tensor([3.0, 0.3]), tol=3 * tok.bin_step)


def test_encode_then_decode_shape_contract():
    tok = _make_tokenizer()
    pos = torch.randn(2, 3, 16, 2)
    tokens = tok.encode(pos)
    assert tokens.shape == (2, 3, 16)
    assert tokens.min() >= 0 and tokens.max() < tok.V
    rec = tok.decode(tokens)
    assert rec.shape == (2, 3, 16, 2)


if __name__ == "__main__":
    test_paper_spec_values()
    test_pack_unpack()
    test_zero_action_means_repeat_bin()
    test_round_trip_zero_init_slow_trajectories()
    test_round_trip_with_init_bin_tracks_any_speed()
    test_encode_then_decode_shape_contract()
    print("Tokenizer tests passed.")
