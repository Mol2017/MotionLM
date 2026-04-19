import torch

from model.config import MotionLMConfig
from model.motionlm import MotionLM


def _tiny_cfg() -> MotionLMConfig:
    return MotionLMConfig(
        d=32, d_ff=64, heads=4,
        A=8, T_past=11, D_a=14,
        R=8, P=8, D_r=26,
        L=2, D_tl=13,
        T=4, T_future=20,
        bins_per_coord=13, vocab_size=169, bin_range=18.0,
        latents=8, n_enc=1, n_dec=1, N_max=2,
        K=4, M_modes=2, tau=1.0, nms_threshold=2.0,
    )


def test_rollout_shapes():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = MotionLM(cfg).eval()
    B, N = 2, 1

    ah = torch.randn(B, N, cfg.A, cfg.T_past, cfg.D_a)
    am = torch.ones(B, N, cfg.A, cfg.T_past, dtype=torch.bool)
    rg = torch.randn(B, N, cfg.R, cfg.P, cfg.D_r)
    rm = torch.ones(B, N, cfg.R, cfg.P, dtype=torch.bool)
    tl = torch.randn(B, N, cfg.T_past, cfg.L, cfg.D_tl)
    tm = torch.ones(B, N, cfg.T_past, cfg.L, dtype=torch.bool)
    x0 = torch.randn(B, N)
    y0 = torch.randn(B, N)
    h0 = torch.randn(B, N) * 0.5

    batch = dict(
        agent_history=ah, agent_mask=am,
        roadgraph=rg, roadgraph_mask=rm,
        traffic_lights=tl, tl_mask=tm,
        x0=x0, y0=y0, h0=h0,
    )

    out = model.forward_infer(batch)
    assert out["sampled_tokens"].shape == (B, cfg.K, N, cfg.T)
    assert out["trajectories_2hz"].shape == (B, cfg.K, N, cfg.T, 2)
    assert out["waypoints_world"].shape == (B, cfg.K, N, cfg.T_future, 2)
    assert out["trajectories_world"].shape == (B, N, cfg.M_modes, cfg.T_future, 2)
    assert out["probs"].shape == (B, N, cfg.M_modes)
    # probs sum to 1 per agent (or all-zero if K=0; our K>0 so sum=1)
    assert torch.allclose(out["probs"].sum(dim=-1), torch.ones(B, N), atol=1e-5)


if __name__ == "__main__":
    test_rollout_shapes()
    print("Inference tests passed.")
