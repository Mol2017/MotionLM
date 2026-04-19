import torch

from model.config import MotionLMConfig
from model.motion_decoder import MotionDecoder, build_block_staircase_mask
from model.motionlm import MotionLM
from model.scene_encoder import SceneEncoder


def _tiny_cfg() -> MotionLMConfig:
    """Small config to keep shape tests fast on CPU."""
    return MotionLMConfig(
        d=64, d_ff=128, heads=4,
        A=8, T_past=11, D_a=14,
        R=16, P=16, D_r=26,
        L=4, D_tl=13,
        T=8, T_future=40,
        bins_per_coord=13, vocab_size=169, bin_range=18.0,
        latents=16, n_enc=2, n_dec=2, N_max=2,
        K=4, M_modes=3, tau=1.0, nms_threshold=2.0,
    )


def test_scene_encoder_shapes_and_masks():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    Bp = 3
    enc = SceneEncoder(cfg).eval()

    ah = torch.randn(Bp, cfg.A, cfg.T_past, cfg.D_a)
    am = torch.ones(Bp, cfg.A, cfg.T_past, dtype=torch.bool)
    am[0, 1, :] = False                # agent 1 entirely invalid
    am[1, 0, cfg.T_past // 2 :] = False  # modeled agent partially invalid

    rg = torch.randn(Bp, cfg.R, cfg.P, cfg.D_r)
    rm = torch.ones(Bp, cfg.R, cfg.P, dtype=torch.bool)
    rm[0, 5, :] = False                 # one empty chunk

    tl = torch.randn(Bp, cfg.T_past, cfg.L, cfg.D_tl)
    tm = torch.ones(Bp, cfg.T_past, cfg.L, dtype=torch.bool)

    out = enc(ah, am, rg, rm, tl, tm)
    assert out.shape == (Bp, cfg.latents, cfg.d)
    assert not torch.isnan(out).any()


def test_decoder_shapes_and_staircase_mask():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    B, N, T = 2, 2, cfg.T
    dec = MotionDecoder(cfg).eval()

    tokens = torch.randint(0, cfg.vocab_size, (B, N, T))
    scene = torch.randn(B * N, cfg.latents, cfg.d)
    logits = dec(tokens, scene)
    assert logits.shape == (B, N, T, cfg.vocab_size)
    assert not torch.isnan(logits).any()

    mask = build_block_staircase_mask(N, T, tokens.device)
    # Attention allowed iff t_k <= t_q. Check a few entries.
    def idx(a, t):
        return a * T + t
    # (a=1, t=0) must NOT see (a=0, t=1)
    assert bool(mask[idx(1, 0), idx(0, 1)])
    # (a=0, t=1) must see (a=1, t=1) — same time
    assert not bool(mask[idx(0, 1), idx(1, 1)])
    # (a=0, t=0) must see (a=1, t=0) — same time
    assert not bool(mask[idx(0, 0), idx(1, 0)])


def test_motionlm_train_backward_and_infer():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = MotionLM(cfg)
    B, N = 2, 1

    ah = torch.randn(B, N, cfg.A, cfg.T_past, cfg.D_a)
    am = torch.ones(B, N, cfg.A, cfg.T_past, dtype=torch.bool)
    rg = torch.randn(B, N, cfg.R, cfg.P, cfg.D_r)
    rm = torch.ones(B, N, cfg.R, cfg.P, dtype=torch.bool)
    tl = torch.randn(B, N, cfg.T_past, cfg.L, cfg.D_tl)
    tm = torch.ones(B, N, cfg.T_past, cfg.L, dtype=torch.bool)
    gt = torch.randint(0, cfg.vocab_size, (B, N, cfg.T))
    fv = torch.ones(B, N, cfg.T, dtype=torch.bool)
    x0 = torch.zeros(B, N)
    y0 = torch.zeros(B, N)
    h0 = torch.zeros(B, N)

    batch = dict(
        agent_history=ah, agent_mask=am,
        roadgraph=rg, roadgraph_mask=rm,
        traffic_lights=tl, tl_mask=tm,
        gt_tokens=gt, future_valid=fv,
        x0=x0, y0=y0, h0=h0,
    )

    counts = model.get_parameter_count()
    print(f"params: total={counts['total_parameters']:,} trainable={counts['trainable_parameters']:,}")

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss, logits = model.forward_train(batch)
    assert logits.shape == (B, N, cfg.T, cfg.vocab_size)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    opt.step()


if __name__ == "__main__":
    test_scene_encoder_shapes_and_masks()
    test_decoder_shapes_and_staircase_mask()
    test_motionlm_train_backward_and_infer()
    print("Model tests passed.")
