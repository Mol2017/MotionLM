import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.config import MotionLMConfig
from models.scene_encoder import SceneEncoder
from models.motion_decoder import MotionDecoder
from models.motionlm import MotionLM
 

def test_scene_encoder():   
    torch.manual_seed(0)

    cfg = MotionLMConfig()

    # Dimensions
    F_AGENT, F_LANE, F_TL = 12, 11, 12
    B, H, T = 8, cfg.H, cfg.T
    Na, Nl, S = cfg.Na_max, cfg.Nl_max, cfg.S_max
    Hs = cfg.H_scene_tokens
    D = cfg.d_model
    V = cfg.vocab_size
    
    # Input tensors
    agents_hist = torch.randn(B, Na, H, F_AGENT)
    lanes       = torch.randn(B, Nl, S, F_LANE)
    tls         = torch.randn(B, Nl, H, F_TL)

    agent_hist_valid = torch.ones(B, Na, H, dtype=torch.bool)
    agent_hist_valid[0, 1, :] = False  # Agent 1 in batch 0 is invalid for all time steps
    agent_hist_valid[1, 0, T//2:] = False  # Agent 0 in batch 1 is invalid for second half
    lane_valid  = torch.ones(B, Nl, S, dtype=torch.bool)
    tl_valid    = torch.ones(B, Nl, H, dtype=torch.bool)

    batch = dict(
        agents_hist=agents_hist, 
        lanes=lanes, 
        tls=tls,
        agent_hist_valid=agent_hist_valid,
        lane_valid=lane_valid, 
        tl_valid=tl_valid,
    )

    # Inference
    model = SceneEncoder(cfg, F_AGENT, F_LANE, F_TL)
    model.eval()
    # [B, N, Hs, D]
    mem = model.forward(
        batch['agents_hist'], 
        batch['lanes'], 
        batch['tls'],
        agent_hist_valid=batch['agent_hist_valid'], 
        lane_valid=batch['lane_valid'], 
        tl_valid=batch['tl_valid']
    )

    # Verification
    assert mem.shape == (B, cfg.H_scene_tokens, cfg.d_model)
    assert not torch.isnan(mem).any(), "Scene memory contains NaN values"


def test_motion_decoder():
    torch.manual_seed(0)
    
    cfg = MotionLMConfig()

    # Dimensions
    F_AGENT, F_LANE, F_TL = 12, 11, 12
    B, H, T = 8, cfg.H, cfg.T
    Na, Nl, S = cfg.Na_max, cfg.Nl_max, cfg.S_max
    Hs = cfg.H_scene_tokens
    D = cfg.d_model
    V = cfg.vocab_size

    # Input tensors
    scene_mem = torch.randn(B, Hs, D)
    tokens = torch.randint(0, V, (B, T))

    # Inference
    model = MotionDecoder(cfg)
    model.eval()
    logits = model(tokens, scene_mem)

    assert not torch.isnan(logits).any(), "Logits contains NaN values"


def test_motionlm():
    torch.manual_seed(0)

    cfg = MotionLMConfig()

    # Dimensions
    F_AGENT, F_LANE, F_TL = 12, 11, 12
    B, H, T = 8, cfg.H, cfg.T
    Na, Nl, S = cfg.Na_max, cfg.Nl_max, cfg.S_max
    Hs = cfg.H_scene_tokens
    D = cfg.d_model
    V = cfg.vocab_size
    
    # Input tensors
    agents_hist = torch.randn(B, Na, H, F_AGENT)
    lanes       = torch.randn(B, Nl, S, F_LANE)
    tls         = torch.randn(B, Nl, H, F_TL)

    agent_hist_valid = torch.ones(B, Na, H, dtype=torch.bool)
    agent_hist_valid[0, 1, :] = False  # Agent 1 in batch 0 is invalid for all time steps
    agent_hist_valid[1, 0, T//2:] = False  # Agent 0 in batch 1 is invalid for second half
    lane_valid  = torch.ones(B, Nl, S, dtype=torch.bool)
    tl_valid    = torch.ones(B, Nl, H, dtype=torch.bool)

    batch = dict(
        agents_hist=agents_hist, 
        lanes=lanes, 
        tls=tls,
        agent_hist_valid=agent_hist_valid,
        lane_valid=lane_valid, 
        tl_valid=tl_valid,
    )

    tokens_gt = torch.randint(0, V, (B, T))

    batch = dict(
        agents_hist=agents_hist,
        lanes=lanes,
        tls=tls,
        agent_hist_valid=agent_hist_valid,
        lane_valid=lane_valid,
        tl_valid=tl_valid,
        tokens_gt=tokens_gt
    )

    # Inference
    model = MotionLM(cfg, F_AGENT, F_LANE, F_TL)
    
    # Num of model parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params : {total:,} ({total/1e6:.2f}M)")
    print(f"Trainable    : {trainable:,} ({trainable/1e6:.2f}M)")

    # One training step
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    model.train()
    loss, logits = model.forward_train(batch)
    loss.backward()
    opt.step()
    print(f"Train step → loss={loss.item():.3f}, logits.shape={tuple(logits.shape)}")  # [B,N,T,V]

    # One inference step
    model.eval()
    pred_tokens = model.infer_autoregressive(batch)  # [B,N,T]
    print("Pred tokens shape:", tuple(pred_tokens.shape))


if __name__ == "__main__":
    test_scene_encoder()
    test_motion_decoder()
    test_motionlm()
    print("Model tests passed!")