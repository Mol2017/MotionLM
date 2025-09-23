import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.config import MotionLMConfig
from models.motion_tokenizer import MotionTokenizer
 

def test_motion_tokenizer():   
    cfg = MotionLMConfig()
    motion_tokenizer = MotionTokenizer(cfg)

    # Test with batch of trajectories [B, T, 2]
    B, T = 4, 16
    
    # Create different trajectory patterns for each batch element
    pos = torch.zeros(B, T, 2)
    
    # Batch 0: straight line
    pos[0, 0] = torch.tensor([0.0, 0.0])
    for t in range(1, T):
        pos[0, t] = pos[0, t-1] + torch.tensor([1.0, 0.0])
    
    # Batch 1: left curve
    pos[1, 0] = torch.tensor([0.0, 0.0])
    for t in range(1, T):
        pos[1, t] = pos[1, t-1] + torch.tensor([1.0, 0.25])
    
    # Batch 2: right curve
    pos[2, 0] = torch.tensor([0.0, 0.0])
    for t in range(1, T):
        pos[2, t] = pos[2, t-1] + torch.tensor([1.0, -0.25])
    
    # Batch 3: zigzag
    pos[3, 0] = torch.tensor([0.0, 0.0])
    for t in range(1, T):
        y_delta = 0.25 if t % 2 == 0 else -0.25
        pos[3, t] = pos[3, t-1] + torch.tensor([1.0, y_delta])

    print(f"Input pos shape: {pos.shape}")

    tokens = motion_tokenizer.xy_to_tokens(pos)
    print(f"Tokens shape (default): {tokens.shape}")  # [4, 16]

    pos_rec = motion_tokenizer.tokens_to_xy(tokens)
    print(f"Reconstructed pos shape: {pos_rec.shape}")  # [4, 16, 2]
    
    # Check reconstruction quality
    mse = torch.mean((pos - pos_rec) ** 2)
    print(f"Reconstruction MSE: {mse.item():.6f}")



if __name__ == "__main__":
    test_motion_tokenizer()
    print("Tokenizer tests passed!")
    