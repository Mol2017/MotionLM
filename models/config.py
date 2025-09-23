from dataclasses import dataclass

BOS_ID = 169

@dataclass
class MotionLMConfig:
    # token embedding size
    d_model: int = 256 

    # cross attention config
    nhead: int = 8
    ffw_mult: int = 4
    dropout: float = 0.1

    Na_max: int = 64
    Nl_max: int = 256
    S_max: int = 40
    

    # =========================
    # Motion Tokenizer
    # =========================
    bins_per_coord: int = 13
    vocab_size: int = 13 * 13
    max_corr: float = 1.5

    # =========================
    # Scene Encoder
    # =========================
    H_scene_tokens: int = 128
    N_enc_self_attn_layers: int = 4
    
    # =========================
    # Motion Decoder
    # =========================
    H: int = 10              # history steps 1s * 10Hz
    T: int = 16              # prediction steps 8s * 2Hz
    N_max: int = 8                # jointly modeled agents
    N_dec_self_attn_layers: int = 6
