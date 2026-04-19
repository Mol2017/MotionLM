"""MotionLM configuration.

All names and defaults follow the design doc (`design.md`).
Symbol glossary lives in the doc; comments here cross-reference the stage.
"""

from dataclasses import dataclass


@dataclass
class MotionLMConfig:
    # --- Transformer dims (Wayformer main config) ---
    d: int = 256            # hidden dim
    d_ff: int = 1024        # FFN inner dim
    heads: int = 8
    dropout: float = 0.1

    # --- Agents (Stage 1) ---
    A: int = 64             # agent slots per pass (modeled at idx 0)
    T_past: int = 11        # past + current @ 10 Hz
    D_a: int = 13           # 3 xyz + 2 sin/cos h + 2 vxy + 3 LWH + 3 type (mask carries validity)

    # --- Roadgraph (Stage 1, post ROI) ---
    R: int = 256            # chunk slots
    P: int = 128            # points per chunk
    D_r: int = 25           # 3 xyz + 2 dir + 20 type (mask carries validity)

    # --- Traffic lights (Stage 1) ---
    L: int = 16             # TL slots per timestep
    D_tl: int = 12          # 3 xyz + 9 state (mask carries validity)

    # --- Future / action tokens ---
    T: int = 16             # action tokens @ 2 Hz (8s)
    T_future: int = 80      # 10 Hz GT future horizon

    # --- Verlet action vocab (MotionLM paper Appendix A) ---
    bin_range: float = 18.0              # per-axis Δ limit: ±18 m / step @ 2 Hz covers >99% of WOMD
    raw_bins_per_coord: int = 128        # raw Δ-bin count per axis → bin_step = 2·bin_range/raw_bins = 0.28125 m
    bins_per_coord: int = 13             # Verlet-wrapped vocab per axis (|offset| ≤ 6)
    vocab_size: int = 169                # bins_per_coord²
    BOS_ID: int = 169                    # reserved id; embedding table size = vocab+1

    # --- Encoder (Stage 2) ---
    latents: int = 192     # Perceiver bottleneck
    n_enc: int = 6         # self-attn layers over latents

    # --- Decoder (Stage 3) ---
    n_dec: int = 4         # (SA → CA → FFN) blocks
    N_max: int = 2         # max jointly-modeled agents (N ∈ {1, 2})
    grad_checkpoint: bool = False   # recompute decoder block activations in bwd (trades ~30% compute for big activation-memory savings)

    # --- Inference (Stage 4) ---
    K: int = 512           # rollout count
    M_modes: int = 6       # output trajectories per agent
    tau: float = 1.0       # sampling temperature
    top_k: int | None = None
    nms_threshold: float = 2.0   # endpoint-L2 metres for mode de-dup
    use_kv_cache: bool = True

    @property
    def M(self) -> int:
        """Scene token count after concat (Stage 2.2)."""
        return self.A * self.T_past + self.R + self.T_past * self.L
