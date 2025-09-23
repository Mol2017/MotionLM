import torch
import torch.nn as nn

from models.config import MotionLMConfig


class CrossFromLatents(nn.Module):
    """Cross-attn to extract information from input tokens."""
    def __init__(self, d_model, nhead, ffw_mult=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ln1  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(nn.Linear(d_model, ffw_mult*d_model), nn.ReLU(),
                                  nn.Linear(ffw_mult*d_model, d_model))
        self.ln2  = nn.LayerNorm(d_model)
    
    def forward(self, lat, inp, key_padding_mask=None, attn_mask=None):
        """
        Inputs:
        lat:              [B, Hs, D]
        inp:              [B, Na*H+Nl*S+Nl*H, D]
        Masks (optional):
        key_padding_mask: [B, Na*H+Nl*S+Nl*H] True = ignore K/V at that position
        attn_mask:        [Hs, Na*H+Nl*S+Nl*H]

        Returns:
        lat:              [B, Hs, D]
        """
        z,_ = self.attn(query=lat, key=inp, value=inp,
                        key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        lat = self.ln1(lat + z)
        lat = self.ln2(lat + self.ff(lat))
        return lat


class SceneEncoder(nn.Module):
    """
    Early fusion scene encoder with cross-attention and self-attention layers.
    
    The encoder processes the input scene information by 
    1) applying cross-attention to integrate information from different modalities 
       including agents(ego & neighborings), lanes and traffic lights.
    2) applying self-attention to refine the scene representation.
    """
    def __init__(self, cfg: MotionLMConfig, F_agent:int, F_lane:int, F_tl:int):
        super().__init__()
        D = cfg.d_model
        self.D = D
        self.Hs = cfg.H_scene_tokens

        # simple projections to D
        self.agent_proj = nn.Linear(F_agent, D)
        self.lane_proj  = nn.Linear(F_lane,  D)
        self.tl_proj    = nn.Linear(F_tl,    D)

        # token type embeddings
        self.type_embed = nn.Embedding(3, D)  # 0=agent,1=lane,2=tl
        
        # time embeddings
        self.time_embed = nn.Embedding(cfg.H, D)

        # latent slots per ego
        self.latents = nn.Parameter(torch.randn(1, self.Hs, D))

        # one cross-attn (latents read inputs), then self-attn stack on latents
        self.cross_attn = CrossFromLatents(D, cfg.nhead, ffw_mult=cfg.ffw_mult, dropout=cfg.dropout)
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=D, nhead=cfg.nhead,
                                       batch_first=True, dim_feedforward=cfg.ffw_mult*D,
                                       dropout=cfg.dropout)
            for _ in range(cfg.N_enc_self_attn_layers)
        ])

    def forward(self, agents_hist, lanes, tls,
                agent_hist_valid=None, lane_valid=None, tl_valid=None):
        """
        Inputs:
        agents_hist: [B, Na, H, F_agent]
        lanes:       [B, Nl, S, F_lane]
        tls:         [B, Nl, H, F_tl]
        Masks (optional):
        agent_hist_valid: [B, Na, H]
        lane_valid:  [B, Nl, S]
        tl_valid:    [B, Nl, H]

        Returns:
        scene_mem:   [B, Hs, D]
        """
        B, Na, H, _ = agents_hist.shape
        _, Nl, S, Fl = lanes.shape
        _, _, _, Ft = tls.shape
        D, Hs = self.D, self.Hs
        device = agents_hist.device

        # ----- Build agent embedding -----
        # [B, Na, H, D]
        agent_embedding = self.agent_proj(agents_hist)
        # [H]
        time_indices = torch.arange(H, device=device)  
        # [H, D]
        time_embedding = self.time_embed(time_indices)  
        # [B, Na, H, D]
        agent_embedding += time_embedding.view(1, 1, H, D)
        agent_embedding += self.type_embed.weight[0].view(1, 1, 1, D)
        # [B, Na*H, D]
        agent_embedding = agent_embedding.reshape(B, Na*H, D)


        # [B, Na*H], True if padding
        agent_hist_pad = None
        if agent_hist_valid is not None:
            agent_hist_pad = ~agent_hist_valid.reshape(B, Na*H)

        # ----- Build lane embedding -----
        # [B, Nl, S, D]
        lane_embedding = self.lane_proj(lanes)
        lane_embedding += self.type_embed.weight[1].view(1, 1, 1, D)
        # [B, Nl*S, D]
        lane_embedding = lane_embedding.reshape(B, Nl*S, D)

        # [B, Nl*S], True if padding
        lane_pad = None
        if lane_valid is not None:
            lane_pad = ~lane_valid.reshape(B, Nl*S)

        # ----- Build traffic light embedding -----
        # [B, Nl, H, D]
        tl_embedding = self.tl_proj(tls) 
        tl_embedding += self.type_embed.weight[2].view(1, 1, 1, D)
        # [B, Nl*H, D]
        tl_embedding = tl_embedding.reshape(B, Nl*H, D)

        # [B, Nl*H], True if padding
        tl_pad = None
        if tl_valid is not None:
            tl_pad = ~tl_valid.reshape(B, Nl*H)

        # ----- Early fusion all embedding -----
        # [B, Na*H+Nl*S+Nl*H, D]
        inp = torch.cat([agent_embedding, lane_embedding, tl_embedding], dim=1)

        # [B, Na*H+Nl*S+Nl*H]
        if any(m is not None for m in (agent_hist_pad, lane_pad, tl_pad)):
            # if some modality lacks a mask, assume all-valid for that slice
            input_pad = []
            for m, x in zip((agent_hist_pad, lane_pad, tl_pad), 
                            (agent_embedding, lane_embedding, tl_embedding)):
                if m is None:
                    input_pad.append(torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=device))
                else:
                    input_pad.append(m)
            input_pad = torch.cat(input_pad, dim=1)
        else:
            input_pad = None

        # ----- Latent cross-attn -----
        # [B, Hs, D]
        lat = self.latents.expand(B, Hs, D)
        # [B, Hs, D]
        mem = self.cross_attn(lat, inp, key_padding_mask=input_pad)

        # ----- Latent self-attn refinement -----
        for lyr in self.self_attn_layers:
            # [B, Hs, D]
            mem = lyr(mem)

        return mem
    