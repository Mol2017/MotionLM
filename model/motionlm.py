"""MotionLM — end-to-end orchestrator (design doc §Stage 3 train + §Stage 4 infer)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import MotionLMConfig
from model.motion_decoder import MotionDecoder
from model.motion_tokenizer import MotionTokenizer
from model.scene_encoder import SceneEncoder
from model.utils import spline_to_10hz


def _fold_bn(x: torch.Tensor) -> torch.Tensor:
    """Fold [B, N, ...] into [B*N, ...]."""
    return x.reshape(x.size(0) * x.size(1), *x.shape[2:])


# --- inference helpers (formerly model/stage4.py) -------------------------

def _replicate_latents(scene_latents: torch.Tensor, B: int, N: int, K: int) -> torch.Tensor:
    """[B*N, latents, d] → [B*K*N, latents, d] by tiling each (b, n) along K."""
    latents, d = scene_latents.shape[-2:]
    x = scene_latents.view(B, N, latents, d)
    x = x.unsqueeze(1).expand(B, K, N, latents, d).contiguous()
    return x.view(B * K * N, latents, d)


def _ar_sample(
    decoder: MotionDecoder,
    scene_latents_rep: torch.Tensor,
    B: int,
    N: int,
    K: int,
    cfg: MotionLMConfig,
    tau: float,
    top_k: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autoregressive sample, Option B sequential-within-step ordering.

    Returns (sampled_tokens [B, K, N, T], sample_logprobs [B, K, N, T]).
    """
    device = scene_latents_rep.device
    T = cfg.T
    tau_safe = max(float(tau), 1e-6)

    tokens = torch.full((B * K, N, T), cfg.BOS_ID, dtype=torch.long, device=device)
    sampled = torch.zeros(B * K, N, T, dtype=torch.long, device=device)
    logp = torch.zeros(B * K, N, T, device=device)

    for t in range(T):
        for a in range(N):
            logits = decoder(tokens, scene_latents_rep)
            logit = logits[:, a, t, :] / tau_safe
            if top_k is not None and top_k > 0:
                kth = logit.topk(min(top_k, logit.size(-1)), dim=-1).values[:, -1:]
                logit = torch.where(logit < kth, torch.full_like(logit, float("-inf")), logit)
            probs = F.softmax(logit, dim=-1)
            s = torch.multinomial(probs, 1).squeeze(-1)
            sampled[:, a, t] = s
            logp[:, a, t] = torch.log(
                probs.gather(-1, s.unsqueeze(-1)).squeeze(-1).clamp_min(1e-20)
            )
            if t + 1 < T:
                tokens[:, a, t + 1] = s

    return sampled.view(B, K, N, T), logp.view(B, K, N, T)


def _inverse_frame(
    waypoints_agent: torch.Tensor,
    x0: torch.Tensor,
    y0: torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    """Agent-frame [B, K, N, Tf, 2] → world frame, using per-sample (x0, y0, h0)."""
    c = torch.cos(h0)
    s = torch.sin(h0)
    c_ = c.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    s_ = s.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    x = waypoints_agent[..., 0:1]
    y = waypoints_agent[..., 1:2]
    x_world = x * c_ - y * s_ + x0.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    y_world = x * s_ + y * c_ + y0.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    return torch.cat([x_world, y_world], dim=-1)


def _nms_aggregate(
    waypoints_world: torch.Tensor,
    logprobs: torch.Tensor,
    M_modes: int,
    dist_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy NMS per agent on endpoint distance.

    Returns (trajectories [B, N, M_modes, Tf, 2], probs [B, N, M_modes]).
    """
    B, K, N, Tf, _ = waypoints_world.shape
    device = waypoints_world.device
    score = logprobs.sum(dim=-1)

    trajectories = torch.zeros(B, N, M_modes, Tf, 2, device=device)
    probs = torch.zeros(B, N, M_modes, device=device)

    for b in range(B):
        for n in range(N):
            order = torch.argsort(score[b, :, n], descending=True)
            selected_idxs: list[int] = []
            cluster_counts: list[int] = []
            endpoints = waypoints_world[b, :, n, -1, :]
            for idx in order.tolist():
                cand = endpoints[idx]
                placed = False
                for m, sel in enumerate(selected_idxs):
                    if torch.norm(cand - endpoints[sel]) <= dist_thresh:
                        cluster_counts[m] += 1
                        placed = True
                        break
                if not placed:
                    if len(selected_idxs) < M_modes:
                        selected_idxs.append(idx)
                        cluster_counts.append(1)
                    else:
                        dists = torch.stack(
                            [torch.norm(cand - endpoints[s]) for s in selected_idxs]
                        )
                        nearest = int(torch.argmin(dists).item())
                        cluster_counts[nearest] += 1

            for m, sel in enumerate(selected_idxs):
                trajectories[b, n, m] = waypoints_world[b, sel, n]
            counts = torch.tensor(cluster_counts, dtype=torch.float32, device=device)
            pad = M_modes - counts.numel()
            if pad > 0:
                counts = torch.cat([counts, torch.zeros(pad, device=device)])
            probs[b, n] = counts / counts.sum().clamp_min(1.0)

    return trajectories, probs


# --------------------------------------------------------------------------


class MotionLM(nn.Module):
    """Scene encoder + joint multi-agent action-token decoder.

    Batch convention:
        agent_history:  [B, N, A, T_past, D_a]
        agent_mask:     [B, N, A, T_past]
        roadgraph:      [B, N, R, P, D_r]
        roadgraph_mask: [B, N, R, P]
        traffic_lights: [B, N, T_past, L, D_tl]
        tl_mask:        [B, N, T_past, L]
        gt_tokens:      [B, N, T]
        future_valid:   [B, N, T]
        x0, y0, h0:     [B, N]      (optional; needed for inference world-frame)
    """

    def __init__(self, cfg: MotionLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = MotionTokenizer(cfg)
        self.encoder = SceneEncoder(cfg)
        self.decoder = MotionDecoder(cfg)

    # --- shared encode path ---

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run scene encoder. Returns scene_latents [B*N, latents, d]."""
        ah = _fold_bn(batch["agent_history"])
        am = _fold_bn(batch["agent_mask"]) if batch.get("agent_mask") is not None else None
        rg = _fold_bn(batch["roadgraph"])
        rm = _fold_bn(batch["roadgraph_mask"]) if batch.get("roadgraph_mask") is not None else None
        tl = _fold_bn(batch["traffic_lights"])
        tm = _fold_bn(batch["tl_mask"]) if batch.get("tl_mask") is not None else None
        return self.encoder(ah, am, rg, rm, tl, tm)

    # --- training ---

    def forward_train(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        gt_tokens: torch.Tensor = batch["gt_tokens"]
        future_valid: torch.Tensor = batch["future_valid"]
        B, N, T = gt_tokens.shape
        assert T == self.cfg.T

        scene_latents = self.encode(batch)

        bos = torch.full((B, N, 1), self.cfg.BOS_ID, dtype=torch.long, device=gt_tokens.device)
        tokens_in = torch.cat([bos, gt_tokens[..., :-1]], dim=-1)

        logits = self.decoder(tokens_in, scene_latents)

        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            gt_tokens.reshape(-1),
            reduction="none",
        ).view(B, N, T)

        mask = future_valid.to(ce.dtype)
        denom = mask.sum().clamp_min(1.0)
        loss = (ce * mask).sum() / denom
        return loss, logits

    # --- inference ---

    @torch.no_grad()
    def forward_infer(self, batch: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Multi-mode rollout + aggregation (design doc §Stage 4).

        Returns dict with:
            trajectories_world: [B, N, M_modes, T_future, 2] in world frame (if x0/y0/h0 present)
                                or agent-frame otherwise.
            probs:              [B, N, M_modes]
            trajectories_2hz:   [B, K, N, T, 2] agent-frame (pre-aggregation)
            waypoints_world:    [B, K, N, T_future, 2] (pre-NMS, all K rollouts)
            sampled_tokens:     [B, K, N, T]
            sample_logprobs:    [B, K, N, T]
        """
        cfg = self.cfg
        K = kwargs.get("K") if kwargs.get("K") is not None else cfg.K
        tau = kwargs.get("tau") if kwargs.get("tau") is not None else cfg.tau
        top_k = kwargs.get("top_k") if kwargs.get("top_k") is not None else cfg.top_k
        M_modes = kwargs.get("M_modes") if kwargs.get("M_modes") is not None else cfg.M_modes
        dist_thresh = kwargs.get("nms_threshold") if kwargs.get("nms_threshold") is not None else cfg.nms_threshold

        scene_latents = self.encode(batch)
        B = batch["agent_history"].shape[0]
        N = batch["agent_history"].shape[1]

        scene_rep = _replicate_latents(scene_latents, B, N, K)
        sampled, logp = _ar_sample(
            self.decoder, scene_rep, B, N, K, cfg, tau=tau, top_k=top_k
        )

        init_bin = batch.get("init_bin")
        if init_bin is not None:
            ib = init_bin.view(B, 1, N, 2).expand(B, K, N, 2)
        else:
            ib = None
        waypoints_2hz = self.tokenizer.decode(sampled, init_bin=ib)

        waypoints_10hz = spline_to_10hz(waypoints_2hz, cfg.T_future)

        x0 = batch.get("x0")
        y0 = batch.get("y0")
        h0 = batch.get("h0")
        if x0 is not None and y0 is not None and h0 is not None:
            waypoints_world = _inverse_frame(waypoints_10hz, x0, y0, h0)
        else:
            waypoints_world = waypoints_10hz

        trajectories, probs = _nms_aggregate(
            waypoints_world, logp, M_modes=M_modes, dist_thresh=dist_thresh
        )

        return {
            "trajectories_world": trajectories,
            "probs": probs,
            "trajectories_2hz": waypoints_2hz,
            "waypoints_world": waypoints_world,
            "sampled_tokens": sampled,
            "sample_logprobs": logp,
        }

    # --- convenience ---

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        mode: str = "train",
        **kwargs: Any,
    ) -> Any:
        if mode == "train":
            return self.forward_train(batch)
        if mode == "infer":
            return self.forward_infer(batch, **kwargs)
        raise ValueError(f"Unknown mode: {mode}")

    def compute_metrics(
        self,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor | None = None,
    ) -> dict[str, float]:
        gt = batch.get("gt_tokens")
        if gt is None:
            return {}
        if predictions is None:
            out = self.forward_infer(batch)
            predictions = out["sampled_tokens"][:, 0]
        correct = (predictions == gt).float()
        return {
            "token_accuracy": correct.mean().item(),
            "sequence_accuracy": correct.all(dim=-1).float().mean().item(),
        }

    def get_parameter_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}
