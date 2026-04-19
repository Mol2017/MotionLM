"""Plot MotionLM inference output (M predicted modes) as a GIF.

Given a batch (from the shard loader) and the dict returned by
``MotionLM.forward_infer``, animate 11 past frames + 80 future frames @ 10 Hz
with all ``M_modes`` predicted trajectories drawn simultaneously. Each mode is
colored distinctly and its alpha is scaled by the predicted probability.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from viz.plot_scenario import (
    _draw_agents_at_frame, _draw_frozen_context, _draw_roadgraph,
    _draw_tl_at_frame, _style_axes, _unpack,
)


def _mode_colors(M: int) -> list[str]:
    """Distinct color per mode — uses matplotlib's tab10, enough for M ≤ 10."""
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(M)]


_GT_COLOR = "#111111"


def _draw_gt_trail(ax, gt: np.ndarray, t_upto: int, *, final_marker: bool) -> None:
    """Draw the GT trail up to (and including) future index ``t_upto`` (0-based)."""
    if t_upto < 0:
        return
    trail = gt[: t_upto + 1]
    ax.plot(trail[:, 0], trail[:, 1], "--",
            color=_GT_COLOR, linewidth=1.8, alpha=0.9, zorder=7,
            label="ground truth" if final_marker else None)
    ax.plot([trail[-1, 0]], [trail[-1, 1]], "s",
            color=_GT_COLOR, markersize=7,
            markeredgecolor="white", markeredgewidth=0.8,
            alpha=0.95, zorder=8)


def plot_inference_gif(
    batch: dict,
    infer_out: dict,
    save_path: Path,
    fps: int = 10,
    future_hz: int = 10,
    agent_idx: int = 0,
    gt_trajectory: np.ndarray | None = None,
    gt_valid_len: int | None = None,
) -> None:
    """Animate M predicted modes over past + future frames.

    Args:
        batch: shard-loader batch dict (``[B, N, ...]`` leading dims).
        infer_out: dict from ``MotionLM.forward_infer`` — uses ``trajectories_world``
            (``[B, N, M, T_future, 2]``) and ``probs`` (``[B, N, M]``). When the
            batch has no x0/y0/h0, these are already in agent frame.
        save_path: where to write the .gif.
        fps: playback frame rate.
        future_hz: expected temporal resolution of ``T_future`` (default 10 Hz).
        agent_idx: which agent slot (``N`` axis) to visualize — the modeled
            agent lives at idx 0.
        gt_trajectory: optional ``[T_future, 2]`` agent-frame GT waypoints to
            overlay as a dashed black line.
        gt_valid_len: number of valid GT frames from the start (clips the trail).
    """
    s = _unpack(batch)
    trajs = infer_out["trajectories_world"][0, agent_idx].detach().cpu().numpy()   # [M, Tf, 2]
    probs = infer_out["probs"][0, agent_idx].detach().cpu().numpy()                # [M]
    M, Tf, _ = trajs.shape
    colors = _mode_colors(M)
    order = np.argsort(-probs)                                    # draw low-prob first, top on top
    total = s.Tp + Tf
    gt_cap = Tf if gt_valid_len is None else min(int(gt_valid_len), Tf)

    fig, ax = plt.subplots(figsize=(10, 8))

    def draw(k: int) -> None:
        ax.clear()
        _draw_roadgraph(ax, s.rg, s.rm)

        if k < s.Tp:
            _draw_tl_at_frame(ax, s.tl, s.tm, k)
            _draw_agents_at_frame(ax, s.ah, s.am, k)
            rel = k - (s.Tp - 1)
            stamp = "now" if rel == 0 else f"{rel * 0.1:+.1f}s"
            title = f"Frame {k:>3}/{total}   · past {stamp}   (inference)"
        else:
            _draw_tl_at_frame(ax, s.tl, s.tm, s.Tp - 1)
            _draw_frozen_context(ax, s.ah, s.am, s.Tp - 1)
            t = k - s.Tp

            for m in order:                                       # low → high prob
                color = colors[m]
                trail = trajs[m, : t + 1]
                alpha = 0.25 + 0.7 * float(probs[m])              # readable even for small probs
                ax.plot(trail[:, 0], trail[:, 1], "-",
                        color=color, linewidth=1.6, alpha=alpha, zorder=4)
                ax.plot([trajs[m, t, 0]], [trajs[m, t, 1]], "o",
                        color=color, markersize=8,
                        markeredgecolor="black", markeredgewidth=0.6,
                        alpha=alpha, zorder=5,
                        label=f"mode {m}  p={probs[m]:.2f}" if t == 0 else None)

            if gt_trajectory is not None and gt_cap > 0:
                _draw_gt_trail(ax, gt_trajectory, min(t, gt_cap - 1),
                               final_marker=(t == 0))

            # modeled origin marker
            ax.plot(0, 0, "*", color="#d62728", markersize=12,
                    markeredgecolor="black", markeredgewidth=0.6, zorder=6)
            title = f"Frame {k:>3}/{total}   · future +{(t + 1) / future_hz:.1f}s   (inference, M={M})"
            if t == 0:
                ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

        _style_axes(ax, title)

    anim = FuncAnimation(fig, draw, frames=total, interval=int(1000 / fps))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def plot_inference_scene(
    batch: dict,
    infer_out: dict,
    save_path: Path,
    agent_idx: int = 0,
    gt_trajectory: np.ndarray | None = None,
    gt_valid_len: int | None = None,
) -> None:
    """Static overlay: roadgraph + past trails + all M predicted modes (agent frame).

    Optional ``gt_trajectory`` (``[T_future, 2]`` agent frame) is drawn as a dashed
    black line clipped to ``gt_valid_len``.
    """
    s = _unpack(batch)
    trajs = infer_out["trajectories_world"][0, agent_idx].detach().cpu().numpy()
    probs = infer_out["probs"][0, agent_idx].detach().cpu().numpy()
    M, Tf, _ = trajs.shape
    colors = _mode_colors(M)

    from viz.plot_scenario import _draw_agent_trails, _save

    fig, ax = plt.subplots(figsize=(11, 9))
    _draw_roadgraph(ax, s.rg, s.rm)
    _draw_tl_at_frame(ax, s.tl, s.tm, s.Tp - 1)
    _draw_agent_trails(ax, s.ah, s.am)

    for m in range(M):
        alpha = 0.25 + 0.7 * float(probs[m])
        ax.plot(trajs[m, :, 0], trajs[m, :, 1], "-",
                color=colors[m], linewidth=1.8, alpha=alpha, zorder=5,
                label=f"mode {m}  p={probs[m]:.2f}")
        ax.plot(trajs[m, -1, 0], trajs[m, -1, 1], "o",
                color=colors[m], markersize=7, markeredgecolor="black",
                markeredgewidth=0.5, alpha=alpha, zorder=6)

    if gt_trajectory is not None:
        cap = Tf if gt_valid_len is None else min(int(gt_valid_len), Tf)
        if cap > 0:
            _draw_gt_trail(ax, gt_trajectory, cap - 1, final_marker=True)

    ax.plot(0, 0, "*", color="#d62728", markersize=15, markeredgecolor="black",
            markeredgewidth=0.8, label="modeled @ t=0", zorder=9)

    _style_axes(
        ax, f"Inference: {M} predicted modes (8 s @ 10 Hz) over past context"
    )
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    _save(fig, save_path)
