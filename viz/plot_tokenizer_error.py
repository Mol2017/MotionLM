"""Plot motion-tokenizer reconstruction error.

Produces a single figure containing several ground-truth trajectories overlaid
with their tokenizer-reconstructed versions, and a matching per-step L2-error
curve for each. Supports both synthetic mock trajectories and real WOMD tracks
from a tfrecord.

Usage::

    uv run python -m viz.plot_motion_tokenizer_reconstruction_error --source mock
    uv run python -m viz.plot_motion_tokenizer_reconstruction_error --source real \\
        --tfrecord PATH --n 6
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from model import MotionLMConfig
from model.motion_tokenizer import MotionTokenizer


Trajectory = tuple[str, torch.Tensor, torch.Tensor]   # (label, pos[T,2], init_bin[2])


def _round_trip(
    pos: torch.Tensor,
    tokenizer: MotionTokenizer,
    init_bin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    toks = tokenizer.encode(pos.unsqueeze(0), init_bin=init_bin.unsqueeze(0)).squeeze(0)
    rec = tokenizer.decode(toks.unsqueeze(0), init_bin=init_bin).squeeze(0)
    err = torch.linalg.vector_norm(pos - rec, dim=-1)
    return rec, err


def _square_like_bounds(ax, min_ratio: float = 0.6) -> None:
    """Pad axis limits so the shorter data extent is ≥ ``min_ratio`` × the longer.

    Keeps ``aspect='equal'`` intact (trajectory shape unchanged) while ensuring the
    plot area isn't degenerate for near-straight trajectories.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xr, yr = x1 - x0, y1 - y0
    if xr <= 0 or yr <= 0:
        return
    if yr < xr * min_ratio:
        pad = (xr * min_ratio - yr) / 2
        ax.set_ylim(y0 - pad, y1 + pad)
    elif xr < yr * min_ratio:
        pad = (yr * min_ratio - xr) / 2
        ax.set_xlim(x0 - pad, x1 + pad)


def _render_block(
    fig,
    axes_row_traj,
    axes_row_err,
    trajectories: list[Trajectory],
    tokenizer: MotionTokenizer,
    per_axis_floor: float,
    two_d_floor: float,
    row_label: str,
) -> None:
    """Render one (trajectories, errors) row-pair into the given axes arrays."""
    for col, (label, pos, ib) in enumerate(trajectories):
        rec, err = _round_trip(pos, tokenizer, ib)
        ax_t = axes_row_traj[col]
        ax_e = axes_row_err[col]

        ax_t.plot(pos[:, 0], pos[:, 1], "o-", color="#1f77b4",
                  label="ground truth", linewidth=1.6, markersize=5)
        ax_t.plot(rec[:, 0], rec[:, 1], "x--", color="#d62728",
                  label="reconstruction", linewidth=1.4, markersize=7, alpha=0.9)
        ax_t.set_aspect("equal", adjustable="box")
        _square_like_bounds(ax_t, min_ratio=0.6)
        ax_t.set_xlabel("x (m)", fontsize=9); ax_t.set_ylabel("y (m)", fontsize=9)
        title = f"[{row_label}] {label}" if col == 0 else label
        ax_t.set_title(title, fontsize=9)
        ax_t.grid(alpha=0.3); ax_t.legend(fontsize=7, loc="best")
        ax_t.tick_params(labelsize=8)

        ax_e.plot(range(len(err)), err, "o-", color="#d62728",
                  linewidth=1.4, markersize=4)
        ax_e.axhline(per_axis_floor, color="#2ca02c", linestyle=":",
                     label=f"per-axis floor ({per_axis_floor:.3f} m)", linewidth=1.1)
        ax_e.axhline(two_d_floor, color="#ff7f0e", linestyle=":",
                     label=f"2-D floor ({two_d_floor:.3f} m)", linewidth=1.1)
        ax_e.set_xlabel("t (2 Hz step)", fontsize=9)
        ax_e.set_ylabel("L2 error (m)", fontsize=9)
        ax_e.set_title(
            f"L2 error — max={err.max():.3f} m, mean={err.mean():.3f} m", fontsize=9
        )
        ax_e.grid(alpha=0.3); ax_e.legend(fontsize=7, loc="best")
        ax_e.tick_params(labelsize=8)


def plot_motion_tokenizer_reconstruction_error(
    groups: list[tuple[str, list[Trajectory]]],
    save_path: Path,
    tokenizer: MotionTokenizer | None = None,
    suptitle: str | None = None,
) -> None:
    """Render multiple groups of trajectories into a single figure.

    Each ``(group_label, trajectories)`` becomes a pair of rows (trajectories,
    per-step L2 error), producing a ``2 · len(groups) × max(n)`` grid. Ground
    truth is blue, reconstruction is red; trajectory panels are aspect-locked
    and gently padded so near-straight lines don't collapse to a sliver.
    """
    import matplotlib.pyplot as plt

    tokenizer = tokenizer or MotionTokenizer(MotionLMConfig())
    per_axis_floor = tokenizer.bin_step / 2
    two_d_floor = tokenizer.bin_step * math.sqrt(2) / 2

    max_n = max(len(trajs) for _, trajs in groups)
    n_rows = 2 * len(groups)
    fig, axes = plt.subplots(n_rows, max_n,
                             figsize=(4.2 * max_n, 4.0 * n_rows), squeeze=False)
    # trajectory-row cells need ~square aspect; matplotlib gives equal-height rows,
    # which combined with 'equal' axes and _square_like_bounds keeps panels readable.

    for gi, (group_label, trajs) in enumerate(groups):
        row_t = axes[2 * gi]
        row_e = axes[2 * gi + 1]
        _render_block(fig, row_t, row_e, trajs, tokenizer,
                      per_axis_floor, two_d_floor, group_label)
        # hide any unused columns if this group has fewer trajectories
        for col in range(len(trajs), max_n):
            row_t[col].set_visible(False)
            row_e[col].set_visible(False)

    default_suptitle = (
        f"Motion-tokenizer reconstruction error  ·  "
        f"bin_step={tokenizer.bin_step:.4f} m  ·  vocab={tokenizer.V}  ·  "
        f"offset clip ±{tokenizer.C}"
    )
    fig.suptitle(suptitle or default_suptitle, y=1.0, fontsize=12)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------- ground-truth sources ----------

def mock_trajectories(tokenizer: MotionTokenizer) -> list[Trajectory]:
    """Hand-crafted synthetic trajectories covering typical driving patterns."""
    T = 16
    dt = 0.5
    out: list[Trajectory] = []

    def seed(v0: float) -> torch.Tensor:
        return tokenizer.velocity_to_init_bin(v0, 0.0, dt=dt)

    # straight low speed
    v0 = 4.0
    pos = torch.stack([torch.tensor([v0 * dt * (t + 1), 0.0]) for t in range(T)])
    out.append((f"straight @ {v0:.0f} m/s", pos, seed(v0)))

    # straight highway
    v0 = 25.0
    pos = torch.stack([torch.tensor([v0 * dt * (t + 1), 0.0]) for t in range(T)])
    out.append((f"straight @ {v0:.0f} m/s (highway)", pos, seed(v0)))

    # accelerating curve
    v0, a, amp = 8.0, 0.4, 1.5
    pos = torch.zeros(T, 2)
    for k in range(T):
        t = (k + 1) * dt
        pos[k, 0] = v0 * t + 0.5 * a * t * t
        pos[k, 1] = amp * math.sin(2 * math.pi * t / 10.0)
    out.append((f"accel+swerve v0={v0:.0f} m/s", pos, seed(v0)))

    # gentle right turn
    v0 = 10.0
    pos = torch.stack(
        [torch.tensor([v0 * dt * (t + 1), -0.3 * ((t + 1) ** 1.3)]) for t in range(T)]
    )
    out.append((f"right turn @ {v0:.0f} m/s", pos, seed(v0)))

    # moderate deceleration
    v0, a = 18.0, -1.5
    pos = torch.zeros(T, 2)
    for k in range(T):
        t = (k + 1) * dt
        pos[k, 0] = max(0.0, v0 * t + 0.5 * a * t * t)
    out.append((f"braking v0={v0:.0f} m/s a=-1.5", pos, seed(v0)))

    return out


def real_trajectories(tfrecord_path: Path, n: int, tokenizer: MotionTokenizer) -> list[Trajectory]:
    """Pull up to ``n`` valid VEHICLE futures from a WOMD tfrecord."""
    from data.convert import (
        OBJ_VEHICLE, _compute_init_bin, _world_to_agent,
        parse_scenario, read_scenarios,
    )

    out: list[Trajectory] = []
    for blob in read_scenarios(tfrecord_path):
        if len(out) >= n:
            break
        sc = parse_scenario(blob)
        cti = sc["cti"]
        for tidx in sc["tracks_to_predict"]:
            if len(out) >= n:
                break
            track = sc["tracks"][tidx]
            _tid, ot, states = track
            if ot != OBJ_VEHICLE or not states[cti]["valid"]:
                continue
            fut = []
            ok = True
            for k in range(1, 17):
                idx = cti + 5 * k
                if idx >= len(states) or not states[idx]["valid"]:
                    ok = False
                    break
                fut.append(states[idx])
            if not ok:
                continue
            st0 = states[cti]
            x0, y0, h0 = st0["x"], st0["y"], st0["h"]
            cos_h, sin_h = math.cos(h0), math.sin(h0)
            pos = torch.tensor(
                [_world_to_agent(s["x"], s["y"], x0, y0, cos_h, sin_h) for s in fut],
                dtype=torch.float32,
            )
            ib = _compute_init_bin(states, cti, x0, y0, cos_h, sin_h, tokenizer)
            speed = math.hypot(st0["vx"], st0["vy"])
            label = f"track {track[0]}  v₀={speed:.1f} m/s"
            out.append((label, pos, ib))
    if not out:
        raise RuntimeError("no valid vehicle tracks found")
    return out


# ---------- CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrecord", type=Path, default=None,
                    help="If given, include real WOMD trajectories; else mock-only.")
    ap.add_argument("--n", type=int, default=5,
                    help="Number of real trajectories to include.")
    ap.add_argument("--out-dir", type=Path, default=Path("img"))
    args = ap.parse_args()

    tok = MotionTokenizer(MotionLMConfig())
    groups: list[tuple[str, list[Trajectory]]] = [("mock", mock_trajectories(tok))]
    if args.tfrecord is not None:
        groups.append(("real WOMD", real_trajectories(args.tfrecord, args.n, tok)))

    save_path = args.out_dir / "motion_tokenizer_reconstruction_error.png"
    plot_motion_tokenizer_reconstruction_error(groups, save_path, tokenizer=tok)
    print(f"wrote {save_path}")


if __name__ == "__main__":
    main()
