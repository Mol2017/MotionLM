"""Plot a scenario from a shard — optionally side-by-side with the raw tfrecord.

- **Frame plot**: single timestep (default: current time ``t = T_past − 1``).
- **Scene plot**: full temporal sweep + modeled agent's tokenized future.
- **GIF**: animate 11 past + 80 future frames @ 10 Hz.

When ``--tfrecord`` is given, the script also renders the *raw* (pre-conversion)
version of the same scenario+track — same visuals, same agent frame, but **no
ROI clip, no A/R/L slot truncation, no stratified chunking**. Comparing the two
surfaces the information loss that Stage-0 preprocessing introduces.

Usage::

    uv run python -m viz.plot_scenario \\
        --shard /home/wentao/Downloads/shard_00000.pt.zst \\
        --tfrecord /home/wentao/Downloads/uncompressed_scenario_training_training.tfrecord-00000-of-01000 \\
        --sample-idx 0 --out-dir img
"""

from __future__ import annotations

import argparse
import math
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from data import make_loader
from data.frame_norm import ROI_BOX
from model import MotionLMConfig
from model.motion_tokenizer import MotionTokenizer


# ---------- palettes ----------

# Roadgraph type → (color, linewidth, label), indexed by the 20 subtypes
# defined in convert.TYPE_OFFSET. Anything ≥ 20 falls through to "other".
_RG_STYLES: list[tuple[str, float, str]] = (
    [("#bbbbbb", 0.6, "lane")] * 4              # LaneCenter × 4
    + [("#d4a017", 1.0, "road line")] * 9       # RoadLine × 9
    + [("#000000", 1.4, "road edge")] * 3       # RoadEdge × 3
    + [("#e63946", 2.0, "stop sign"),
       ("#06aed5", 1.0, "crosswalk"),
       ("#8338ec", 1.0, "speed bump"),
       ("#fb8500", 0.8, "driveway")]
)
_RG_FALLBACK = ("#cccccc", 0.6, "other")

_TL_STATES: list[tuple[str, str]] = [
    ("UNKNOWN", "#888888"),       ("ARROW_STOP", "#d62728"),
    ("ARROW_CAUTION", "#ff7f0e"), ("ARROW_GO", "#2ca02c"),
    ("STOP", "#d62728"),          ("CAUTION", "#ff7f0e"),
    ("GO", "#2ca02c"),            ("FLASHING_STOP", "#e63946"),
    ("FLASHING_CAUTION", "#ffa600"),
]

_AGENT_STYLES: dict[int, tuple[str, str]] = {
    0: ("VEHICLE", "#1f77b4"),
    1: ("PEDESTRIAN", "#2ca02c"),
    2: ("CYCLIST", "#ff7f0e"),
}


# ---------- unpack & axes helpers ----------

_Scene = namedtuple("_Scene", "ah am rg rm tl tm Tp")


def _unpack(batch: dict) -> _Scene:
    ah = batch["agent_history"][0, 0]
    return _Scene(
        ah=ah,                              am=batch["agent_mask"][0, 0],
        rg=batch["roadgraph"][0, 0],        rm=batch["roadgraph_mask"][0, 0],
        tl=batch["traffic_lights"][0, 0],   tm=batch["tl_mask"][0, 0],
        Tp=ah.shape[1],
    )


def _style_axes(ax, title: str, pad: float = 5.0) -> None:
    (xmin, xmax), (ymin, ymax) = ROI_BOX
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)  (agent-centric, +x = heading)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(alpha=0.2)


# ---------- drawing primitives ----------

def _draw_roadgraph(ax, rg: torch.Tensor, rm: torch.Tensor) -> None:
    seen: set[str] = set()
    for c in range(rg.shape[0]):
        valid = rm[c]
        if not valid.any():
            continue
        pts = rg[c, valid, :2].cpu().numpy()
        idx = int(rg[c, valid, 5:25].float().mean(dim=0).argmax())
        color, lw, name = _RG_STYLES[idx] if idx < len(_RG_STYLES) else _RG_FALLBACK
        label = name if name not in seen else None
        seen.add(name)
        ax.plot(pts[:, 0], pts[:, 1], "-",
                color=color, linewidth=lw, alpha=0.75, label=label)


def _draw_agents_at_frame(ax, ah: torch.Tensor, am: torch.Tensor, frame: int) -> None:
    """Modeled agent (slot 0) styled larger + red edge; context agents by type."""
    seen: set[str] = set()
    for a in range(ah.shape[0]):
        if not am[a, frame]:
            continue
        feats = ah[a, frame].cpu().numpy()
        x, y, sin_h, cos_h = feats[0], feats[1], feats[3], feats[4]
        type_idx = int(feats[10:13].argmax())
        type_name, color = _AGENT_STYLES.get(type_idx, ("?", "#999"))
        is_modeled = (a == 0)
        key = "modeled" if is_modeled else type_name
        label = key if key not in seen else None
        seen.add(key)
        edge = "red" if is_modeled else color
        ax.plot([x], [y], "o", color=color,
                markersize=12 if is_modeled else 6,
                markeredgecolor=edge, markeredgewidth=1.2,
                label=label, zorder=5)
        ax.arrow(x, y, cos_h * 3.0, sin_h * 3.0,
                 head_width=0.6, head_length=0.8, fc=edge, ec=edge,
                 length_includes_head=True, alpha=0.9, zorder=4)


def _draw_frozen_context(ax, ah: torch.Tensor, am: torch.Tensor, frame: int) -> None:
    """Draw non-modeled agents frozen at ``frame`` with reduced alpha (for future frames)."""
    for a in range(1, ah.shape[0]):
        if not am[a, frame]:
            continue
        feats = ah[a, frame].cpu().numpy()
        x, y, sin_h, cos_h = feats[0], feats[1], feats[3], feats[4]
        type_idx = int(feats[10:13].argmax())
        _, color = _AGENT_STYLES.get(type_idx, ("?", "#999"))
        ax.plot([x], [y], "o", color=color, markersize=6,
                markeredgecolor=color, markeredgewidth=0.8, alpha=0.55, zorder=4)
        ax.arrow(x, y, cos_h * 3.0, sin_h * 3.0,
                 head_width=0.5, head_length=0.7, fc=color, ec=color,
                 length_includes_head=True, alpha=0.4, zorder=3)


def _draw_tl_at_frame(ax, tl: torch.Tensor, tm: torch.Tensor, frame: int) -> None:
    seen: set[str] = set()
    for k in range(tl.shape[1]):
        if not tm[frame, k]:
            continue
        feats = tl[frame, k].cpu().numpy()
        name, color = _TL_STATES[int(feats[3:12].argmax())]
        label = name if name not in seen else None
        seen.add(name)
        ax.plot([feats[0]], [feats[1]], "s", color=color, markersize=9,
                markeredgecolor="black", markeredgewidth=0.6,
                label=label, alpha=0.9, zorder=3)


def _draw_agent_trails(ax, ah: torch.Tensor, am: torch.Tensor) -> None:
    for a in range(ah.shape[0]):
        valid = am[a]
        if valid.sum() < 2:
            continue
        pts = ah[a, valid.nonzero(as_tuple=False).squeeze(-1), :2].cpu().numpy()
        is_mod = (a == 0)
        color = "#d62728" if is_mod else "#4c78a8"
        lw, alpha, msize, edge = ((1.8, 0.9, 5, "black") if is_mod
                                  else (1.0, 0.55, 3, color))
        ax.plot(pts[:, 0], pts[:, 1], "-",
                color=color, linewidth=lw, alpha=alpha, zorder=3)
        ax.plot(pts[-1, 0], pts[-1, 1], "o", color=color, markersize=msize,
                markeredgecolor=edge, markeredgewidth=0.6, zorder=4)


# ---------- future decoding / interpolation ----------

def _decode_future(batch: dict, tokenizer: MotionTokenizer) -> torch.Tensor:
    """Verlet-decode stored ``gt_tokens`` → ``[T, 2]`` agent-frame waypoints @ 2 Hz."""
    tok = batch["gt_tokens"][0, 0]
    ib = batch["init_bin"][0, 0]
    return tokenizer.decode(tok.unsqueeze(0), init_bin=ib).squeeze(0)


def _upsample_future(
    fut_2hz: torch.Tensor, future_hz: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """2 Hz future → ``future_hz`` positions + per-step headings (carried over stillness)."""
    T_tok = fut_2hz.shape[0]
    anchored = torch.cat([torch.zeros(1, 2), fut_2hz], dim=0)
    n_fut = T_tok * (future_hz // 2)
    positions = torch.nn.functional.interpolate(
        anchored.T.unsqueeze(0), size=n_fut + 1, mode="linear", align_corners=True,
    )[0].T[1:]                                          # [n_fut, 2]

    deltas = positions - torch.cat([torch.zeros(1, 2), positions[:-1]], dim=0)
    headings = torch.atan2(deltas[:, 1], deltas[:, 0])
    still = deltas.abs().sum(dim=-1) < 1e-6             # stationary → carry previous
    for i in range(n_fut):
        if still[i]:
            headings[i] = 0.0 if i == 0 else headings[i - 1]
    return positions, headings


# ---------- public plots ----------

def plot_frame(batch: dict, save_path: Path, frame: int = -1) -> None:
    """Single-timestep snapshot. ``frame=-1`` = current time (T_past-1)."""
    s = _unpack(batch)
    if frame < 0:
        frame = s.Tp + frame
    rel = frame - (s.Tp - 1)
    stamp = "current" if rel == 0 else f"{rel * 0.1:+.1f}s"

    fig, ax = plt.subplots(figsize=(10, 9))
    _draw_roadgraph(ax, s.rg, s.rm)
    _draw_tl_at_frame(ax, s.tl, s.tm, frame)
    _draw_agents_at_frame(ax, s.ah, s.am, frame)
    _style_axes(ax, f"Frame t = {rel:+d}  ({stamp} at 10 Hz)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    _save(fig, save_path)


def plot_scene(
    batch: dict,
    save_path: Path,
    tokenizer: MotionTokenizer | None = None,
) -> None:
    """Full scene: roadgraph + all-agent past trails + modeled-agent decoded future."""
    tokenizer = tokenizer or MotionTokenizer(MotionLMConfig())
    s = _unpack(batch)

    fig, ax = plt.subplots(figsize=(11, 9))
    _draw_roadgraph(ax, s.rg, s.rm)
    _draw_tl_at_frame(ax, s.tl, s.tm, s.Tp - 1)
    _draw_agent_trails(ax, s.ah, s.am)

    fut = _decode_future(batch, tokenizer).cpu().numpy()
    ax.plot(fut[:, 0], fut[:, 1], "x--", color="#e63946", markersize=7,
            linewidth=1.5, alpha=0.95, zorder=5,
            label="modeled future (2 Hz, token-decoded)")
    ax.plot(0, 0, "*", color="#d62728", markersize=15, markeredgecolor="black",
            markeredgewidth=0.8, label="modeled @ t=0", zorder=6)
    _style_axes(
        ax, f"Scene: roadgraph + all-agent past trails (T_past={s.Tp}) + modeled 8 s future"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    _save(fig, save_path)


def plot_scene_gif(
    batch: dict,
    save_path: Path,
    tokenizer: MotionTokenizer | None = None,
    fps: int = 10,
    future_hz: int = 10,
) -> None:
    """Animate 11 past + 80 future frames @ ``future_hz``.

    Modeled agent follows the token-decoded (linearly interpolated) trajectory;
    context agents and TL states freeze at current time during the future window
    (the shard carries no future observations for them).
    """
    tokenizer = tokenizer or MotionTokenizer(MotionLMConfig())
    s = _unpack(batch)
    fut_hz, fut_head = _upsample_future(_decode_future(batch, tokenizer), future_hz)
    n_fut = fut_hz.shape[0]
    total = s.Tp + n_fut

    fig, ax = plt.subplots(figsize=(10, 8))

    def draw(k: int) -> None:
        ax.clear()
        _draw_roadgraph(ax, s.rg, s.rm)
        if k < s.Tp:
            _draw_tl_at_frame(ax, s.tl, s.tm, k)
            _draw_agents_at_frame(ax, s.ah, s.am, k)
            rel = k - (s.Tp - 1)
            stamp = "now" if rel == 0 else f"{rel * 0.1:+.1f}s"
            title = f"Frame {k:>3}/{total}   · past {stamp}"
        else:
            _draw_tl_at_frame(ax, s.tl, s.tm, s.Tp - 1)
            _draw_frozen_context(ax, s.ah, s.am, s.Tp - 1)
            t = k - s.Tp
            x, y, h = float(fut_hz[t, 0]), float(fut_hz[t, 1]), float(fut_head[t])
            ax.plot([x], [y], "o", color="#1f77b4", markersize=12,
                    markeredgecolor="red", markeredgewidth=1.4, zorder=6)
            ax.arrow(x, y, math.cos(h) * 3.0, math.sin(h) * 3.0,
                     head_width=0.6, head_length=0.8, fc="red", ec="red",
                     length_includes_head=True, alpha=0.95, zorder=5)
            trail = fut_hz[:t + 1].cpu().numpy()
            ax.plot(trail[:, 0], trail[:, 1], "-",
                    color="#e63946", linewidth=1.2, alpha=0.8)
            title = f"Frame {k:>3}/{total}   · future +{(t + 1) / future_hz:.1f}s"
        _style_axes(ax, title)

    anim = FuncAnimation(fig, draw, frames=total, interval=int(1000 / fps))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def _save(fig, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------- raw tfrecord path (no ROI / no slot limits) ----------

def _find_scenario(tfrecord_path: Path, scenario_id: bytes | str):
    """Return the first scenario in ``tfrecord_path`` matching ``scenario_id[:16]``."""
    from data.convert import parse_scenario, read_scenarios
    target = scenario_id.encode() if isinstance(scenario_id, str) else scenario_id
    for blob in read_scenarios(tfrecord_path):
        sc = parse_scenario(blob)
        if sc["scenario_id"][:16] == target[:16]:
            return sc
    raise RuntimeError(f"scenario_id {target!r} not found in {tfrecord_path}")


def _resolve_track_idx(tfrecord_path: Path, scenario_id: bytes,
                       stored_track_id: int) -> int:
    """Convert the shard's stored ``track_id`` into the positional index in ``scenario.tracks``."""
    sc = _find_scenario(tfrecord_path, scenario_id)
    for i, (tid, _, _) in enumerate(sc["tracks"]):
        if int(tid) == int(stored_track_id):
            return i
    if 0 <= int(stored_track_id) < len(sc["tracks"]):           # fallback: already an index
        return int(stored_track_id)
    raise RuntimeError(f"track_id {stored_track_id} not found in scenario")


def _raw_batch_from_tfrecord(
    tfrecord_path: Path,
    scenario_id: bytes,
    track_idx: int,
    tokenizer: MotionTokenizer,
) -> dict:
    """Build a shard-compatible batch dict from the raw tfrecord — no ROI/slot limits."""
    from data.convert import (
        OBJ_CYCLIST, OBJ_PEDESTRIAN,
        _wrap_angle, _world_to_agent,
    )

    sc = _find_scenario(tfrecord_path, scenario_id)
    tracks, cti = sc["tracks"], sc["cti"]
    _, _, states = tracks[track_idx]
    st0 = states[cti]
    x0, y0, h0 = st0["x"], st0["y"], st0["h"]
    cos_h, sin_h = math.cos(h0), math.sin(h0)
    Tp = 11

    # --- agents [A, Tp, 13] (modeled at slot 0) ---
    A = len(tracks)
    ah = torch.zeros(A, Tp, 13)
    am = torch.zeros(A, Tp, dtype=torch.bool)
    reorder = [track_idx] + [i for i in range(A) if i != track_idx]
    for slot, j in enumerate(reorder):
        _, ot_j, sts_j = tracks[j]
        type_code = 1 if ot_j == OBJ_PEDESTRIAN else 2 if ot_j == OBJ_CYCLIST else 0
        for t in range(Tp):
            idx = cti - (Tp - 1) + t
            if idx < 0 or idx >= len(sts_j) or not sts_j[idx]["valid"]:
                continue
            st = sts_j[idx]
            xp, yp = _world_to_agent(st["x"], st["y"], x0, y0, cos_h, sin_h)
            h_rel = _wrap_angle(st["h"] - h0)
            vxp = st["vx"] * cos_h + st["vy"] * sin_h
            vyp = -st["vx"] * sin_h + st["vy"] * cos_h
            ah[slot, t, :10] = torch.tensor(
                [xp, yp, st["z"], math.sin(h_rel), math.cos(h_rel),
                 vxp, vyp, st["L"], st["W"], st["H"]]
            )
            ah[slot, t, 10 + type_code] = 1.0
            am[slot, t] = True

    # --- roadgraph [R, P, 25] ---
    feats = sc["map_features"]
    R = len(feats) or 1
    P = max((len(pts) for _, _, pts in feats), default=1)
    rg = torch.zeros(R, P, 25)
    rm = torch.zeros(R, P, dtype=torch.bool)
    for fi, (_, type_idx, pts) in enumerate(feats):
        prev = None
        for pi, (x, y, z) in enumerate(pts):
            xp, yp = _world_to_agent(x, y, x0, y0, cos_h, sin_h)
            if prev is None:
                dxp, dyp = 1.0, 0.0
            else:
                dxp, dyp = xp - prev[0], yp - prev[1]
                n = math.hypot(dxp, dyp) + 1e-9
                dxp /= n; dyp /= n
            rg[fi, pi, :5] = torch.tensor([xp, yp, z, dxp, dyp])
            rg[fi, pi, 5 + type_idx] = 1.0
            rm[fi, pi] = True
            prev = (xp, yp)

    # --- traffic lights [Tp, L, 12] ---
    dms = sc["dynamic_map_states"]
    past_dms = [dms[i] if 0 <= i < len(dms) else []
                for i in range(cti - (Tp - 1), cti + 1)]
    L = max((len(lst) for lst in past_dms), default=1)
    tl = torch.zeros(Tp, L, 12)
    tm = torch.zeros(Tp, L, dtype=torch.bool)
    for t, lst in enumerate(past_dms):
        for k, (state, sp_xyz) in enumerate(lst):
            if sp_xyz is None:
                continue
            xp, yp = _world_to_agent(sp_xyz[0], sp_xyz[1], x0, y0, cos_h, sin_h)
            tl[t, k, :3] = torch.tensor([xp, yp, sp_xyz[2]])
            tl[t, k, 3 + int(state)] = 1.0
            tm[t, k] = True

    # --- tokenized future (same recipe as the shard) ---
    fut = []
    for k in range(1, 17):
        idx = cti + 5 * k
        if idx >= len(states) or not states[idx]["valid"]:
            fut.append((0.0, 0.0))
        else:
            fut.append(_world_to_agent(states[idx]["x"], states[idx]["y"],
                                       x0, y0, cos_h, sin_h))
    fut_t = torch.tensor(fut, dtype=torch.float32)
    vxp = st0["vx"] * cos_h + st0["vy"] * sin_h
    vyp = -st0["vx"] * sin_h + st0["vy"] * cos_h
    init_bin = tokenizer.velocity_to_init_bin(vxp, vyp, dt=0.5)
    tokens = tokenizer.encode(fut_t.unsqueeze(0),
                              init_bin=init_bin.unsqueeze(0)).squeeze(0)

    def _wrap(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0).unsqueeze(0)

    return {
        "agent_history":  _wrap(ah),  "agent_mask":     _wrap(am),
        "roadgraph":      _wrap(rg),  "roadgraph_mask": _wrap(rm),
        "traffic_lights": _wrap(tl),  "tl_mask":        _wrap(tm),
        "gt_tokens":      _wrap(tokens.long()),
        "init_bin":       _wrap(init_bin),
        "future_raw":     fut_t,
    }


# ---------- CLI ----------

def _render_all(
    batch: dict, prefix: str, out_dir: Path, tok: MotionTokenizer,
    frame: int, gif: bool, fps: int,
) -> None:
    frame_path = out_dir / f"scenario_frame_{prefix}.png"
    scene_path = out_dir / f"scenario_whole_{prefix}.png"
    plot_frame(batch, frame_path, frame=frame)
    plot_scene(batch, scene_path, tokenizer=tok)
    print(f"wrote {frame_path}")
    print(f"wrote {scene_path}")
    if gif:
        gif_path = out_dir / f"scenario_whole_{prefix}.gif"
        plot_scene_gif(batch, gif_path, tokenizer=tok, fps=fps)
        print(f"wrote {gif_path}")


def _first(x):
    return x[0] if isinstance(x, list) else x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=Path,
                    default=Path("/home/wentao/Downloads/shard_00000.pt.zst"))
    ap.add_argument("--tfrecord", type=Path, default=None,
                    help="Also render raw tfrecord plots for the same scenario+track.")
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument("--frame", type=int, default=-1,
                    help="Frame for plot_frame (-1 = current time).")
    ap.add_argument("--gif", action="store_true",
                    help="Also write an animated scenario_whole_*.gif.")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=Path("img"))
    args = ap.parse_args()

    cfg = MotionLMConfig()
    tok = MotionTokenizer(cfg)
    loader = make_loader([args.shard], cfg=cfg, batch_size=1, num_workers=0,
                         shuffle_buffer=1, pin_memory=False)
    it = iter(loader)
    for _ in range(args.sample_idx + 1):
        batch = next(it)

    sid, tid = _first(batch["scenario_id"]), _first(batch["track_id"])
    sid_hex = sid[:8].hex() if isinstance(sid, (bytes, bytearray)) else str(sid)
    print(f"scenario_id = {sid_hex}   track_id = {tid}   "
          f"init_bin = {batch['init_bin'][0, 0].tolist()}")

    _render_all(batch, "shard", args.out_dir, tok, args.frame, args.gif, args.fps)
    if args.tfrecord is not None:
        raw = _raw_batch_from_tfrecord(
            args.tfrecord, sid,
            track_idx=_resolve_track_idx(args.tfrecord, sid, tid),
            tokenizer=tok,
        )
        _render_all(raw, "tfrecord", args.out_dir, tok, args.frame, args.gif, args.fps)


if __name__ == "__main__":
    main()
