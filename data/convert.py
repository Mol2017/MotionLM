"""WOMD tfrecord → sharded ``.pt.zst`` converter (TF-free).

Hand-rolled TFRecord framing reader + minimal protobuf parser for the WOMD
``Scenario`` message — no ``tensorflow`` or ``waymo_open_dataset`` dependency.

Mirrors design doc §"Data pipeline · Converter & dataloader".  One tfrecord
→ one shard; filters modeled agents to ``ALLOWED_MODELED_TYPES`` (VEHICLE +
PEDESTRIAN + CYCLIST by default) with ≥ 4 valid future steps. Sparse on-disk
representation: fp16 features, int8 type indices, bit-packed ``future_valid``,
zstd-9 compressed.

Usage::

    python -m data.convert  INPUT_TFRECORD  OUTPUT_PT_ZST
"""

from __future__ import annotations

import argparse
import io
import math
import struct
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import zstandard as zstd

from model.config import MotionLMConfig
from model.motion_tokenizer import MotionTokenizer


# ----------------------------- constants (match design doc) ----------------------------

ROI_X_MIN, ROI_X_MAX = -40.0, 120.0
ROI_Y_MIN, ROI_Y_MAX = -50.0, 50.0

A_SLOTS = 64
T_PAST = 11
T_FUTURE_2HZ = 16
R_SLOTS = 256
P_POINTS = 128
L_SLOTS = 16

# WOMD MapFeature proto tags
SUB_LANECENTER, SUB_ROADLINE, SUB_ROADEDGE = 3, 4, 5
SUB_STOPSIGN, SUB_CROSSWALK, SUB_SPEEDBUMP, SUB_DRIVEWAY = 7, 8, 9, 10
SUBTYPE_NAMES = {3: "LaneCenter", 4: "RoadLine", 5: "RoadEdge",
                 7: "StopSign", 8: "Crosswalk", 9: "SpeedBump", 10: "Driveway"}
POINT_FIELD = {3: 8, 4: 2, 5: 2, 7: 2, 8: 1, 9: 1, 10: 1}
TYPE_FIELD = {3: 2, 4: 1, 5: 1}
# Offsets into the unified 20-slot roadgraph type vocab.
TYPE_OFFSET = {3: 0, 4: 4, 5: 13, 7: 16, 8: 17, 9: 18, 10: 19}
# Stratified-truncation priority (lower = kept first) so rare subtypes survive.
SUBTYPE_PRIORITY = {7: 0, 8: 1, 9: 2, 10: 3, 5: 4, 4: 5, 3: 6}

OBJ_VEHICLE = 1
OBJ_PEDESTRIAN = 2
OBJ_CYCLIST = 3
# Modeled-agent types accepted by the converter. WOMD's ObjectType spans
# UNSET=0, VEHICLE=1, PEDESTRIAN=2, CYCLIST=3, OTHER=4. We emit the first three;
# the 3-slot ``ag_type`` one-hot already accommodates all of them.
ALLOWED_MODELED_TYPES: frozenset[int] = frozenset({OBJ_VEHICLE, OBJ_PEDESTRIAN, OBJ_CYCLIST})

# Verlet config is delegated to ``MotionTokenizer``; the converter never hard-codes
# bin_size / offset_clip so both paths stay in lockstep with ``MotionLMConfig``.


# --------------------- protobuf wire-format primitives --------------------

def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    r, sh = 0, 0
    while True:
        b = buf[pos]; pos += 1
        r |= (b & 0x7F) << sh
        if not (b & 0x80):
            return r, pos
        sh += 7


def _read_tag(buf: bytes, pos: int) -> tuple[int, int, int]:
    t, pos = _read_varint(buf, pos)
    return t >> 3, t & 0x7, pos


def _skip(buf: bytes, pos: int, w: int) -> int:
    if w == 0:
        _, pos = _read_varint(buf, pos)
    elif w == 1:
        pos += 8
    elif w == 2:
        n, pos = _read_varint(buf, pos); pos += n
    elif w == 5:
        pos += 4
    return pos


# --------------------------- WOMD proto parsers -------------------------

def _parse_map_point(buf, start, end):
    pos = start
    x = y = z = 0.0
    while pos < end:
        f, w, pos = _read_tag(buf, pos)
        if w == 1:
            (val,) = struct.unpack("<d", buf[pos:pos + 8])
            if f == 1: x = val
            elif f == 2: y = val
            elif f == 3: z = val
            pos += 8
        else:
            pos = _skip(buf, pos, w)
    return x, y, z


def _parse_subtype_points(buf, start, end, sub):
    pf = POINT_FIELD[sub]
    pts = []
    pos = start
    while pos < end:
        f, w, pos = _read_tag(buf, pos)
        if f == pf and w == 2:
            length, pos = _read_varint(buf, pos)
            pts.append(_parse_map_point(buf, pos, pos + length))
            pos += length
        else:
            pos = _skip(buf, pos, w)
    return pts


def _parse_subtype_type_value(buf, start, end, sub):
    tf = TYPE_FIELD.get(sub)
    if tf is None:
        return 0
    pos = start
    while pos < end:
        f, w, pos = _read_tag(buf, pos)
        if f == tf and w == 0:
            val, pos = _read_varint(buf, pos)
            return val
        pos = _skip(buf, pos, w)
    return 0


def _parse_map_feature(buf, start, end):
    pos = start
    sub = None
    ss = se = None
    while pos < end:
        f, w, pos = _read_tag(buf, pos)
        if f in SUBTYPE_NAMES and w == 2:
            sub = f
            bl, pos = _read_varint(buf, pos)
            ss, se = pos, pos + bl
            pos = se
        else:
            pos = _skip(buf, pos, w)
    if sub is None:
        return None
    type_val = _parse_subtype_type_value(buf, ss, se, sub)
    pts = _parse_subtype_points(buf, ss, se, sub)
    type_idx = TYPE_OFFSET[sub] + (type_val if sub in TYPE_FIELD else 0)
    return sub, type_idx, pts


def _parse_track(buf, start, end):
    pos = start
    tid = 0
    obj_type = 0
    states: list[dict[str, Any]] = []
    while pos < end:
        f, w, pos = _read_tag(buf, pos)
        if f == 1 and w == 0:
            tid, pos = _read_varint(buf, pos)
        elif f == 2 and w == 0:
            obj_type, pos = _read_varint(buf, pos)
        elif f == 3 and w == 2:
            length, pos = _read_varint(buf, pos)
            sp, se = pos, pos + length
            st = dict(x=0.0, y=0.0, z=0.0, L=0.0, W=0.0, H=0.0,
                      h=0.0, vx=0.0, vy=0.0, valid=False)
            while sp < se:
                ff, ww, sp = _read_tag(buf, sp)
                if ww == 1:
                    (val,) = struct.unpack("<d", buf[sp:sp + 8])
                    if ff == 2: st["x"] = val
                    elif ff == 3: st["y"] = val
                    elif ff == 4: st["z"] = val
                    sp += 8
                elif ww == 5:
                    (val,) = struct.unpack("<f", buf[sp:sp + 4])
                    if ff == 5: st["L"] = val
                    elif ff == 6: st["W"] = val
                    elif ff == 7: st["H"] = val
                    elif ff == 8: st["h"] = val
                    elif ff == 9: st["vx"] = val
                    elif ff == 10: st["vy"] = val
                    sp += 4
                elif ww == 0:
                    val, sp = _read_varint(buf, sp)
                    if ff == 11: st["valid"] = bool(val)
                else:
                    sp = _skip(buf, sp, ww)
            states.append(st)
            pos += length
        else:
            pos = _skip(buf, pos, w)
    return tid, obj_type, states


def _parse_required_prediction(buf, start, end):
    pos = start
    ti = None
    while pos < end:
        f, w, pos = _read_tag(buf, pos)
        if f == 1 and w == 0:
            ti, pos = _read_varint(buf, pos)
        else:
            pos = _skip(buf, pos, w)
    return ti


def _parse_tl_state(buf, start, end):
    pos = start
    state = 0
    sp_xyz = None
    while pos < end:
        f, w, pos = _read_tag(buf, pos)
        if f == 2 and w == 0:
            state, pos = _read_varint(buf, pos)
        elif f == 3 and w == 2:
            length, pos = _read_varint(buf, pos)
            sp_xyz = _parse_map_point(buf, pos, pos + length)
            pos += length
        else:
            pos = _skip(buf, pos, w)
    return state, sp_xyz


def _parse_dynamic_map_state(buf, start, end):
    tls = []
    pos = start
    while pos < end:
        f, w, pos = _read_tag(buf, pos)
        if f == 1 and w == 2:
            length, pos = _read_varint(buf, pos)
            tls.append(_parse_tl_state(buf, pos, pos + length))
            pos += length
        else:
            pos = _skip(buf, pos, w)
    return tls


def parse_scenario(data: bytes) -> dict[str, Any]:
    """Full Scenario proto → dict with the fields we need."""
    out: dict[str, Any] = dict(
        scenario_id=b"", tracks=[], map_features=[],
        dynamic_map_states=[], tracks_to_predict=[], cti=10,
    )
    pos, n = 0, len(data)
    while pos < n:
        f, w, pos = _read_tag(data, pos)
        if f == 5 and w == 2:
            length, pos = _read_varint(data, pos)
            out["scenario_id"] = bytes(data[pos:pos + length])
            pos += length
        elif f == 2 and w == 2:
            length, pos = _read_varint(data, pos)
            out["tracks"].append(_parse_track(data, pos, pos + length))
            pos += length
        elif f == 7 and w == 2:
            length, pos = _read_varint(data, pos)
            out["dynamic_map_states"].append(_parse_dynamic_map_state(data, pos, pos + length))
            pos += length
        elif f == 8 and w == 2:
            length, pos = _read_varint(data, pos)
            mf = _parse_map_feature(data, pos, pos + length)
            if mf is not None:
                out["map_features"].append(mf)
            pos += length
        elif f == 10 and w == 0:
            out["cti"], pos = _read_varint(data, pos)
        elif f == 11 and w == 2:
            length, pos = _read_varint(data, pos)
            ti = _parse_required_prediction(data, pos, pos + length)
            if ti is not None:
                out["tracks_to_predict"].append(ti)
            pos += length
        else:
            pos = _skip(data, pos, w)
    return out


def read_scenarios(tfrecord_path: Path) -> Iterable[bytes]:
    """Yield raw scenario bytes from a TFRecord file (one scenario per record)."""
    with open(tfrecord_path, "rb") as fh:
        while True:
            head = fh.read(8)
            if len(head) < 8:
                return
            (length,) = struct.unpack("<Q", head)
            fh.read(4)                # crc of length
            data = fh.read(length)
            fh.read(4)                # crc of data
            yield data


# --------------------------- Stage 0 helpers ---------------------------

def _world_to_agent(x, y, x0, y0, cos_h, sin_h):
    dx, dy = x - x0, y - y0
    return dx * cos_h + dy * sin_h, -dx * sin_h + dy * cos_h


def _wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


# ------------------------ sparse builders ------------------------

def _build_agents_sparse(tracks, modeled_idx, cti, x0, y0, h0, cos_h, sin_h):
    candidates = []
    for j, (_tid, _ot, states) in enumerate(tracks):
        if j == modeled_idx:
            continue
        st = states[cti]
        if not st["valid"]:
            continue
        dx, dy = st["x"] - x0, st["y"] - y0
        candidates.append((dx * dx + dy * dy, j))
    candidates.sort()
    selected = [modeled_idx] + [j for _, j in candidates[:A_SLOTS - 1]]

    ag_feats: list[list[float]] = []
    ag_type: list[int] = []
    ag_slot: list[int] = []
    ag_time: list[int] = []

    for slot, j in enumerate(selected):
        _tid, ot, states = tracks[j]
        if ot == OBJ_PEDESTRIAN:
            type_code = 1
        elif ot == OBJ_CYCLIST:
            type_code = 2
        else:
            type_code = 0

        for t in range(T_PAST):
            idx = cti - (T_PAST - 1) + t
            if idx < 0 or idx >= len(states):
                continue
            st = states[idx]
            if not st["valid"]:
                continue
            xp, yp = _world_to_agent(st["x"], st["y"], x0, y0, cos_h, sin_h)
            h_rel = _wrap_angle(st["h"] - h0)
            vxp = st["vx"] * cos_h + st["vy"] * sin_h
            vyp = -st["vx"] * sin_h + st["vy"] * cos_h
            ag_feats.append([xp, yp, st["z"],
                             math.sin(h_rel), math.cos(h_rel),
                             vxp, vyp,
                             st["L"], st["W"], st["H"]])
            ag_type.append(type_code)
            ag_slot.append(slot)
            ag_time.append(t)

    return dict(
        ag_feats=torch.as_tensor(np.asarray(ag_feats, dtype=np.float16)),
        ag_type=torch.as_tensor(np.asarray(ag_type, dtype=np.int8)),
        ag_slot=torch.as_tensor(np.asarray(ag_slot, dtype=np.int8)),
        ag_time=torch.as_tensor(np.asarray(ag_time, dtype=np.int8)),
    )


def _build_roadgraph_sparse(map_features, x0, y0, h0, cos_h, sin_h):
    chunks = []
    for sub, type_idx, pts in map_features:
        xy_agent = [(_world_to_agent(x, y, x0, y0, cos_h, sin_h), z) for (x, y, z) in pts]
        in_roi = [(xp, yp, z) for ((xp, yp), z) in xy_agent
                  if ROI_X_MIN <= xp <= ROI_X_MAX and ROI_Y_MIN <= yp <= ROI_Y_MAX]
        if not in_roi:
            continue
        n = len(in_roi)
        dirs = []
        for i in range(n):
            if i + 1 < n:
                dx = in_roi[i + 1][0] - in_roi[i][0]
                dy = in_roi[i + 1][1] - in_roi[i][1]
            else:
                dx = in_roi[i][0] - in_roi[i - 1][0] if i > 0 else 1.0
                dy = in_roi[i][1] - in_roi[i - 1][1] if i > 0 else 0.0
            norm = math.hypot(dx, dy) + 1e-9
            dirs.append((dx / norm, dy / norm))
        for c_start in range(0, n, P_POINTS):
            chunk_pts = []
            for i in range(c_start, min(c_start + P_POINTS, n)):
                xp, yp, zp = in_roi[i]
                dx, dy = dirs[i]
                chunk_pts.append((xp, yp, zp, dx, dy))
            chunks.append((sub, type_idx, chunk_pts))

    chunks.sort(key=lambda c: SUBTYPE_PRIORITY[c[0]])
    chunks = chunks[:R_SLOTS]

    rg_xyz, rg_dir, rg_type, rg_chunk_idx, rg_point_idx = [], [], [], [], []
    for cidx, (_sub, type_idx, pts) in enumerate(chunks):
        for pidx, (xp, yp, zp, dx, dy) in enumerate(pts):
            rg_xyz.append([xp, yp, zp])
            rg_dir.append([dx, dy])
            rg_type.append(type_idx)
            rg_chunk_idx.append(cidx)
            rg_point_idx.append(pidx)

    return dict(
        rg_xyz=torch.as_tensor(np.asarray(rg_xyz, dtype=np.float16)).reshape(-1, 3),
        rg_dir=torch.as_tensor(np.asarray(rg_dir, dtype=np.float16)).reshape(-1, 2),
        rg_type=torch.as_tensor(np.asarray(rg_type, dtype=np.int8)),
        rg_chunk_idx=torch.as_tensor(np.asarray(rg_chunk_idx, dtype=np.int16)),
        rg_point_idx=torch.as_tensor(np.asarray(rg_point_idx, dtype=np.int8)),
    )


def _build_tls_sparse(dynamic_map_states, cti, x0, y0, cos_h, sin_h):
    tl_feats: list[list[float]] = []
    tl_state: list[int] = []
    tl_slot: list[int] = []
    tl_time: list[int] = []
    for t in range(T_PAST):
        idx = cti - (T_PAST - 1) + t
        if idx < 0 or idx >= len(dynamic_map_states):
            continue
        tls = dynamic_map_states[idx]
        scored = []
        for (state, sp_xyz) in tls:
            if sp_xyz is None:
                continue
            xp, yp = _world_to_agent(sp_xyz[0], sp_xyz[1], x0, y0, cos_h, sin_h)
            d = xp * xp + yp * yp
            scored.append((d, state, xp, yp, sp_xyz[2]))
        scored.sort()
        for slot, (_, state, xp, yp, zp) in enumerate(scored[:L_SLOTS]):
            tl_feats.append([xp, yp, zp])
            tl_state.append(int(state))
            tl_slot.append(slot)
            tl_time.append(t)

    return dict(
        tl_feats=torch.as_tensor(np.asarray(tl_feats, dtype=np.float16)).reshape(-1, 3),
        tl_state=torch.as_tensor(np.asarray(tl_state, dtype=np.int8)),
        tl_slot=torch.as_tensor(np.asarray(tl_slot, dtype=np.int8)),
        tl_time=torch.as_tensor(np.asarray(tl_time, dtype=np.int8)),
    )


def _compute_init_bin(
    states: list[dict[str, Any]],
    cti: int,
    x0: float,
    y0: float,
    cos_h: float,
    sin_h: float,
    tokenizer: MotionTokenizer,
) -> torch.Tensor:
    """Return the agent-frame Δ-bin at t=−1 (estimated first future 0.5 s displacement).

    Preferred source: WOMD's ``state.velocity`` field at ``cti``, scaled by 0.5 s.
    Empirically beats a position-based backward FD by ~0.25 bins mean error because
    WOMD's velocity is a centered derivative (instantaneous at cti), whereas a 0.5 s
    backward FD lags by ~0.25 s under acceleration.

    Fallback — state.velocity is exactly zero *and* a past frame is available: use
    ``(pos[cti] − pos[cti−5])`` as a 0.5 s backward FD. This catches the rare cases
    where WOMD reports zero velocity while the position trace says otherwise.
    """
    st0 = states[cti]
    vx, vy = st0["vx"], st0["vy"]
    past_idx = cti - 5
    if vx == 0.0 and vy == 0.0 and past_idx >= 0 and states[past_idx]["valid"]:
        st_past = states[past_idx]
        dx_w = x0 - st_past["x"]
        dy_w = y0 - st_past["y"]
        dx = dx_w * cos_h + dy_w * sin_h
        dy = -dx_w * sin_h + dy_w * cos_h
        return tokenizer.delta_to_init_bin(dx, dy)

    vxp = vx * cos_h + vy * sin_h
    vyp = -vx * sin_h + vy * cos_h
    return tokenizer.velocity_to_init_bin(vxp, vyp, dt=0.5)


def _build_targets(track, cti, x0, y0, cos_h, sin_h, tokenizer: MotionTokenizer):
    """Extract future @ 2 Hz, Verlet-tokenize via the shared ``MotionTokenizer``.

    All discretization (bin size, offset clip, velocity-to-bin) is delegated to
    ``tokenizer`` so the converter and the model cannot drift apart.
    """
    _tid, _ot, states = track
    future_pos: list[tuple[float, float]] = []
    valid_mask: list[bool] = []
    for k in range(1, T_FUTURE_2HZ + 1):
        idx = cti + 5 * k
        if idx >= len(states) or not states[idx]["valid"]:
            future_pos.append((0.0, 0.0))
            valid_mask.append(False)
            continue
        st = states[idx]
        xp, yp = _world_to_agent(st["x"], st["y"], x0, y0, cos_h, sin_h)
        future_pos.append((xp, yp))
        valid_mask.append(True)

    init_bin = _compute_init_bin(states, cti, x0, y0, cos_h, sin_h, tokenizer)   # [2] long

    fut_t = torch.as_tensor(future_pos, dtype=torch.float32)                     # [T, 2]
    tokens = tokenizer.encode(
        fut_t.unsqueeze(0), init_bin=init_bin.unsqueeze(0),
    ).squeeze(0)                                                                 # [T] long

    mask_bits = 0
    for i, v in enumerate(valid_mask):
        if v:
            mask_bits |= 1 << i
    future_valid = torch.tensor([mask_bits & 0xFF, (mask_bits >> 8) & 0xFF], dtype=torch.uint8)
    return (
        dict(
            gt_tokens=tokens.to(torch.int16),
            future_valid=future_valid,
            init_bin=init_bin.to(torch.int16),
        ),
        sum(valid_mask),
    )


# --------------------------- disk IO ---------------------------

def save_shard_zstd(shard: dict[str, Any], path: Path, level: int = 9) -> None:
    buf = io.BytesIO()
    torch.save(shard, buf)
    cctx = zstd.ZstdCompressor(level=level)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cctx.compress(buf.getvalue()))


def load_shard_zstd(path: Path) -> dict[str, Any]:
    data = Path(path).read_bytes()
    dctx = zstd.ZstdDecompressor()
    buf = io.BytesIO(dctx.decompress(data))
    return torch.load(buf, weights_only=False)


# --------------------------- main converter ---------------------------

def convert_tfrecord(
    tfrecord_path: Path,
    out_path: Path,
    cfg: MotionLMConfig | None = None,
    zstd_level: int = 9,
    verbose: bool = True,
    min_future_valid: int = 4,
) -> int:
    """Convert one tfrecord → sharded ``.pt.zst``. Returns n_examples written.

    A single :class:`MotionTokenizer` instance (built from ``cfg``) handles all
    Verlet encoding — this is the **same** tokenizer the model uses at train/infer
    time, so the shard's tokens and the model's decode semantics can never drift.

    Set ``min_future_valid=0`` to keep tracks with no GT future — needed for the
    WOMD test split, whose scenarios withhold future states.
    """
    if cfg is None:
        cfg = MotionLMConfig()
    assert cfg.A == A_SLOTS and cfg.R == R_SLOTS and cfg.P == P_POINTS and cfg.L == L_SLOTS
    tokenizer = MotionTokenizer(cfg)

    t_start = time.time()
    examples: list[dict[str, Any]] = []
    n_scenarios = n_candidates = n_after_type = n_after_future = 0
    per_type_kept = {OBJ_VEHICLE: 0, OBJ_PEDESTRIAN: 0, OBJ_CYCLIST: 0}

    for blob in read_scenarios(tfrecord_path):
        sc = parse_scenario(blob)
        n_scenarios += 1
        tracks = sc["tracks"]
        cti = sc["cti"]
        for tidx in sc["tracks_to_predict"]:
            n_candidates += 1
            if tidx >= len(tracks):
                continue
            track = tracks[tidx]
            _tid, ot, states = track
            if ot not in ALLOWED_MODELED_TYPES:
                continue
            n_after_type += 1
            st0 = states[cti]
            if not st0["valid"]:
                continue

            x0, y0, h0 = st0["x"], st0["y"], st0["h"]
            cos_h, sin_h = math.cos(h0), math.sin(h0)

            targets, valid_count = _build_targets(track, cti, x0, y0, cos_h, sin_h, tokenizer)
            if valid_count < min_future_valid:
                continue
            n_after_future += 1
            per_type_kept[ot] = per_type_kept.get(ot, 0) + 1

            ex: dict[str, Any] = {}
            ex.update(_build_agents_sparse(tracks, tidx, cti, x0, y0, h0, cos_h, sin_h))
            ex.update(_build_roadgraph_sparse(sc["map_features"], x0, y0, h0, cos_h, sin_h))
            ex.update(_build_tls_sparse(sc["dynamic_map_states"], cti, x0, y0, cos_h, sin_h))
            ex.update(targets)
            ex["x0"] = float(x0)
            ex["y0"] = float(y0)
            ex["h0"] = float(h0)
            ex["scenario_id"] = sc["scenario_id"][:16].ljust(16, b"\x00")
            ex["track_id"] = int(track[0])
            examples.append(ex)

    shard = {"version": 1, "count": len(examples), "examples": examples}
    save_shard_zstd(shard, out_path, level=zstd_level)

    if verbose:
        wall = time.time() - t_start
        sz = out_path.stat().st_size
        print(f"  scenarios read       : {n_scenarios}")
        print(f"  tracks_to_predict    : {n_candidates}")
        print(f"  after type filter    : {n_after_type}  (VEHICLE / PEDESTRIAN / CYCLIST)")
        print(f"  after future≥4       : {n_after_future}")
        print(f"  examples in shard    : {len(examples)}  "
              f"(V={per_type_kept.get(OBJ_VEHICLE, 0)}  "
              f"P={per_type_kept.get(OBJ_PEDESTRIAN, 0)}  "
              f"C={per_type_kept.get(OBJ_CYCLIST, 0)})")
        print(f"  compressed shard     : {sz / 1024 / 1024:.2f} MB"
              f"  ({sz / max(len(examples), 1):.0f} B/ex)")
        print(f"  wall time            : {wall:.1f} s")
    return len(examples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_tfrecord", type=Path)
    parser.add_argument("output_pt_zst", type=Path)
    parser.add_argument("--zstd-level", type=int, default=9)
    parser.add_argument("--min-future-valid", type=int, default=4,
                        help="Drop tracks with fewer than N valid future steps. "
                             "Set to 0 for the WOMD test split (no GT future).")
    args = parser.parse_args()

    args.output_pt_zst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Converting: {args.input_tfrecord}")
    print(f"      → {args.output_pt_zst} (zstd level {args.zstd_level})")
    convert_tfrecord(args.input_tfrecord, args.output_pt_zst,
                     zstd_level=args.zstd_level,
                     min_future_valid=args.min_future_valid)


if __name__ == "__main__":
    main()
