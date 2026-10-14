#!/usr/bin/env python3
"""Visualize the tracked trajectories: real video (GT) next to an I2V candidate.

Three modes, all sharing the repo's trail look (magma colormap, oldest dark /
newest bright, as in ``examples/01_quickstart.py``):

  video  side-by-side MP4 -- the GT video with its 2D tracks burned in next to the
         candidate video with its own tracks. Shows what the tracker actually saw,
         frame by frame, and where the candidate's motion departs after the switch.
  trail  the repo's built-in ``render_trajectory_mp4`` (examples/01_quickstart.py),
         run on the lifted 3D tracks from the switch frame onward: a growing
         projected trail over the static switch frame, GT | candidate.
  3d     GT vs similarity-aligned candidate in 3D over the prediction window --
         the quantity ``scripts/similarity_3d.py`` actually scores.
  grid   one mosaic per scene: GT plus every candidate playing in sync, each with its
         own tracks, ordered best-to-worst by the metric. Ignores --candidates.

Usage:
    python scripts/visualize_tracks.py --scene 0028 --candidates best,worst
    python scripts/visualize_tracks.py --scene 0001 --candidates c0,c7 --mode all
    python scripts/visualize_tracks.py --scene 0004 --mode 3d --candidates worst

Outputs land in outputs/viz/<scene>/<mode>_<candidate>.mp4
"""
from __future__ import annotations

import argparse, importlib.util, json, re, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "outputs" / "runs"
SIM = REPO / "outputs" / "similarity_3d"
SWITCH = 72                     # conditioning is frames 0..71; generation starts at 72

MAGMA_FLOOR, MAGMA_TOP = 0.15, 0.88


# ── loading ──────────────────────────────────────────────────────────────────

def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_scene(scene: str) -> str:
    """Accept a full video id or any unique prefix/substring (e.g. '0028')."""
    gt_dir = RUNS / "gt" / "tracks_3d" / "physics_iq"
    ids = sorted(p.name.replace("_merged_3d_tracks.npz", "") for p in gt_dir.glob("*.npz"))
    if scene in ids:
        return scene
    hits = [v for v in ids if v.startswith(scene)] or [v for v in ids if scene in v]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(f"no scene matching {scene!r} in {gt_dir}")
    raise SystemExit(f"{scene!r} is ambiguous:\n  " + "\n  ".join(hits[:10]))


def run_dir(cand: str) -> Path:
    return RUNS / ("gt" if cand == "gt" else f"magi_{cand}")


def load_tracks_2d(cand, vid):
    z = np.load(run_dir(cand) / "tracks_2d" / vid / f"{vid}_merged.npz")
    return z["tracks"].astype(np.float32), z["visibility"].astype(bool)   # (T,N,2), (T,N)


def load_tracks_3d(cand, vid):
    z = np.load(run_dir(cand) / "tracks_3d" / "physics_iq" / f"{vid}_merged_3d_tracks.npz")
    V = z["visibility"]
    return z["points_3d"], (V[..., 0] if V.ndim == 3 else V).astype(bool)  # (N,T,3), (N,T)


def load_camera(cand, vid, frame):
    """(K, c2w) at `frame` from the ViPE artifacts of that run."""
    intr = np.load(run_dir(cand) / "vipe_results" / "intrinsics" / f"{vid}.npz")["data"]
    pose = np.load(run_dir(cand) / "vipe_results" / "pose" / f"{vid}.npz")["data"]
    fx, fy, cx, cy = intr[min(frame, len(intr) - 1)]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K, pose[min(frame, len(pose) - 1)].astype(np.float64)


def read_video(cand, vid, n_max=None):
    import imageio.v2 as imageio
    rd = imageio.get_reader(str(run_dir(cand) / "videos_480p" / f"{vid}.mp4"))
    frames = []
    for i, f in enumerate(rd):
        if n_max is not None and i >= n_max:
            break
        frames.append(np.asarray(f))
    rd.close()
    return frames


def scene_metrics(vid):
    p = SIM / f"{vid}.json"
    return json.loads(p.read_text())["candidates"] if p.exists() else {}


def pick_candidates(vid, spec, metric="ade"):
    """Expand 'best'/'worst'/'all'/'c3' into concrete candidate keys."""
    m = scene_metrics(vid)
    order = sorted(m, key=lambda k: m[k].get(metric, float("inf")))
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok == "best" and order:
            out.append(order[0])
        elif tok == "worst" and order:
            out.append(order[-1])
        elif tok == "all":
            out += order
        elif re.fullmatch(r"c\d+", tok):
            out.append(tok)
        elif tok:
            raise SystemExit(f"unrecognised candidate token {tok!r}")
    seen, uniq = set(), []
    for c in out:
        if c not in seen and (run_dir(c) / "tracks_2d" / vid).exists():
            seen.add(c); uniq.append(c)
    return uniq


# ── shared helpers ───────────────────────────────────────────────────────────

def magma_bgr(u):
    """Magma colour for u in [0,1], remapped into the repo's [floor, top] sub-range."""
    import matplotlib
    c = matplotlib.colormaps["magma"](MAGMA_FLOOR + (MAGMA_TOP - MAGMA_FLOOR) * float(u))
    return tuple(int(255 * v) for v in c[2::-1])          # RGB -> BGR


def subsample(n, k):
    return np.arange(n) if k <= 0 or n <= k else np.linspace(0, n - 1, k).round().astype(int)


def world_to_cam(P, c2w):
    """(N,T,3) world -> camera frame of the pose `c2w`."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    return np.einsum("ij,ntj->nti", R.T, P - t)


def ffill_nan(P):
    """Forward-fill NaN positions along time so a lost point freezes instead of vanishing."""
    P = P.copy()
    bad = ~np.isfinite(P).all(-1)
    for n in range(P.shape[0]):
        last = None
        for t in range(P.shape[1]):
            if bad[n, t]:
                if last is not None:
                    P[n, t] = last
            else:
                last = P[n, t]
    return P


# ── mode: video ──────────────────────────────────────────────────────────────

def render_video(vid, cand, out_path, *, trail=16, max_points=120, fps=24, dot=2):
    """GT video + its tracks | candidate video + its tracks, one panel each."""
    import cv2, imageio.v2 as imageio

    T2g, Vg = load_tracks_2d("gt", vid)
    T2c, Vc = load_tracks_2d(cand, vid)
    fg, fc = read_video("gt", vid), read_video(cand, vid)
    T = min(len(fg), len(fc), len(T2g), len(T2c))
    idx = subsample(min(T2g.shape[1], T2c.shape[1]), max_points)
    met = scene_metrics(vid).get(cand, {})

    colours = [magma_bgr(1.0 - a / max(1, trail)) for a in range(trail + 1)]
    H, W = fg[0].shape[:2]
    bar = 34

    def panel(img, tr, vis, t, title, hot):
        c = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
        for a in range(min(trail, t)):                    # oldest segment first
            t1, t0 = t - a, t - a - 1
            ok = vis[t0, idx] & vis[t1, idx]
            if not ok.any():
                continue
            segs = np.stack([tr[t0, idx][ok], tr[t1, idx][ok]], axis=1).round().astype(np.int32)
            cv2.polylines(c, list(segs), False, colours[a + 1], 1, cv2.LINE_AA)
        ok = vis[t, idx]
        for (x, y) in tr[t, idx][ok].round().astype(int):
            cv2.circle(c, (x, y), dot, colours[0], -1, cv2.LINE_AA)
        c = cv2.copyMakeBorder(c, bar, 0, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18))
        cv2.putText(c, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (60, 90, 255) if hot else (235, 235, 235), 1, cv2.LINE_AA)
        return c

    ade = met.get("ade")
    right_title = f"{cand}" + (f"   ADE {ade:.3f} m" if ade is not None else "")
    frames = []
    for t in range(T):
        gen = t >= SWITCH
        tag = "GENERATED" if gen else "conditioning (identical footage)"
        left = panel(fg[t], T2g, Vg, t, f"GT (real video)   frame {t:3d}/{T - 1}", False)
        right = panel(fc[t], T2c, Vc, t, f"{right_title}   {tag}", gen)
        if gen:                                            # red rule at the switch
            cv2.line(right, (0, bar), (right.shape[1], bar), (60, 90, 255), 2)
        frames.append(cv2.cvtColor(np.hstack([left, right]), cv2.COLOR_BGR2RGB))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=fps, codec="libx264",
                    macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    return out_path



# ── mode: grid ───────────────────────────────────────────────────────────────

def render_grid(vid, out_path, *, trail=12, max_points=90, fps=24, cols=4, cell_w=427,
                dot=1, metric="ade"):
    """One mosaic per scene: GT plus every candidate, in sync, each with its own tracks.

    Candidates are ordered by `metric` (best first) so the grid reads as a ranking:
    if the metric is doing its job, the physics should visibly degrade left-to-right.
    Frames are streamed one at a time from all N videos at once -- holding 16 decoded
    480p clips in memory at once would cost several GB.
    """
    import cv2, imageio.v2 as imageio

    met = scene_metrics(vid)
    cands = sorted((k for k in met if (run_dir(k) / "videos_480p" / f"{vid}.mp4").exists()),
                   key=lambda k: met[k].get(metric, float("inf")))
    cells = ["gt"] + cands

    tracks = {c: load_tracks_2d(c, vid) for c in cells}
    readers = {c: imageio.get_reader(str(run_dir(c) / "videos_480p" / f"{vid}.mp4"))
               for c in cells}
    iters = {c: iter(readers[c]) for c in cells}
    T = min(min(len(v[0]) for v in tracks.values()), 10_000)
    idx = subsample(min(v[0].shape[1] for v in tracks.values()), max_points)

    colours = [magma_bgr(1.0 - a / max(1, trail)) for a in range(trail + 1)]
    rows = int(np.ceil(len(cells) / cols))
    scale = None
    bar = 20

    frames = []
    for t in range(T):
        tiles = []
        for c in cells:
            try:
                img = np.asarray(next(iters[c]))
            except StopIteration:
                img = np.zeros((480, 854, 3), np.uint8)
            if scale is None:
                scale = cell_w / img.shape[1]
            cell_h = int(round(img.shape[0] * scale)) // 2 * 2
            im = cv2.cvtColor(cv2.resize(img, (cell_w, cell_h)), cv2.COLOR_RGB2BGR)
            tr, vis = tracks[c]
            tr = tr * scale
            for a in range(min(trail, t)):
                t1, t0 = t - a, t - a - 1
                ok = vis[t0, idx] & vis[t1, idx]
                if ok.any():
                    segs = np.stack([tr[t0, idx][ok], tr[t1, idx][ok]],
                                    axis=1).round().astype(np.int32)
                    cv2.polylines(im, list(segs), False, colours[a + 1], 1, cv2.LINE_AA)
            for (x, y) in tr[t, idx][vis[t, idx]].round().astype(int):
                cv2.circle(im, (x, y), dot, colours[0], -1, cv2.LINE_AA)

            gen = t >= SWITCH and c != "gt"
            if gen:                                    # candidate footage is generated now
                cv2.rectangle(im, (0, 0), (cell_w - 1, cell_h - 1), (60, 90, 255), 1)
            im = cv2.copyMakeBorder(im, bar, 2, 2, 2, cv2.BORDER_CONSTANT, value=(18, 18, 18))
            if c == "gt":
                label, col = "GT (real video)", (120, 235, 120)
            else:
                a_ = met[c].get(metric)
                label = f"{c}   {metric.upper()} {a_:.3f} m" if a_ is not None else c
                col = (60, 90, 255) if gen else (235, 235, 235)
            cv2.putText(im, label, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
            tiles.append(im)

        blank = np.full_like(tiles[0], 18)
        tiles += [blank] * (rows * cols - len(tiles))
        mosaic = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        head = np.full((28, mosaic.shape[1], 3), 18, np.uint8)
        state = "GENERATED" if t >= SWITCH else "conditioning (identical footage)"
        cv2.putText(head, f"{vid[:4]} {vid.split('trimmed-')[-1]}   frame {t:3d}/{T - 1}   {state}"
                    f"   -- candidates ordered by {metric.upper()}, best first",
                    (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (60, 90, 255) if t >= SWITCH else (235, 235, 235), 1, cv2.LINE_AA)
        out = np.vstack([head, mosaic])
        out = out[: out.shape[0] // 2 * 2, : out.shape[1] // 2 * 2]
        frames.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    for r in readers.values():
        r.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=fps, codec="libx264",
                    macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    return out_path


# ── mode: trail (built-in renderer) ──────────────────────────────────────────

def _anchor_frame(vid, cand, min_points=5):
    """First frame >= SWITCH where both GT and candidate show `min_points` tracked points."""
    Pg, Vg = load_tracks_3d("gt", vid)
    Pc, Vc = load_tracks_3d(cand, vid)
    _, vg2 = load_tracks_2d("gt", vid)
    _, vc2 = load_tracks_2d(cand, vid)
    T = min(Pg.shape[1], Pc.shape[1], len(vg2), len(vc2))
    for t in range(SWITCH, T):
        g = Vg[:, t] & np.isfinite(Pg[:, t]).all(-1) & vg2[t][: Pg.shape[0]]
        c = Vc[:, t] & np.isfinite(Pc[:, t]).all(-1) & vc2[t][: Pc.shape[0]]
        if g.sum() >= min_points and c.sum() >= min_points:
            return t
    return None


def render_trail(vid, cand, out_path, *, max_points=80, seconds=3.0, fps=24):
    """Repo's `render_trajectory_mp4` on the lifted 3D tracks, GT | candidate."""
    import subprocess, tempfile
    from PIL import Image
    qs = _load_module(REPO / "examples" / "01_quickstart.py", "mm_quickstart")

    # The trail is anchored on one frame, so that frame must actually show the points.
    # Where the object is occluded at the switch (0004: the ball sits behind the paper),
    # anchoring on SWITCH would draw nothing at all -- so both panels share the first
    # frame at or after the switch where the GT and the candidate each have enough
    # visible points. Sharing it keeps the two panels comparable.
    anchor = _anchor_frame(vid, cand)
    if anchor is None:
        print(f"  [trail] {cand}: no frame after the switch has >= 5 visible tracked "
              f"points in both streams -- skipping (use --mode video instead)")
        return None
    if anchor != SWITCH:
        print(f"  [trail] {cand}: anchored on frame {anchor} "
              f"(nothing visible at the switch frame {SWITCH})")

    parts = []
    tmp = Path(tempfile.mkdtemp())
    for who in ("gt", cand):
        P, V = load_tracks_3d(who, vid)
        K, c2w = load_camera(who, vid, anchor)
        tr2d, vis2d = load_tracks_2d(who, vid)
        keep = V[:, anchor] & np.isfinite(P[:, anchor]).all(-1) & vis2d[anchor]
        sel = np.where(keep)[0]
        sel = sel[subsample(len(sel), max_points)]
        cam = ffill_nan(world_to_cam(P[sel, anchor:], c2w))
        frame = read_video(who, vid, n_max=anchor + 1)[anchor]
        p = tmp / f"{who}.mp4"
        qs.render_trajectory_mp4(cam, t0_image=Image.fromarray(frame), intrinsics=K,
                                 points_2d_at_t0=tr2d[SWITCH, sel], output_path=str(p),
                                 fps=fps, seconds=seconds)
        parts.append(p)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(parts[0]), "-i", str(parts[1]),
                    "-filter_complex",
                    "[0:v]drawtext=text='GT (real video)':x=10:y=10:fontcolor=white:fontsize=20[a];"
                    f"[1:v]drawtext=text='{cand}':x=10:y=10:fontcolor=white:fontsize=20[b];"
                    "[a][b]hstack=inputs=2",
                    "-pix_fmt", "yuv420p", str(out_path)], check=True)
    return out_path


# ── mode: 3d ─────────────────────────────────────────────────────────────────

def render_3d(vid, cand, out_path, *, max_points=60, fps=24):
    """GT vs similarity-aligned candidate in 3D, using similarity_3d's own alignment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    sim = _load_module(REPO / "scripts" / "similarity_3d.py", "mm_similarity_3d")

    Pg, Vg = load_tracks_3d("gt", vid)
    Pc, Vc = load_tracks_3d(cand, vid)
    N, T = min(Pg.shape[0], Pc.shape[0]), min(Pg.shape[1], Pc.shape[1])
    Pg, Vg, Pc = Pg[:N, :T], Vg[:N, :T], Pc[:N, :T]

    vis = (Vg & Vg[:, [0]]) & np.isfinite(Pg).all(-1) & np.isfinite(Pc).all(-1)
    cond = slice(0, SWITCH)
    X, Y = Pc[:, cond][vis[:, cond]], Pg[:, cond][vis[:, cond]]
    if len(X) >= sim.MIN_COND_PTS:
        s, R, t = sim.umeyama(X, Y)
    else:
        s, R, t = 1.0, np.eye(3), np.zeros(3)
    Pc_a = s * np.einsum("ij,ntj->nti", R, Pc) + t          # candidate in GT's frame

    sel = np.where(vis[:, SWITCH:].mean(1) > 0.5)[0]
    if len(sel) < 5:
        sel = np.where(Vg[:, 0])[0]
    sel = sel[subsample(len(sel), max_points)]
    G, C = ffill_nan(Pg[sel]), ffill_nan(Pc_a[sel])

    pts = np.concatenate([G.reshape(-1, 3), C.reshape(-1, 3)])
    pts = pts[np.isfinite(pts).all(-1)]
    lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    ctr, rad = (lo + hi) / 2, float((hi - lo).max()) * 0.58 + 1e-3
    met = scene_metrics(vid).get(cand, {})

    frames = []
    for k in range(T):
        fig = plt.figure(figsize=(7.2, 5.4), dpi=100, facecolor="#0d0d0d")
        ax = fig.add_subplot(111, projection="3d", facecolor="#0d0d0d")
        for arr, col, lab in ((G, "#4da3ff", "GT"), (C, "#ff8c42", f"{cand} (aligned)")):
            for i in range(len(arr)):
                ax.plot(*arr[i, : k + 1].T, color=col, lw=0.9, alpha=0.55)
            ax.scatter(*arr[:, k].T, color=col, s=9, depthshade=False, label=lab)
        ax.set_xlim(ctr[0] - rad, ctr[0] + rad); ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
        ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
        ax.set_box_aspect((1, 1, 1), zoom=1.45)          # fill the canvas
        ax.view_init(elev=-72, azim=-90 + 25 * np.sin(2 * np.pi * k / max(1, T - 1)))
        ax.set_axis_off()
        state = "GENERATED" if k >= SWITCH else "conditioning"
        ade = met.get("ade")
        title = f"{vid[:4]}  frame {k}/{T-1}  [{state}]"
        if ade is not None:
            title += f"   ADE {ade:.3f} m   scale {s:.3f}"
        ax.set_title(title, color="#dddddd", fontsize=10)
        ax.legend(loc="upper right", facecolor="#0d0d0d", labelcolor="#dddddd",
                  edgecolor="#333333", fontsize=9)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(buf[: buf.shape[0] // 2 * 2, : buf.shape[1] // 2 * 2])
        plt.close(fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=fps, codec="libx264",
                    macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    return out_path


# ── cli ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="video id or unique prefix, e.g. 0028")
    ap.add_argument("--candidates", default="best,worst",
                    help="comma list of c<N> / best / worst / all (default: best,worst)")
    ap.add_argument("--mode", default="video",
                    choices=["video", "trail", "3d", "grid", "all"])
    ap.add_argument("--out", default=str(REPO / "outputs" / "viz"))
    ap.add_argument("--max-points", type=int, default=120)
    ap.add_argument("--trail", type=int, default=16, help="trail length in frames (video mode)")
    ap.add_argument("--fps", type=int, default=24)
    a = ap.parse_args()

    vid = resolve_scene(a.scene)
    cands = pick_candidates(vid, a.candidates)
    if not cands:
        raise SystemExit(f"no candidates available for {vid}")
    modes = ["video", "trail", "3d"] if a.mode == "all" else [a.mode]
    outdir = Path(a.out) / vid

    if "grid" in modes:                    # one mosaic for the whole scene, not per candidate
        out = outdir / "grid.mp4"
        render_grid(vid, out, trail=a.trail, max_points=a.max_points, fps=a.fps)
        print(f"  [grid ] {len(scene_metrics(vid))} candidates + GT -> {out}")
        modes = [m for m in modes if m != "grid"]
        if not modes:
            return

    for cand in cands:
        for mode in modes:
            out = outdir / f"{mode}_{cand}.mp4"
            if mode == "video":
                render_video(vid, cand, out, trail=a.trail, max_points=a.max_points, fps=a.fps)
            elif mode == "trail":
                if render_trail(vid, cand, out, max_points=min(a.max_points, 80),
                                fps=a.fps) is None:
                    continue
            else:
                render_3d(vid, cand, out, max_points=min(a.max_points, 60), fps=a.fps)
            print(f"  [{mode:5s}] {cand} -> {out}")


if __name__ == "__main__":
    main()
