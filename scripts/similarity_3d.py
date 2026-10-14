#!/usr/bin/env python3
"""Per-scene 3D trajectory similarity between the real video and each I2V candidate.

Writes one JSON per scene to outputs/similarity_3d/<scene>.json and merges new
candidates into an existing file, so this can be re-run as more candidates finish.

Why the alignment step exists: ViPE estimates metric scale independently per video,
and measured across scenes the candidate/GT scale ratio spans 0.86-1.11 (median 0.96).
Comparing raw coordinates would fold that ~10% calibration error into every number.

The alignment is fit on the conditioning segment (frames 0..SWITCH-1). That footage is
byte-for-byte the same real video in both streams -- verified: mean |GT - candidate|
pixel difference is ~2 through frame 71 and jumps at frame 72 -- so the true 3D
structure there is identical, and any transform between the two reconstructions is
pure reconstruction bias. Fitting there keeps the alignment independent of the
post-switch motion being scored, which is what makes the residual meaningful; fitting
on the prediction window would partly absorb the very error we want to measure.

Query points are shared by construction (candidates reuse GT's Stage-1 grounding), so
track n is the same physical point in both -- no correspondence estimation needed.
"""
import argparse, json, os, re
from pathlib import Path

import numpy as np

REPO = Path("/mmfs1/gscratch/krishna/renjiz2/2026/physics_realism/molmomotion")
SWITCH = 72          # conditioning is frames 0..71; generation starts at 72
MIN_COND_PTS = 12    # need enough commonly-visible points to fit a similarity transform


def load_tracks(path):
    z = np.load(path)
    P = z["points_3d"]                       # (N, T, 3)
    V = z["visibility"]
    V = V[..., 0] if V.ndim == 3 else V       # (N, T)
    return P, V.astype(bool)


def umeyama(X, Y):
    """Least-squares similarity (s, R, t) with s*R@x + t ~= y. Umeyama 1991."""
    n = len(X)
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    C = (Yc.T @ Xc) / n
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_x = (Xc ** 2).sum() / n
    s = float(np.trace(np.diag(D) @ S) / var_x) if var_x > 1e-12 else 1.0
    t = my - s * (R @ mx)
    return s, R, t


def compare(Pg, Vg, Pc, Vc):
    """GT vs one candidate. Returns a metrics dict (status says whether it is usable)."""
    N = min(Pg.shape[0], Pc.shape[0])
    T = min(Pg.shape[1], Pc.shape[1])
    Pg, Vg, Pc, Vc = Pg[:N, :T], Vg[:N, :T], Pc[:N, :T], Vc[:N, :T]

    # Paper C.4 evaluation mask: v_eval[t,n] = v[t,n] AND v[0,n]. Ground-truth visibility
    # only -- a point is scored where the real video shows it and it was visible at the
    # query frame. The candidate's own visibility must NOT shrink this set: doing so scores
    # heavily-occluding candidates on fewer frames, which is what previously inverted the
    # within-scene ranking. Candidate NaNs (occluded in the generated video) are excluded
    # separately, since ViPE writes no 3D there and they cannot enter a distance.
    v_eval = Vg & Vg[:, [0]]
    finite = np.isfinite(Pg).all(-1) & np.isfinite(Pc).all(-1)
    vis = v_eval & finite

    # Scene scale: largest distance between two GT query points at frame 0.
    A = Pg[:, 0][Vg[:, 0] & np.isfinite(Pg[:, 0]).all(-1)]
    if len(A) >= 2:
        i, j = np.triu_indices(len(A), 1)
        scene_extent = float(np.linalg.norm(A[i] - A[j], axis=1).max())
    else:
        scene_extent = 0.0

    cond, pred = slice(0, SWITCH), slice(SWITCH, T)
    out = {
        "n_points": int(N), "n_frames": int(T),
        "n_eval": int(Vg[:, 0].sum()),                     # paper's N_eval = sum_n v[0,n]
        "n_scored": int(vis[:, pred].sum()),               # |S|
        "scored_frac": float(vis[:, pred].sum() / max(1, v_eval[:, pred].sum())),
    }

    # ---- align on the conditioning segment only ----
    # Nothing is dropped for tracking badly. A low overlap is a property worth scoring, not
    # a reason to abstain, and a threshold here had counted GT's own occlusion as candidate
    # failure (ball-in-basket: the ball sits inside the basket, GT is visible for 5% of the
    # window, and a candidate that tracked 76% of those frames was still marked failed).
    # Only a genuinely empty intersection produces inf, below.
    X = Pc[:, cond][vis[:, cond]]
    Y = Pg[:, cond][vis[:, cond]]
    if len(X) >= MIN_COND_PTS:
        s, R, t = umeyama(X, Y)
        resid = np.linalg.norm((s * (R @ X.T).T + t) - Y, axis=1)
        out["align"] = {"scale": s, "rot_deg": float(np.degrees(np.arccos(
            np.clip((np.trace(R) - 1) / 2, -1, 1)))), "trans_norm": float(np.linalg.norm(t)),
            "cond_resid_median_m": float(np.median(resid)),
            "cond_resid_p90_m": float(np.percentile(resid, 90)), "fitted": True}
    else:
        # too little conditioning overlap to fit a transform: fall back to identity so the
        # pair still yields a number instead of a hole in the ranking
        s, R, t = 1.0, np.eye(3), np.zeros(3)
        out["align"] = {"scale": 1.0, "rot_deg": 0.0, "trans_norm": 0.0,
                        "cond_resid_median_m": None, "cond_resid_p90_m": None,
                        "fitted": False}

    Pc_a = s * np.einsum("ij,ntj->nti", R, Pc) + t          # aligned candidate

    # ---- ADE / FDE, paper C.4 ----
    m = vis[:, pred]                                   # S = {(t,n): t >= T_cond, v_eval=1}
    err = np.linalg.norm(Pc_a[:, pred] - Pg[:, pred], axis=-1)      # (N, T_pred)
    # An empty S means nothing was tracked on one side or the other, so the error is
    # unbounded rather than unknown: inf ranks the pair last instead of dropping it.
    out["ade"] = float(err[m].mean()) if m.any() else float("inf")

    last = m[:, -1]                                    # S_T = {n: v_eval[T-1,n]=1}
    out["fde"] = float(err[:, -1][last].mean()) if last.any() else float("inf")

    # ---- penalised variant ----
    # Where the real video shows the point but the candidate lost it, there is no predicted
    # position and no distance to take. Skipping those entries rewards losing the object:
    # solid-ball-peakaboo/c1 scored on 29% of its points and came out best in the corpus at
    # 1.4 cm. Instead charge a miss the scene's own scale -- the largest distance between two
    # query points at frame 0. That is fixed by the frame-0 geometry alone, so it is identical
    # for every candidate and immune to tracking or motion-estimation error, unlike GT path
    # length. It does run light where an object travels far relative to its size (path > 3x
    # extent in 23 of 96 scenes), which errs toward false negatives rather than false alarms.
    missed = v_eval[:, pred] & ~m
    err_p = np.where(m, err, 0.0) + missed * scene_extent
    denom = (m | missed).sum()
    out["ade_penalized"] = float(err_p[m | missed].sum() / denom) if denom else float("inf")
    lastp = v_eval[:, pred][:, -1]
    if lastp.any():
        e_last = np.where(m[:, -1], err[:, -1], scene_extent)
        out["fde_penalized"] = float(e_last[lastp].mean())
    else:
        out["fde_penalized"] = float("inf")
    out["scene_extent_m"] = float(scene_extent)
    out["miss_frac"] = float(missed.sum() / max(1, v_eval[:, pred].sum()))

    out["status"] = "ok"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "outputs" / "similarity_3d"))
    ap.add_argument("--candidates", default="", help="e.g. 0,1,2 (default: all present)")
    a = ap.parse_args()
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)

    gt_dir = REPO / "outputs/runs/gt/tracks_3d/physics_iq"
    gt_files = {os.path.basename(p).replace("_merged_3d_tracks.npz", ""): p
                for p in map(str, gt_dir.glob("*_3d_tracks.npz"))}

    if a.candidates:
        ks = [int(x) for x in a.candidates.split(",")]
    else:
        ks = sorted(int(re.search(r"magi_c(\d+)$", str(d)).group(1))
                    for d in (REPO / "outputs/runs").glob("magi_c*")
                    if (d / "tracks_3d/physics_iq").is_dir())

    n_written = 0
    for vid, gpath in sorted(gt_files.items()):
        Pg, Vg = load_tracks(gpath)
        jpath = outdir / f"{vid}.json"
        doc = json.loads(jpath.read_text()) if jpath.exists() else {
            "scene": vid, "switch_frame": SWITCH, "candidates": {}}

        added = []
        for k in ks:
            cpath = REPO / f"outputs/runs/magi_c{k}/tracks_3d/physics_iq/{vid}_merged_3d_tracks.npz"
            if not cpath.exists():
                continue
            key = f"c{k}"
            if key in doc["candidates"]:
                continue                      # keep what is already there; re-run adds only new ones
            Pc, Vc = load_tracks(cpath)
            doc["candidates"][key] = compare(Pg, Vg, Pc, Vc)
            added.append(key)

        if added or not jpath.exists():
            doc["n_candidates"] = len(doc["candidates"])
            jpath.write_text(json.dumps(doc, indent=2))
            n_written += 1
    print(f"wrote/updated {n_written} scene JSONs in {outdir}")


if __name__ == "__main__":
    main()
