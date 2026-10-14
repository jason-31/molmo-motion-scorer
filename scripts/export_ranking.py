#!/usr/bin/env python3
"""Export trajectory-similarity rankings in the stage_3 scorer format.

Reads outputs/similarity_3d/*.json and writes one JSONL row per
(scene, method, candidate):

    {"scene", "method", "candidate", "rank", "score", "max_score"}

Ranking convention: rank 1 is best, rank 15 worst, and rank is the ascending order of
`score`, as in outputs/stage_3/oracle/physics-iq-verified.jsonl. `score` is the error
itself, so smaller is better and the worst possible score is +inf -- what a pair with
nothing tracked on either side receives. `max_score` holds the magnitude of the rank-1
score, i.e. the best error in the scene.

Every candidate gets a number -- there are no nulls. Poor tracking is scored, not
excluded; only a completely empty overlap becomes infinite.
"""
import argparse, json, re
from pathlib import Path

import numpy as np

REPO = Path("/mmfs1/gscratch/krishna/renjiz2/2026/physics_realism/molmomotion")
DST = Path("/gscratch/krishna/renjiz2/2026/physics_realism/vlm_scorer/outputs/stage_3/trajectory/molmomotion.jsonl")
VID = re.compile(r"^(\d+)_full-videos_24FPS_perspective-(\w+)_take-1_(.+)$")
METHODS = ["ade", "ade_penalized", "fde", "fde_penalized"]


def _round(v):
    """Value as it will be serialised, so ranks and written scores agree."""
    return v if np.isinf(v) else round(v, 6)


def scene_key(vid):
    """GT video id -> the scene id used by the scorer files."""
    m = VID.match(vid)
    return f"{m.group(1)}_perspective-{m.group(2)}_{m.group(3)}" if m else vid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "outputs" / "similarity_3d"))
    ap.add_argument("--out", default=str(DST))
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    n_scene = 0
    for f in sorted(Path(a.src).glob("*.json")):
        doc = json.loads(f.read_text())
        scene = scene_key(doc["scene"])
        cands = doc["candidates"]
        if not cands:
            continue
        n_scene += 1
        for method in METHODS:
            # Rank on the rounded value that actually gets written, otherwise two errors
            # that differ only past the 6th decimal (0.003729694 vs 0.003730500) land as
            # equal scores carrying different ranks, and the file contradicts itself.
            vals = [(int(k[1:]), _round(float(c.get(method, float("inf")))))
                    for k, c in cands.items()]
            vals.sort(key=lambda x: x[1])          # ascending error: best first
            best = vals[0][1]
            # Competition ranking, as in the oracle and the VLM scorer files: equal scores
            # share the smallest rank and the next distinct score skips the positions they
            # occupy (1,2,2,4). Numbering ties sequentially would invent an ordering that
            # the metric does not support -- and 6 groups here are entirely tied at inf.
            prev, prev_rank = None, 0
            for i, (idx, v) in enumerate(vals, 1):
                rank = prev_rank if (prev is not None and v == prev) else i
                prev, prev_rank = v, rank
                rows.append({"scene": scene, "method": method,
                             "candidate": f"candidate_{idx}", "rank": rank,
                             "score": v,
                             "max_score": best if np.isinf(best) else abs(best)})

    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows for {n_scene} scenes x {len(METHODS)} methods -> {out}")


if __name__ == "__main__":
    main()
