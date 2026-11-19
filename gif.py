#!/usr/bin/env python3
"""gif.py -- a GIF of the organ from a traced run.

    python3 gif.py results/turns_x3_s0.json --from 400 --to 520 --out cuts.gif --title "12 of 16 tissues cut"

Cells by kind (red core, blue active, grey sensory); each tissue's halo by form (teal B, orange A), brighter as
its gain is higher; the title says which memory has access.
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

COL = {0: "#d62728", 1: "#1f77b4", 2: "#7f7f7f"}
FORM = {True: "#1A7F8E", False: "#D9822B"}
ZOOM = 0.15
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import AMP, TURN_EPS as EPS, RING_SPLIT


def make(path, out, t0=1, t1=None, every=1, fps=12, title="", hold=6, lim=2.8):
    d = json.load(open(path))
    ph = {t: np.array(p) for t, p in d["clock_phase_t"]}
    acc = {t: (k, np.array(x)) for (t, k, U, ev, c), (_, x) in zip(d["access"], d["access_x"])}
    T = len(d["types_t"])
    t1 = min(t1 or T, T)
    steps = list(range(t0 - 1, t1, every)) + [t1 - 1] * hold
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    centre = np.array(d["organ_pos_t"][steps[0]])[:, 0]

    def draw(t):
        Pg, dl = np.array(d["organ_pos_t"][t]), np.array(d["form_dl_t"][t])
        own, ty, P = np.array(d["own_t"][t]), np.array(d["types_t"][t]), np.array(d["pos_t"][t])
        g = 1 + AMP * np.sin(ph[t + 1]) if (t + 1) in ph and ph[t + 1].size == Pg.shape[1] else np.ones(Pg.shape[1])
        k, x = acc.get(t + 1, (None, None))
        if x is not None and x.size == g.size:
            g = g * (1 + EPS * x)
        ax.cla()
        for j in range(Pg.shape[1]):
            ax.add_patch(plt.Circle(Pg[:, j], 0.33, color=FORM[bool(j < dl.size and dl[j] >= 0)],
                                    alpha=float(np.clip(0.05 + 0.5 * (g[j] - 0.6), 0.03, 0.6)), lw=0))
        X = Pg[:, own] + ZOOM * P
        for kk in (2, 1, 0):
            m = ty == kk
            ax.plot(X[0, m], X[1, m], "o", color=COL[kk], ms=2.5 if kk else 3.5)
        ax.add_patch(plt.Circle(Pg[:, 0], RING_SPLIT, fill=False, ls=":", lw=0.8, color="k"))
        ax.set_aspect("equal"); ax.set_xlim(centre[0] - lim, centre[0] + lim); ax.set_ylim(centre[1] - lim, centre[1] + lim)
        ax.set_xticks([]); ax.set_yticks([])
        on = "" if k is None else ("   access: the inner ring's memory" if k == 0 else "   access: the outer ring's memory")
        ax.set_title((f"{title}\n" if title else "") + f"step {t + 1}: {Pg.shape[1]} tissues, {own.size} cells{on}", fontsize=10)
    FuncAnimation(fig, draw, frames=steps, interval=1000 / fps).save(out, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print("wrote", out, f"({len(steps)} frames)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--from", dest="t0", type=int, default=1); ap.add_argument("--to", dest="t1", type=int)
    ap.add_argument("--every", type=int, default=1); ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--title", default=""); ap.add_argument("--out", default="run.gif")
    a = ap.parse_args()
    make(a.run, a.out, a.t0, a.t1, a.every, a.fps, a.title)


if __name__ == "__main__":
    main()
