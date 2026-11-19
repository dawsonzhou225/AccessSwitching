"""fig.py -- the figures, from traced runs.

    python3 fig.py turns results/turns_x3_s0.json ... --out turns.png        # which memory has access, read back
    python3 fig.py rhythm results/free_s0.json ... --out rhythm.png          # switches by the core's phase; epoch lengths
    python3 fig.py position results/turns_x3_s0.json ... --out position.png  # each ring's offset against the walk
    python3 fig.py levelt results/inner_x3_s0.json ... --out levelt.png      # epoch length against the bits' strengths
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import AMP, TURN_EPS, RING_SPLIT, AT, foxhound

cw = lambda x: np.angle(np.exp(1j * x))
CI, CO, CP, CG = "#1A7F8E", "#D9822B", "#5B3FA6", "#8A8A8A"     # inner memory, outer memory, paced, grey


def latest(rec, t, n):
    """The last record at or before t whose list has n entries, else None."""
    out = None
    for s, x in rec:
        if s <= t and len(x) == n:
            out = np.array(x)
    return out


# ---- position ------------------------------------------------------------------------------
def wave(p, P, g, ring, m=None):
    mem = (g == ring) & (np.arange(p.size) != 0)
    n = int(mem.sum())
    if n == 0:
        return n, None, np.nan
    th = np.arctan2(P[1] - P[1, 0], P[0] - P[0, 0])
    ts = np.sort(th[mem])
    gap = float(np.max(np.diff(np.r_[ts, ts[0] + 2 * np.pi]))) if n >= 2 else 2 * np.pi
    if m is None:
        if n < 3 or gap >= np.radians(150):                                   # a wide gap hides the winding
            return n, None, np.nan
        q = p[mem][np.argsort(th[mem])]
        m = int(np.round(cw(np.roll(q, -1) - q).sum() / (2 * np.pi)))
    return n, m, float(np.angle(np.mean(np.exp(1j * (p[mem] - m * th[mem])))))


def position(d, t0=300, t1=None, fit=(300, 380)):
    """Each ring's offset minus where the walk put it, in cycles, after a line fitted over `fit`."""
    ph = {t: np.array(p) for t, p in d["clock_phase_t"]}
    walk = {t: np.array(x) for t, v, x in d["clock_drive"]}
    T = np.arange(t0, t1 or len(d["organ_pos_t"]))
    off, X, last = np.full((T.size, 2), np.nan), np.zeros((T.size, 2)), [None, None]
    for i, t in enumerate(T):
        P = np.array(d["organ_pos_t"][t - 1])                                 # phases at t go with positions saved at t - 1
        if P.shape[1] != ph[t].size:
            P = np.array(d["organ_pos_t"][t])
        g = (np.hypot(P[0] - P[0, 0], P[1] - P[1, 0]) >= RING_SPLIT).astype(int)
        g[0] = 1
        for r in (0, 1):
            n, m, o = wave(ph[t], P, g, r)
            if m is not None:
                last[r] = m
            elif n > 0 and last[r] is not None:
                o = wave(ph[t], P, g, r, last[r])[2]
            off[i, r] = o
        X[i] = walk.get(t, np.zeros(2))
    err = np.full(off.shape, np.nan)
    sel = (T >= fit[0]) & (T < fit[1])
    for r in (0, 1):
        y = np.unwrap(off[sel, r]) - 2 * np.pi * X[sel, r]
        c = np.linalg.lstsq(np.c_[np.ones(sel.sum()), T[sel]], y, rcond=None)[0]
        err[:, r] = cw(off[:, r] - (c[0] + c[1] * T + 2 * np.pi * X[:, r])) / (2 * np.pi)
    return T, err


# ---- turn-taking ---------------------------------------------------------------------------
def field(pos, sec, g):
    """The field each tissue senses from the others (cells taken at their tissue), emissions scaled by g."""
    return foxhound(np.asarray(pos), np.asarray(sec) * g[None, :])


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def readout(d):
    """Per step with access: (t, which is on, match to the inner memory, to the outer, the core's phase).
    The residual of what the tissues sense after the clocks' part, against the field each memory would add."""
    ph = {t: np.array(p) for t, p in d["clock_phase_t"]}
    ob = {t: np.array(v) for t, v in d["form_ob_t"]}
    rows = []
    for t, k, U, ev, cph in d["access"]:
        u = t - 1                                                             # the access chosen at t shaped the field saved at t - 1
        if u not in ph or u not in ob:
            continue
        pos, n = np.array(d["organ_pos_t"][u]), ph[u].size
        dl = np.array(d["form_dl_t"][u])
        grp = latest(d["clock_group"], u, n)
        if pos.shape[1] != n or dl.size != n or ob[u].size != n:              # a tissue being born
            continue
        if grp is None:
            grp = (np.hypot(pos[0] - pos[0, 0], pos[1] - pos[1, 0]) >= RING_SPLIT).astype(int)
            grp[0] = 1
        th = np.arctan2(pos[1] - pos[1, 0], pos[0] - pos[0, 0])
        P0 = -np.sign(np.where(grp == 0, dl, ob[u])) * np.cos(th)
        P1 = -np.sign(np.where(grp == 1, dl, ob[u])) * np.sin(th)
        P0[0] = P1[0] = 0.0
        g0, sec = 1 + AMP * np.sin(ph[u]), d["organ_sec_t"][u]
        f0 = field(pos, sec, g0)
        r = (np.array(d["organ_sen_t"][u]) - f0).ravel()
        rows.append((t, k, corr(r, (field(pos, sec, g0 * (1 + TURN_EPS * P0)) - f0).ravel()),
                     corr(r, (field(pos, sec, g0 * (1 + TURN_EPS * P1)) - f0).ravel()), cph))
    return np.array(rows)


def epochs(d):
    """Whole epochs: (start, end, which is on, mean strength of the inner and outer memory, core phase at the start)."""
    a = d["access"]
    t, k = np.array([r[0] for r in a]), np.array([r[1] for r in a])
    ev, ph = np.array([r[3] for r in a]), np.array([r[4] for r in a])
    sw = np.r_[0, np.flatnonzero(np.diff(k) != 0) + 1, len(k)]
    return [(t[s], t[e - 1] + 1, k[s], ev[s:e].mean(axis=0), ph[s]) for s, e in zip(sw[1:-2], sw[2:-1])]


def fig_turns(ds, names, out):
    R = [readout(d) for d in ds]
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw=dict(width_ratios=[2.2, 1, 1.2], wspace=0.28))
    ax, r = axs[0], R[0]
    w = (r[:, 0] >= AT["turns"]) & (r[:, 0] < AT["turns"] + 120)
    for t, k in r[w][:, :2]:
        ax.axvspan(t - 0.5, t + 0.5, color=CI if k == 0 else CO, alpha=0.13, lw=0)
    ax.plot(r[w, 0], r[w, 2], color=CI, lw=2, label="match to the inner ring's memory")
    ax.plot(r[w, 0], r[w, 3], color=CO, lw=2, label="match to the outer ring's memory")
    ax.set_xlabel("step"); ax.set_ylabel("match of what the tissues sense"); ax.grid(alpha=0.2); ax.legend(fontsize=9, loc="lower right")
    ax.set_title(f"{names[0]}: shading, the memory with access; lines, read back from what the tissues sense", fontsize=10)
    ax, X = axs[1], np.arange(2)
    on = [np.mean([r[r[:, 1] == j, 2 + j].mean() for r in R]) for j in (0, 1)]
    off = [np.mean([r[r[:, 1] == 1 - j, 2 + j].mean() for r in R]) for j in (0, 1)]
    avg = [np.mean([r[:, 2 + j].mean() for r in R]) for j in (0, 1)]
    ax.bar(X - 0.27, on, 0.26, color=[CI, CO], label="in its own epochs")
    ax.bar(X, off, 0.26, color=[CI, CO], alpha=0.35, label="in the other's epochs")
    ax.bar(X + 0.27, avg, 0.26, color=CG, label="averaged over all steps")
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(X); ax.set_xticklabels(["inner ring's\nmemory", "outer ring's\nmemory"])
    ax.grid(alpha=0.2, axis="y"); ax.legend(fontsize=8.5)
    right = np.mean(np.concatenate([(r[:, 3] > r[:, 2]) == (r[:, 1] == 1) for r in R]))
    ax.set_title(f"read step by step: {right * 100:.1f}% right ({len(R)} runs);\naveraged over time, each half there", fontsize=10)
    ax = axs[2]
    for d, r, nm in zip(ds, R, names):
        ok = (r[:, 3] > r[:, 2]) == (r[:, 1] == 1)
        edges = np.arange(AT["turns"], r[-1, 0] + 31, 30)
        mid = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:]) if ((r[:, 0] >= a) & (r[:, 0] < b)).any()]
        ax.plot(mid, [ok[(r[:, 0] >= a) & (r[:, 0] < b)].mean() * 100 for a, b in zip(edges[:-1], edges[1:])
                      if ((r[:, 0] >= a) & (r[:, 0] < b)).any()], marker="o", label=nm)
        for c in sorted(set(x[0] for x in d["organ_removals"])):
            ax.axvline(c, color="k", ls="--", lw=0.8)
    ax.set_xlabel("step (dashed: cuts)"); ax.set_ylabel("% of steps read right"); ax.grid(alpha=0.2); ax.legend(fontsize=8)
    ax.set_title("through the cuts", fontsize=10)
    fig.savefig(out, dpi=100, bbox_inches="tight")


def fig_rhythm(ds, names, out):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(wspace=0.3))
    E, Rl = [], []
    for d in ds:
        ep = epochs(d)
        E.append([e - s for s, e, k, ev, ph in ep])
        Rl.append(abs(np.mean(np.exp(1j * np.array([ph for s, e, k, ev, ph in ep])))))
    axs[0].boxplot(E, widths=0.5, showfliers=False)
    axs[0].set_xticks(range(1, len(ds) + 1)); axs[0].set_xticklabels(names, fontsize=8.5); axs[0].set_ylabel("epoch length (steps)")
    axs[0].grid(alpha=0.2, axis="y"); axs[0].set_title("how long each memory stays on", fontsize=10)
    axs[1].bar(range(len(ds)), Rl, color=CP); axs[1].set_xticks(range(len(ds))); axs[1].set_xticklabels(names, fontsize=8.5)
    axs[1].set_ylim(0, 1); axs[1].set_ylabel("locking of the switches to the core's phase (R)"); axs[1].grid(alpha=0.2, axis="y")
    axs[1].set_title("when the switches come", fontsize=10)
    fig.savefig(out, dpi=100, bbox_inches="tight")


def fig_position(ds, names, out):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(wspace=0.25))
    for d, nm in zip(ds, names):
        T, e = position(d)
        for r in (0, 1):
            axs[r].plot(T, e[:, r], lw=1.4, label=f"{nm}: {e[-1, r]:+.4f} at {T[-1]}")
    for r, ax in enumerate(axs):
        for c in sorted(set(x[0] for x in ds[0]["organ_removals"])):
            ax.axvline(c, color="k", ls="--", lw=0.8)
        ax.axvspan(AT["walk"], T[-1], color="#EEEEEE", zorder=-1)
        ax.set_ylim(-0.05, 0.05); ax.grid(alpha=0.2); ax.legend(fontsize=8, loc="lower left"); ax.set_xlabel("step (grey: walking)")
        ax.set_ylabel("position error (cycles)"); ax.set_title(f"{'inner' if r == 0 else 'outer'} ring", fontsize=10)
    fig.savefig(out, dpi=100, bbox_inches="tight")


def fig_levelt(ds, names, out):
    rows = []
    for d in ds:
        for s, e, k, ev, ph in epochs(d):
            if min(ev) > 0.02:
                rows.append((e - s, ev[1 - k], ev[k], k == 0, np.cos(ph), np.sin(ph)))
    r = np.array(rows, float)
    X = np.c_[np.ones(len(r)), np.log(r[:, 1]), np.log(r[:, 2]), r[:, 3:]]
    y = np.log(r[:, 0])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * np.var(y - X @ b) * len(r) / (len(r) - X.shape[1]))
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(wspace=0.3))
    lag, S = np.arange(-10, 56), []
    for d in ds:
        ev = {t: e for t, k, U, e, p in d["access"]}
        for c in sorted(set(x[0] for x in d["organ_removals"])):
            S.append([ev.get(c + l, [np.nan, np.nan]) for l in lag])
    if S:
        S = np.nanmean(np.array(S, float), axis=0)
        axs[0].plot(lag, S[:, 0], color=CI, lw=2, label="inner ring's memory"); axs[0].plot(lag, S[:, 1], color=CO, lw=2, label="outer ring's memory")
    axs[0].axvline(0, color="k", ls="--", lw=1); axs[0].set_ylim(0, 1.05); axs[0].grid(alpha=0.2); axs[0].legend(fontsize=9)
    axs[0].set_xlabel("steps since a cut"); axs[0].set_ylabel("strength (how clearly its ring shows it)"); axs[0].set_title("after a cut", fontsize=10)
    for ax, j, lab in ((axs[1], 1, "the memory that is off"), (axs[2], 2, "the memory that is on")):
        o = [i for i in range(X.shape[1]) if i not in (0, j)]
        ya = np.exp(y - X[:, o] @ b[o])
        for c, col, nm in ((1, CI, "inner on"), (0, CO, "outer on")):
            m = r[:, 3] == c
            ax.scatter(r[m, j], ya[m], s=18, color=col, alpha=0.7, label=nm)
        g = np.linspace(r[:, j].min(), 1, 50)
        ax.plot(g, np.exp(b[0] + b[j] * np.log(g)), color="k", lw=2, label=f"length ~ strength^{b[j]:+.2f} (+-{se[j]:.2f})")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.2, which="both"); ax.legend(fontsize=8.5)
        ax.set_xlabel(f"strength of {lab} (mean over the epoch)"); ax.set_ylabel("epoch length (steps, adjusted)")
        ax.set_title(f"{len(r)} epochs", fontsize=10)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"length ~ off^{b[1]:+.2f} (+-{se[1]:.2f}) on^{b[2]:+.2f} (+-{se[2]:.2f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["turns", "rhythm", "position", "levelt"])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    ds = [json.load(open(p)) for p in a.runs]
    names = [os.path.basename(p)[:-5] for p in a.runs]
    out = a.out or f"{a.what}.png"
    dict(turns=fig_turns, rhythm=fig_rhythm, position=fig_position, levelt=fig_levelt)[a.what](ds, names, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
