"""run.py -- run the model over seeds, one JSON per run (every step traced).  Runs already in --out are skipped.

    python3 run.py run --name turns_x3 --cut-at 420 480 540 --cut-n 12 --seeds 0-2 --cores 3 --out results
    python3 run.py run --name inner_x3 --cut-at 420 480 540 --cut-n 6 --cut-what inner --seeds 0-2 --cores 3
    python3 run.py run --name free --pace 0        # turn-taking not paced by the core's clock
    python3 run.py run --name noturns --no-turns   # no turn-taking
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
TRACE = ("N", "J", "organ_N", "organ_emit", "pos", "types", "p", "sen", "own", "views", "organ_pos", "organ_types",
         "organ_sec", "organ_sen", "form_dl")


def job_path(out, name, seed):
    return os.path.join(out, f"{name}_s{seed}.json")


def export(o, name, seed, params):
    """A run's JSON: a summary, the events and every step's trace."""
    import numpy as np
    import model
    r, own = o["rec"], np.array(o["own"])
    ty = model.softmax(o["V"]["l"]).argmax(axis=0)
    gty = model.softmax(o["V"]["g"]).argmax(axis=0)
    res = dict(name=name, seed=seed, params=params, seconds=o["seconds"], N_final=o["N_final"], F=o["F"],
               K_final=o["NG"], own_final=own.tolist(), organ_counts=[int((gty == t).sum()) for t in range(3)],
               counts_by_ens={int(k): [int(((ty == t) & (own == k)).sum()) for t in range(3)] for k in range(o["NG"])},
               windings_final=r["clock_shadow"][-1][3] if r["clock_shadow"] else None, organ_types_final=gty.tolist())
    res.update({k + "_t": r[k] for k in TRACE})
    for k in ("divisions", "celled_births", "organ_divisions", "removals", "organ_removals", "form_ob", "clock_group",
              "clock_omega", "clock_phase", "clock_drive", "clock_hold", "clock_shadow", "clock_nu", "access", "access_x"):
        res[k + "_t" if k in ("form_ob", "clock_phase") else k] = r[k]
    return res


def one(a):
    sys.path.insert(0, HERE)
    import model
    params = dict(nY=a.nY, cuts=(a.cut_at, a.cut_n, a.cut_what) if a.cut_at else None, turns=not a.no_turns,
                  pace=a.pace, core_T=a.core_T)
    try:
        res = export(model.run(seed=a.seed, **params), a.name, a.seed, params)
    except FloatingPointError as e:
        res = dict(name=a.name, seed=a.seed, params=params, error=str(e))
    os.makedirs(a.out, exist_ok=True)
    with open(job_path(a.out, a.name, a.seed), "w") as f:
        json.dump(res, f)


def launch(a, seed):
    env = dict(os.environ, **{v: "1" for v in THREAD_VARS})
    cmd = [sys.executable, os.path.abspath(__file__), "one", "--name", a.name, "--seed", str(seed), "--out", a.out,
           "--nY", str(a.nY), "--pace", str(a.pace)]
    if a.cut_at:
        cmd += ["--cut-at", *map(str, a.cut_at), "--cut-n", str(a.cut_n), "--cut-what", a.cut_what]
    cmd += ["--no-turns"] * a.no_turns + (["--core-T", str(a.core_T)] if a.core_T else [])
    t = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return seed, r.returncode, time.time() - t, r.stderr[-1500:]


def run_sweep(a):
    os.makedirs(a.out, exist_ok=True)
    seeds = [s for s in a.seeds if not os.path.exists(job_path(a.out, a.name, s))]
    if not seeds:
        print(f"{a.name}: every requested result already exists")
        return
    workers = min(a.cores or os.cpu_count() or 1, len(seeds))
    print(f"{a.name}: {len(seeds)} runs, {workers} at once, nY = {a.nY}", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for f in as_completed([ex.submit(launch, a, s) for s in seeds]):
            s, rc, dt, err = f.result()
            r = json.load(open(job_path(a.out, a.name, s))) if rc == 0 else {"error": f"exit {rc}\n{err}"}
            if r.get("error"):
                print(f"   seed {s:<3} FAILED: {r['error']}", flush=True)
            else:
                oc, ok = r["organ_counts"], sum(c in ([1, 6, 9], [1, 3, 12]) for c in r["counts_by_ens"].values())
                print(f"   seed {s:<3} {dt:5.0f}s   organ ({oc[0]},{oc[1]},{oc[2]})  blankets of a form: {ok}/{r['K_final']}  "
                      f"windings {r['windings_final']}", flush=True)


def parse_seeds(tokens):
    out = []
    for t in tokens:
        lo, hi = t.split("-") if "-" in t else (t, t)
        out += list(range(int(lo), int(hi) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "one"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--nY", type=int, default=720)
    ap.add_argument("--cut-at", type=int, nargs="+", help="steps at which whole blankets are cut")
    ap.add_argument("--cut-n", type=int, default=12, help="blankets cut each time")
    ap.add_argument("--cut-what", default="random", choices=["random", "inner"], help="inner: the inner ring first")
    ap.add_argument("--no-turns", action="store_true", help="no turn-taking")
    ap.add_argument("--pace", type=float, default=0.6, help="how far the core's clock moves the threshold (0: free)")
    ap.add_argument("--core-T", type=float, help="the core clock's period (default: its own, ~16)")
    ap.add_argument("--seeds", nargs="+", default=["0"]); ap.add_argument("--cores", type=int)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    if a.cmd == "one":
        one(a)
        return
    a.seeds = parse_seeds(a.seeds)
    run_sweep(a)


if __name__ == "__main__":
    main()
