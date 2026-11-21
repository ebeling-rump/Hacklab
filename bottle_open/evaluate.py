"""Evaluate controllers on the 40 unseen test bottles.

    python evaluate.py --controllers classical oracle          # local, seconds
    python evaluate.py --controllers tabpfn                    # cloud TabPFN, needs TABPFN_TOKEN
    python evaluate.py --controllers tabpfn --bottles 5        # first 5 test bottles only

Results are cached per (controller, bottle) in data/results/, so an interrupted TabPFN run
resumes where it stopped. `python evaluate.py --summary` prints the table from the cache.

Metrics per bottle:
  opened        cap off within MAX_STROKES (40) strokes
  time          simulated seconds (strokes + regrasps), excluding model latency
  strokes       actions used
  exploratory   strokes that did not move the cap (slip, or not enough torque)
  slip_strokes  strokes with > 5 deg slip
  damage        strokes where the grip exceeded the bottle's crush limit (excessive force)
  force_ratio   peak grip force / grip force actually needed for the breakaway torque
"""
import argparse
import json
from pathlib import Path

import numpy as np

from controllers import ClassicalAdaptive, Oracle, TabPFNController
from features import PROGRESS_MIN, SLIP_MIN
from sim import OOD_SEEDS, TEST_SEEDS, run_episode, sample_bottle

DATA = Path(__file__).parent / "data"
RES = DATA / "results"
FRESH = False
CONTROLLERS = ["classical", "tabpfn", "tabpfn_prior_only", "oracle"]


def make(name, bottle):
    if name == "classical":
        return ClassicalAdaptive()
    if name == "oracle":
        return Oracle(bottle)
    prior = np.load(DATA / "prior.npz")
    return TabPFNController(prior["X"], prior["Y"], use_own=(name == "tabpfn"))


def metrics(ep):
    h, b = ep["history"], ep["bottle"]
    f_need = b["t_break"] / (2 * b["mu"] * 0.016)
    return {
        "opened": bool(ep["opened"]), "time": ep["time"], "strokes": len(h),
        "exploratory": sum(o["progress"] <= PROGRESS_MIN for o in h),
        "slip_strokes": sum(o["slip"] > SLIP_MIN for o in h),
        "damage": sum(o["damage"] for o in h),
        "force_ratio": max(o["fn"] for o in h) / f_need,
    }


def run(name, seed):
    path = RES / f"{name}_{seed}.json"
    if path.exists() and not FRESH:
        return json.loads(path.read_text())
    b = sample_bottle(seed)
    ctrl = make(name, b)
    ep = run_episode(b, ctrl, seed=seed)
    out = {"controller": name, "seed": seed, **metrics(ep), "bottle": ep["bottle"], "history": ep["history"]}
    if hasattr(ctrl, "log"):
        out["decisions"] = ctrl.log
    RES.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=lambda o: o.item()))
    return out


def summary(names, seeds):
    print(f"\n{len(seeds)} unseen bottles ({'out-of-distribution' if seeds[0] in OOD_SEEDS else 'in-distribution'})")
    print(f"{'controller':<20}{'opened':>8}{'time s':>9}{'strokes':>9}{'explor.':>9}{'slips':>7}"
          f"{'damage':>8}{'dmg btl':>8}{'F/Fneed':>9}")
    table = {}
    for n in names:
        rs = [json.loads((RES / f"{n}_{s}.json").read_text()) for s in seeds if (RES / f"{n}_{s}.json").exists()]
        if not rs:
            continue
        ok = [r for r in rs if r["opened"]]
        row = {
            "n": len(rs), "opened": np.mean([r["opened"] for r in rs]),
            "time": np.mean([r["time"] for r in ok]) if ok else np.nan,
            "strokes": np.mean([r["strokes"] for r in ok]) if ok else np.nan,
            "exploratory": np.mean([r["exploratory"] for r in rs]),
            "slip_strokes": np.mean([r["slip_strokes"] for r in rs]),
            "damage": np.mean([r["damage"] for r in rs]),
            "damaged_bottles": np.mean([r["damage"] > 0 for r in rs]),
            "force_ratio": np.median([r["force_ratio"] for r in rs]),
        }
        table[n] = row
        print(f"{n + f' ({len(rs)})':<20}{row['opened']:>8.0%}{row['time']:>9.1f}{row['strokes']:>9.1f}"
              f"{row['exploratory']:>9.1f}{row['slip_strokes']:>7.1f}{row['damage']:>8.2f}"
              f"{row['damaged_bottles']:>8.0%}{row['force_ratio']:>9.2f}")
    print("time / strokes: mean over opened bottles; exploratory, slips, damage: mean per bottle;"
          " dmg btl: share of bottles damaged at least once; F/Fneed: median")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+", default=["classical", "oracle"], choices=CONTROLLERS)
    ap.add_argument("--set", default="test", choices=["test", "ood"], help="in-distribution or out-of-distribution bottles")
    ap.add_argument("--bottles", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", help="run only these seeds (must belong to --set)")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore and overwrite cached results")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    seeds = list(TEST_SEEDS if a.set == "test" else OOD_SEEDS)[:a.bottles]
    FRESH = a.fresh
    if a.seeds:
        run_seeds = [s for s in a.seeds if s in seeds]
        assert len(run_seeds) == len(a.seeds), "--seeds must belong to the chosen --set"
    else:
        run_seeds = seeds
    if not a.summary:
        for n in a.controllers:
            for s in run_seeds:
                r = run(n, s)
                lat = np.mean([d["latency"] for d in r.get("decisions", [])]) if r.get("decisions") else 0
                print(f"{n:<18} bottle {s}: opened={r['opened']!s:<5} strokes={r['strokes']:>2} "
                      f"explor={r['exploratory']:>2} damage={r['damage']} "
                      f"F/Fneed={r['force_ratio']:.2f}" + (f"  latency {lat:.1f}s" if lat else ""), flush=True)
    table = summary(CONTROLLERS if a.summary else a.controllers, seeds)
    if a.summary:
        out = Path(__file__).parent / "results"
        out.mkdir(exist_ok=True)
        (out / f"summary_{a.set}.json").write_text(json.dumps(table, indent=1, default=float))
