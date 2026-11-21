"""Diagnostics on the cached results: what the controllers actually chose, where damage happened,
TabPFN latency and prediction quality.

    python analyze.py
"""
import json
from collections import Counter

import numpy as np

from evaluate import RES
from sim import OOD_SEEDS, TEST_SEEDS


def load(name, seeds):
    return [json.loads((RES / f"{name}_{s}.json").read_text()) for s in seeds if (RES / f"{name}_{s}.json").exists()]


def main(seeds, tag):
    print(f"\n######## {tag}")
    for name in ["classical", "tabpfn", "tabpfn_prior_only", "oracle"]:
        rs = load(name, seeds)
        if not rs:
            continue
        first = Counter((round(r["history"][0]["F"]), round(r["history"][0]["tau"], 2)) for r in rs)
        dmg = [(r["seed"], [i for i, o in enumerate(r["history"]) if o["damage"]]) for r in rs if r["damage"]]
        print(f"\n== {name}")
        print("  first stroke (F, tau): ", first.most_common(4))
        print("  mean stroke time [s]:  %.2f" % np.mean([r["time"] / r["strokes"] for r in rs]))
        print("  mean |tau| / mean F:   %.2f Nm / %.1f N" % (np.mean([o["tau"] for r in rs for o in r["history"]]),
                                                         np.mean([o["F"] for r in rs for o in r["history"]])))
        print("  damage (seed, stroke idx):", dmg)
        if "decisions" in rs[0]:
            lat = [d["latency"] for r in rs for d in r["decisions"]]
            print("  TabPFN latency: median %.1f s, p90 %.1f s, %d calls" % (np.median(lat), np.percentile(lat, 90), len(lat)))
            pred = np.array([d["pred"] for r in rs for d in r["decisions"]])
            obs = np.array([[o["progress"], o["slip"], o["damage"]] for r in rs for o in r["history"]])
            err = np.abs(pred - obs).mean(0)
            print("  prediction MAE of chosen action: progress %.1f deg, slip %.1f deg, damage %.2f" % tuple(err))
            # does the controller adapt within an episode? count distinct actions per bottle
            print("  distinct actions per bottle: %.1f" % np.mean([len({(o['F'], o['tau']) for o in r["history"]}) for r in rs]))


if __name__ == "__main__":
    main(TEST_SEEDS, "in-distribution test bottles")
    main(OOD_SEEDS, "out-of-distribution bottles")
