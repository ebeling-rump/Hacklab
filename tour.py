"""A 1-minute tour of the results - no TabPFN, GPU or simulation needed.

Reads the saved results in results/*.json (written by the experiment scripts)
and prints the headline numbers, then lists the videos and the commands that
reproduce or replay each experiment.

    python tour.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
R = ROOT / "results"


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def table(title, header, rows):
    print(f"\n{title}")
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = "  " + "  ".join(f"{h:<{w}}" for h, w in zip(header, widths))
    print(line)
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print("  " + "  ".join(f"{str(x):<{w}}" for x, w in zip(r, widths)))


def main():
    print("Can a robot dog learn where the ice is from a table?  (TabPFN on a simulated Unitree A1)")

    gv = load("ice_governor.json")
    if gv:
        rows = []
        for name in ("fixed 1.0", "fixed 0.3", "detector governor (no TabPFN)",
                     "XGBoost governor (risk < 0.65)", "TabPFN governor (risk < 0.65)"):
            r = [x for x in gv if x["strategy"] == name and x["reached"]]
            if r:
                rows.append((name, f"{len(r)}/5", f"{np.mean([x['time'] for x in r]):.1f} s",
                             f"{np.mean([x['slip_s'] for x in r]):.1f} s"))
        table("1. Speed governor on ice patches (unseen), A -> B, 5 layouts",
              ("strategy", "reached", "time", "seconds slipping"), rows)

    pt = load("ice_patrol.json")
    if pt:
        rows = []
        for name in ("no map", "detector map", "TabPFN map"):
            s = [x["slip_s"] for x in pt if x["strategy"] == name and x["leg"] >= 2]
            rows.append((name, f"{np.mean(s):.1f} s", f"{np.median(s):.1f} s"))
        table("2. Patrol A <-> B with a slip map, legs 3-6, 5 layouts",
              ("strategy", "mean slipping per leg", "median"), rows)

    cd, tt = load("charts_data.json"), load("terrain_test.json")
    if cd and tt:
        f = cd["fall_roc_auc"]
        p1, p2 = tt["prediction"]["steep hills"]["fall_auc"], tt["prediction"]["moderate hills"]["fall_auc"]
        rows = [("TabPFN (2,000 rows)", f"{f['TabPFN, 2,000 rows']:.3f}", f"{p1['TabPFN']:.3f}", f"{p2['TabPFN']:.3f}"),
                ("XGBoost (18,067 rows)", f"{f['XGBoost, 18,067 rows']:.3f}", f"{p1['XGBoost']:.3f}", f"{p2['XGBoost']:.3f}"),
                ("logistic regression", f"{f['Logistic regression']:.3f}", f"{p1['logistic']:.3f}", f"{p2['logistic']:.3f}"),
                ("tilt rule (no ML)", f"{f['Tilt rule, 2 s average (no ML)']:.3f}", f"{p1['tilt 2 s (no ML)']:.3f}",
                 f"{p2['tilt 2 s (no ML)']:.3f}")]
        table("3. Fall within 1 s: ROC-AUC of frozen models on conditions never seen",
              ("model", "13-15 deg slopes", "steep hills", "moderate hills"), rows)

    dm, bb = load("danger_mode.json"), load("balancer_baseline.json")
    if dm and bb and tt:
        def falls(rows, key, kind_field, kind):
            return sum(x["fell"] for x in rows if x[key[0]] == key[1] and x[kind_field] == kind)
        rows = []
        for label, k_dm, k_tt in (("blind trot", "never brace", "blind trot"),
                                  ("always brace", "always brace", "always brace"),
                                  ("tilt rule", "tilt rule", "tilt rule"),
                                  ("TabPFN danger mode", "TabPFN", "TabPFN danger mode")):
            rows.append((label, f"{falls(dm, ('controller', k_dm), 'kind', 'dangerous')}/40",
                         f"{sum(x['fell'] for x in tt['behaviour'] if x['strategy'] == k_tt)}/60"))
        rows.append(("balance reflex (no TabPFN)",
                     f"{sum(x['fell'] for x in bb if x['controller'] == 'balancer only' and x['kind'] == 'dangerous')}/40",
                     f"{sum(x['fell'] for x in tt['behaviour'] if x['strategy'] == 'balance reflex')}/60"))
        table("4. Falls, paired walks (lower is better)",
              ("strategy", "steep 13-15 deg slopes", "real hills"), rows)
        print("  -> TabPFN's warning helps a blind robot, but a simple IMU balance reflex removes the falls.")

    videos = sorted((ROOT / "videos").glob("*comparison*.mp4")) + sorted((ROOT / "videos").glob("ice_patrol*.mp4"))
    print("\nVideos" + ("" if videos else " (not recorded yet: python a1_record_videos.py; python a1_ice_videos.py)"))
    for v in videos:
        print(f"  {v.relative_to(ROOT)}")
    print("\nReproduce / replay (need mujoco + tabpfn):")
    for cmd, what in (("python a1_ice_patches.py governor", "speed governor on ice patches"),
                      ("python a1_ice_patches.py patrol", "slip-map patrol"),
                      ("python a1_terrain_test.py", "frozen models on real hills"),
                      ("python a1_danger_replay.py --list", "list danger-mode episodes to replay"),
                      ("python a1_danger_replay.py --kind dangerous --episode 9 --controller TabPFN",
                       "watch one in the viewer"),
                      ("python a1_slip_viewer.py --mu 0.15 --slope 13.5", "live slip/fall risk viewer"),
                      ("python a1_tabpfn_charts.py", "redraw all charts")):
        print(f"  {cmd:<78} {what}")


if __name__ == "__main__":
    main()
