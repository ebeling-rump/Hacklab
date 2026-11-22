"""Map figure for the outdoor course -> results/figures/outdoor_maps.png (+ outdoor_maps_<seed>.png)."""

import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from a1slip.ice import outdoor as O  # noqa: E402

FIG = O.RESULTS / "figures"
EXT = [O.X0, O.X1, O.Y0, O.Y1]


def leg_slices(run):
    """Decision indices (every 0.2 s) belonging to each leg."""
    out, start = [], 0
    for leg in run["legs"]:
        n = int(round(leg["time"] / 0.2))
        out.append(slice(start, start + n))
        start += n
    return out


def draw_obstacles(ax, lay, color="white"):
    for x, y, r in lay["trees"]:
        ax.add_patch(plt.Circle((x, y), 0.32, color=color, zorder=3))
    for x, y, s in lay["rocks"]:
        ax.add_patch(plt.Circle((x, y), s, color=color, zorder=3))


def endpoints(ax):
    for p in (O.A, O.B):
        ax.plot(*p, "o", ms=9, mfc="white", mec="black", zorder=5)


def main(seed=0):
    lay = O.layout(seed)
    types = json.loads((O.RESULTS / "outdoor_types.json").read_text())
    runs = {r["kind"]: r for r in O.load_runs() if r["seed"] == seed}
    allruns = O.load_runs()
    cols = np.array([O.TYPES[n][1] for n in O.NAMES])
    truth = np.array([types[n]["slip_fraction"] for n in O.NAMES])[lay["grid"]]

    fig, axes = plt.subplots(2, 3, figsize=(19, 8.6))
    ax = axes[0, 0]
    ax.imshow(cols[lay["grid"]].transpose(1, 0, 2), origin="lower", extent=EXT)
    draw_obstacles(ax, lay, color="#1d3b1d")
    endpoints(ax)
    ax.set_title("Terrain (truth)", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cols[i]) for i in range(len(O.NAMES))]
    ax.legend(handles, [f"{n} (μ {O.TYPES[n][0]:.2f})" for n in O.NAMES], fontsize=7, loc="lower left",
              ncol=2, framealpha=0.85)
    ax = axes[0, 1]
    im = ax.imshow(truth.T, origin="lower", extent=EXT, cmap="magma", vmin=0, vmax=1)
    draw_obstacles(ax, lay)
    endpoints(ax)
    ax.set_title("True slip rate (measured per terrain type)", loc="left")

    panels = [("obstacles only", axes[0, 2], "Obstacles only (no slip map)"),
              ("felt map", axes[1, 0], "Felt map only"),
              ("felt + seen (XGBoost)", axes[1, 1], "Felt + seen, XGBoost"),
              ("felt + seen (TabPFN)", axes[1, 2], "Felt + seen, TabPFN")]
    for kind, ax, title in panels:
        r = runs.get(kind)
        if r is None:
            ax.set_axis_off()
            continue
        # The map the robot planned trip 2 with, and its route on trip 2.
        risk = np.array(r["risk_leg_start"].get("1", r["risk_end"]))
        img = truth.T if kind == "obstacles only" else risk.T
        ax.imshow(img, origin="lower", extent=EXT, cmap="magma", vmin=0, vmax=1,
                  alpha=0.35 if kind == "obstacles only" else 1.0)
        draw_obstacles(ax, lay)
        tr = np.array(r["trace"])
        sl = leg_slices(r)
        if len(sl) > 1:
            seg = tr[sl[1]]
            ax.plot(seg[:, 0], seg[:, 1], color="#19c39a", lw=2.6, zorder=4)
        endpoints(ax)
        rs = [x for x in allruns if x["kind"] == kind]
        later = [l["slip_s"] for x in rs for l in x["legs"] if l["leg"] >= 1]
        ax.set_title(f"{title}: {np.mean(later):.1f} s slipping per trip (trips 2–4)", loc="left")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("P(slip)")
    fig.suptitle("Outdoor course, layout %d: the map each robot planned trip 2 with, and its route on trip 2 (green)"
                 % (seed + 1), x=0.02, ha="left", fontsize=13)
    FIG.mkdir(exist_ok=True)
    out = FIG / ("outdoor_maps.png" if seed == 0 else f"outdoor_maps_{seed}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
