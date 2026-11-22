"""Explainer video from the simulation results (1920x1080, 30 fps, no sound).

    python make_video.py            # renders video/scene_*.mp4 and video/tabpfn_terrain_navigation.mp4
    python make_video.py --stills   # a few full-size check frames in video/stills/
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

from scipy.stats import norm

from terrain_sim import CELL, N, OBSTACLE, SLIP_THRESHOLD, make_dataset, make_site, observe

W, H, FPS = 1920, 1080, 30
OUT = Path("video")
OUT.mkdir(exist_ok=True)

BG, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#9a988f"
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
SERIES = {"TabPFN v2": BLUE, "CatBoost": ORANGE, "XGBoost (tuned)": GREEN, "Random forest": "#eda100",
          "Cross-site + local XGBoost": "#e87ba4", "TabPFN v2 + archive": "#008300"}
TERRAIN_CM = ListedColormap(["#c8a27a", "#9bc26b", "#5f8f3a", "#2f5a2a", "#6b4a2e", "#e6d39a", "#8c8c8c", "#222222"])
TERRAIN_NAMES = ["Dirt track", "Short grass", "Tall grass", "Bush", "Mud", "Sand", "Rubble", "Tree / boulder"]

plt.rcParams.update({"font.family": "Inter", "text.color": INK, "axes.edgecolor": "#d9d7d0",
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})

SITE_ID, EPISODE = 2009, 3
SHOWN = ["Geometric rule", "Cross-site XGBoost", "CatBoost", "TabPFN v2"]
SHOWN_SUB = ["Hand-tuned height,\nroughness, slope", "Trained on 20 other sites,\nno local data",
             "256 local cells", "256 local cells,\nno training step"]


# ---------------------------------------------------------------- data
def load():
    z = np.load("results/nav_paths.npz")
    nav = pd.read_csv("results/navigation.csv")
    site = make_site(SITE_ID)
    # replay navigate.py's random stream to recover this site's 256 calibration cells
    rng = np.random.default_rng(SITE_ID)
    rr, cc = np.nonzero(site.terrain[:, N // 2:] != OBSTACLE)
    cc = cc + N // 2
    observe(site, rr, cc, rng.uniform(3, 12, len(rr)), rng)
    _, ytr, r, c = make_dataset(site, 256, "calib", rng)
    # nearest-neighbour drive through the calibration cells
    pts = np.column_stack([r, c]).astype(float) + 0.5
    order, left = [int(np.argmin(pts[:, 0] + pts[:, 1]))], set(range(len(pts)))
    left.remove(order[0])
    while left:
        last = pts[order[-1]]
        nxt = min(left, key=lambda i: (pts[i, 0] - last[0]) ** 2 + (pts[i, 1] - last[1]) ** 2)
        order.append(nxt)
        left.remove(nxt)
    calib = dict(pts=pts[order], y=ytr[order])
    paths, outcome, length = {}, {}, {}
    for m in SHOWN:
        p = z[f"{SITE_ID}|{EPISODE}|{m}"].astype(float) + 0.5
        paths[m] = p
        row = nav[(nav.site == SITE_ID) & (nav.episode == EPISODE) & (nav.method == m)].iloc[0]
        outcome[m], length[m] = row.outcome, row.path_m
    risk = {m: z[f"{SITE_ID}|risk|{m}"] for m in SHOWN + ["Oracle"]}
    ends = z[f"{SITE_ID}|{EPISODE}|endpoints"]
    lc = pd.read_csv("results/lc_summary.csv")
    navs = pd.read_csv("results/nav_summary.csv")
    true_risk = norm.cdf((site.slip - SLIP_THRESHOLD) / 0.05)
    return dict(site=site, true_risk=true_risk, calib=calib, paths=paths, outcome=outcome, length=length, risk=risk, ends=ends, lc=lc, nav=navs)


# ---------------------------------------------------------------- helpers
def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def appear(t, start, dur=0.5):
    return ease((t - start) / dur)


def title(fig, text, t, y=0.9, start=0.0, size=40):
    a = appear(t, start)
    fig.text(0.06, y, text, fontsize=size, fontweight="semibold", color=INK, alpha=a)


def line(fig, text, t, y, start, x=0.06, size=26, color=INK2, weight="regular"):
    a = appear(t, start)
    fig.text(x, y - 0.01 * (1 - a), text, fontsize=size, color=color, alpha=a, fontweight=weight, va="top", linespacing=1.35)


def map_ax(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def path_upto(p, dist_m):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1) * CELL
    cum = np.concatenate([[0], np.cumsum(seg)])
    k = np.searchsorted(cum, dist_m, side="right")
    if k >= len(p):
        return p, True
    if k == 0:
        return p[:1], False
    f = (dist_m - cum[k - 1]) / max(seg[k - 1], 1e-9)
    tip = p[k - 1] + f * (p[k] - p[k - 1])
    return np.vstack([p[:k], tip]), False


def render(name, dur, draw, D, fade_in=0.4, fade_out=0.4):
    final = OUT / f"{name}.mp4"
    tmp = OUT / f"_{name}.tmp.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(tmp)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    n = int(round(dur * FPS))
    for i in range(n):
        t = i / FPS
        frame(fig, draw, D, t, dur, fade_in, fade_out)
        proc.stdin.write(fig.canvas.buffer_rgba())
    proc.stdin.close()
    proc.wait()
    plt.close(fig)
    if proc.returncode == 0:
        os.replace(tmp, final)
    else:
        raise RuntimeError(f"ffmpeg failed on {name}")
    return final


def frame(fig, draw, D, t, dur, fade_in=0.4, fade_out=0.4):
    fig.clf()
    fig.patch.set_facecolor(BG)
    draw(fig, t, D)
    a = max(1 - t / fade_in if fade_in else 0, (t - (dur - fade_out)) / fade_out if fade_out else 0, 0)
    if a > 0:
        fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, color=BG, alpha=min(a, 1), zorder=1000))
    fig.canvas.draw()


# ---------------------------------------------------------------- scenes
def s1_title(fig, t, D):
    fig.text(0.06, 0.58, "TabPFN for terrain-aware\nrobot navigation", fontsize=64, fontweight="semibold",
             alpha=appear(t, 0.2, 0.7), va="bottom", linespacing=1.15)
    line(fig, "A simulated study: a wheeled robot with a lidar learns each new site\nfrom a few hundred cells it labels itself", t, 0.53, 1.0, size=30)
    line(fig, "10 test sites for prediction  ·  64 test missions for navigation", t, 0.36, 1.8, size=24, color=MUTED)


def s2_problem(fig, t, D):
    site = D["site"]
    title(fig, "Lidar geometry alone misreads terrain", t)
    ax = map_ax(fig, [0.06, 0.1, 0.4, 0.71])
    ax.imshow(site.terrain, cmap=TERRAIN_CM, vmin=0, vmax=7, origin="lower", interpolation="nearest")
    b = appear(t, 5.5, 1.0)
    if b > 0:
        ax.imshow(np.where(site.terrain == OBSTACLE, np.nan, D["true_risk"]), cmap="magma", vmin=0, vmax=1,
                  origin="lower", alpha=b, interpolation="nearest")
    cap = "Simulated site, 60 m × 60 m" if b < 0.5 else "Where the wheels actually slip (bright = high slip)"
    fig.text(0.06, 0.065, cap, fontsize=20, color=INK2)
    for i, name in enumerate(TERRAIN_NAMES):
        y = 0.78 - i * 0.045
        a = (1 - b) * appear(t, 0.3)
        fig.add_artist(Rectangle((0.48, y - 0.012), 0.016, 0.026, transform=fig.transFigure, color=TERRAIN_CM(i), alpha=a))
        fig.text(0.502, y, name, fontsize=19, color=INK2, va="center", alpha=a)
    x = 0.64
    line(fig, "Tall grass looks like a wall,\nbut the robot can drive through it", t, 0.78, 1.2, x=x)
    line(fig, "Wet clay and dry sand look flat,\nbut the wheels slip and it gets stuck", t, 0.62, 2.6, x=x)
    line(fig, "And it changes from site to site:\nsoil, moisture, vegetation,\neach robot's sensor calibration", t, 0.46, 4.0, x=x)
    line(fig, "So a model has to be learned\nat every new site, from little data", t, 0.24, 7.0, x=x, color=INK, weight="medium")


def s3_calib(fig, t, D):
    site, cal = D["site"], D["calib"]
    title(fig, "Step 1: a short calibration drive labels 256 cells", t)
    ax = map_ax(fig, [0.06, 0.1, 0.4, 0.71])
    ax.imshow(site.terrain, cmap=TERRAIN_CM, vmin=0, vmax=7, origin="lower", interpolation="nearest")
    ax.add_patch(Rectangle((N / 2 - 0.5, -0.5), N / 2, N, color=BG, alpha=0.55))
    ax.text(N * 0.75, N / 2, "unseen half", ha="center", va="center", fontsize=22, color=INK2)
    prog = ease((t - 0.8) / 7.0)
    k = int(prog * len(cal["pts"]))
    if k > 0:
        p = cal["pts"][: k + 1]
        ax.plot(p[:, 1], p[:, 0], color="#ffffff", lw=1.2, alpha=0.8)
        done = cal["pts"][:k]
        cols = np.where(cal["y"][:k] == 1, RED, "#ffffff")
        ax.scatter(done[:, 1], done[:, 0], s=26, c=cols, edgecolors=INK, linewidths=0.6, zorder=3)
        ax.scatter([p[-1, 1]], [p[-1, 0]], s=160, c=BLUE, edgecolors="#ffffff", linewidths=2, zorder=4)
    ax.set_xlim(-0.5, N - 0.5); ax.set_ylim(-0.5, N - 0.5)
    x = 0.54
    line(fig, "The robot drives over cells in the half it knows", t, 0.78, 0.6, x=x)
    line(fig, "Each cell gets a label from its own wheel slip:", t, 0.68, 1.6, x=x)
    fig.add_artist(Rectangle((x, 0.585), 0.012, 0.022, transform=fig.transFigure, color="#ffffff", ec=INK, lw=0.8, alpha=appear(t, 2.2)))
    fig.text(x + 0.02, 0.596, "drivable", fontsize=24, color=INK2, va="center", alpha=appear(t, 2.2))
    fig.add_artist(Rectangle((x + 0.13, 0.585), 0.012, 0.022, transform=fig.transFigure, color=RED, alpha=appear(t, 2.2)))
    fig.text(x + 0.15, 0.596, "high slip", fontsize=24, color=INK2, va="center", alpha=appear(t, 2.2))
    line(fig, "No manual labelling, no training run", t, 0.5, 3.4, x=x)
    fig.text(x, 0.3, f"{k}", fontsize=96, fontweight="semibold", color=BLUE, alpha=appear(t, 0.8))
    fig.text(x, 0.25, "labelled cells", fontsize=24, color=INK2, alpha=appear(t, 0.8))


def risk_panel(ax, R, a=1.0):
    ax.imshow(R[:, N // 2:], cmap="magma", vmin=0, vmax=1, origin="lower", interpolation="nearest", alpha=a)
    ax.set_xlim(-0.5, N / 2 - 0.5); ax.set_ylim(-0.5, N - 0.5)


def s4_costmaps(fig, t, D):
    title(fig, "Step 2: each method predicts slip risk for the unseen half", t)
    names = ["Oracle"] + SHOWN
    heads = ["Truth"] + SHOWN
    subs = ["what really slips"] + SHOWN_SUB
    pw, gap, x0 = 0.15, 0.03, 0.06
    for i, m in enumerate(names):
        s = 0.8 + i * 1.3
        a = appear(t, s, 0.7)
        ax = map_ax(fig, [x0 + i * (pw + gap), 0.27, pw, 0.53])
        if a > 0:
            risk_panel(ax, D["risk"][m], a)
        col = BLUE if m == "TabPFN v2" else INK
        fig.text(x0 + i * (pw + gap), 0.83, heads[i], fontsize=24, fontweight="semibold", color=col, alpha=appear(t, s))
        fig.text(x0 + i * (pw + gap), 0.255, subs[i], fontsize=17, color=INK2, va="top", alpha=appear(t, s), linespacing=1.3)
    line(fig, "Bright = predicted high slip. The models that learned from this site's own cells find slippery patches\n"
              "that the geometric rule and the cross-site model get wrong.", t, 0.105, 7.8, size=21)


def s5_race(fig, t, D):
    title(fig, "Step 3: the same mission, planned on each costmap", t)
    pw, gap, x0 = 0.17, 0.05, 0.06
    longest = max(D["length"].values())
    dist = longest * ease((t - 1.0) / 11.0)
    ends = D["ends"]
    for i, m in enumerate(SHOWN):
        ax = map_ax(fig, [x0 + i * (pw + gap), 0.16, pw, 0.66])
        risk_panel(ax, D["risk"][m])
        p, finished = path_upto(D["paths"][m], dist)
        q = p.copy()
        q[:, 1] -= N // 2
        ax.plot(q[:, 1], q[:, 0], color="#ffffff", lw=6, solid_capstyle="round")
        ax.plot(q[:, 1], q[:, 0], color=GREEN, lw=3.2, solid_capstyle="round")
        ax.scatter(ends[:, 1] - N // 2, ends[:, 0], s=110, c="#ffffff", edgecolors=INK, linewidths=1.5, zorder=5)
        stuck = D["outcome"][m] == "stuck"
        if finished and stuck:
            ax.scatter([q[-1, 1]], [q[-1, 0]], marker="X", s=520, c=RED, edgecolors="#ffffff", linewidths=2, zorder=6)
        elif not finished:
            ax.scatter([q[-1, 1]], [q[-1, 0]], s=140, c=BLUE, edgecolors="#ffffff", linewidths=2, zorder=6)
        col = BLUE if m == "TabPFN v2" else INK
        fig.text(x0 + i * (pw + gap), 0.85, m, fontsize=24, fontweight="semibold", color=col, alpha=appear(t, 0.3))
        if finished:
            if stuck:
                msg, c = f"Stuck after {D['length'][m]:.0f} m", RED
            else:
                msg, c = f"Arrived, {D['length'][m]:.0f} m route", GREEN
            fig.text(x0 + i * (pw + gap), 0.115, msg, fontsize=22, fontweight="semibold", color=c)
    fig.text(0.06, 0.05, "One of the 64 test missions. Getting stuck is random but uses the same draws for every method.",
             fontsize=18, color=MUTED, alpha=appear(t, 1.0))


def s6_curves(fig, t, D):
    lc = D["lc"]
    title(fig, "Prediction quality as the robot labels more cells", t)
    ax = fig.add_axes([0.08, 0.17, 0.52, 0.62])
    ax.set_facecolor(BG)
    ax.set_xscale("log", base=2)
    ns = [32, 64, 128, 256, 512, 1024]
    ax.set_xticks(ns, [str(n) for n in ns], fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_xlim(28, 1170); ax.set_ylim(0.24, 0.62)
    ax.grid(axis="y", color="#e4e2dc"); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Labelled cells from this site", fontsize=20)
    ax.set_ylabel("Prediction error (log loss, lower is better)", fontsize=20)
    ref = lc[(lc.model == "Cross-site XGBoost")].logloss.iloc[0]
    a = appear(t, 0.6)
    ax.axhline(ref, color=MUTED, ls="--", lw=2, alpha=a)
    ax.text(1150, ref + 0.006, "Trained on other sites only", ha="right", fontsize=16, color=INK2, alpha=a)
    ax.text(1150, 0.605, "Geometric rule: 0.88 (off the chart)", ha="right", fontsize=16, color=INK2, alpha=a)

    def draw_series(name, start, dur=3.0, lw=3):
        s = lc[lc.model == name].sort_values("n")
        s = s[s.n > 0]
        prog = ease((t - start) / dur)
        if prog <= 0:
            return
        xs, ys = np.log2(s.n.values), s.logloss.values
        xe = xs[0] + prog * (xs[-1] - xs[0])
        keep = xs <= xe + 1e-9
        px, py = list(xs[keep]), list(ys[keep])
        if xe > px[-1] and keep.sum() < len(xs):
            j = keep.sum()
            f = (xe - xs[j - 1]) / (xs[j] - xs[j - 1])
            px.append(xe); py.append(ys[j - 1] + f * (ys[j] - ys[j - 1]))
        ax.plot(2 ** np.array(px), py, color=SERIES[name], lw=lw, solid_capstyle="round")
        ax.scatter(2 ** xs[keep], ys[keep], s=50, color=SERIES[name], edgecolors=BG, linewidths=1.5, zorder=3)

    for nm in ["XGBoost (tuned)", "Random forest", "CatBoost"]:
        draw_series(nm, 1.0, lw=2.5)
    draw_series("TabPFN v2", 1.0, lw=4)
    archive = appear(t, 7.0, 0.6)
    if archive > 0:
        draw_series("Cross-site + local XGBoost", 7.0, dur=2.5, lw=2.5)
        draw_series("TabPFN v2 + archive", 7.0, dur=2.5, lw=4)
    # legend
    items = [("TabPFN v2", "TabPFN v2", 1.0), ("CatBoost", "CatBoost", 1.0), ("XGBoost (tuned)", "XGBoost (tuned)", 1.0),
             ("Random forest", "Random forest", 1.0), ("Cross-site + local XGBoost", "XGBoost + 20,000 rows from other sites", 7.0),
             ("TabPFN v2 + archive", "TabPFN v2 + 512 rows from other sites", 7.0)]
    for i, (key, label, st) in enumerate(items):
        y = 0.77 - i * 0.045
        a = appear(t, st)
        fig.add_artist(Rectangle((0.64, y - 0.004), 0.025, 0.008, transform=fig.transFigure, color=SERIES[key], alpha=a))
        fig.text(0.672, y, label, fontsize=18, color=INK2, va="center", alpha=a)
    line(fig, "Using only this site's cells, TabPFN has\nthe lowest error from 64 labels on", t, 0.46, 4.0, x=0.64, size=24, color=INK)
    line(fig, "With a few rows from earlier sites in its\ncontext, it has the lowest average error\nat every size tested", t, 0.31, 9.5, x=0.64, size=24, color=INK)
    fig.text(0.08, 0.05, "Mean over 10 simulated test sites, 1,000 held-out cells each.", fontsize=17, color=MUTED, alpha=appear(t, 0.6))


def s7_bars(fig, t, D):
    nav = D["nav"]
    title(fig, "Result: missions completed without getting stuck", t)
    order = ["Oracle", "TabPFN v2", "CatBoost", "Cross-site + local XGBoost", "XGBoost (tuned)", "Cross-site XGBoost",
             "Geometric rule", "No terrain model"]
    label = {"Oracle": "Knows the true slip", "Cross-site + local XGBoost": "XGBoost + other sites",
             "Cross-site XGBoost": "Other sites only", "No terrain model": "Obstacles only"}
    x0, x1, y0, rowh = 0.34, 0.86, 0.76, 0.075
    g = ease((t - 0.8) / 2.5)
    for i, m in enumerate(order):
        v = float(nav[nav.method == m].success.iloc[0]) * 100
        y = y0 - i * rowh
        col = BLUE if m == "TabPFN v2" else ("#cfcdc6" if m != "Oracle" else "#e4e2dc")
        fig.text(x0 - 0.015, y, label.get(m, m), fontsize=22, ha="right", va="center",
                 color=BLUE if m == "TabPFN v2" else INK2, fontweight="semibold" if m == "TabPFN v2" else "regular",
                 alpha=appear(t, 0.4))
        wbar = (x1 - x0) * v / 100 * g
        fig.add_artist(FancyBboxPatch((x0, y - 0.022), max(wbar, 1e-4), 0.044, boxstyle="round,pad=0,rounding_size=0.006",
                                      transform=fig.transFigure, color=col, lw=0))
        if g > 0.05:
            fig.text(x0 + wbar + 0.01, y, f"{v * g:.0f}%", fontsize=22, va="center", color=INK,
                     fontweight="semibold" if m == "TabPFN v2" else "regular")
    fig.text(0.06, 0.08, "64 missions on 8 unseen simulated sites. Learned models had 256 cells from each site.",
             fontsize=18, color=MUTED, alpha=appear(t, 0.6))


def s8_takeaways(fig, t, D):
    title(fig, "Where TabPFN helps in this setup", t)
    pts = [
        ("Little data per site", "Best model trained only on this site's cells, from 64 labels on"),
        ("No training step", "A new site is just a new context: 4 to 13 s on a 2-core CPU"),
        ("Probabilities a planner can use", "81% of missions completed, against 75% for the next best model"),
        ("Keep in mind", "Simulated sites. An archive of earlier sites helps every method,\nand TabPFN can take a sample of it in its context"),
    ]
    for i, (h, d) in enumerate(pts):
        y = 0.74 - i * 0.16
        s = 0.6 + i * 1.2
        col = MUTED if h == "Keep in mind" else BLUE
        fig.add_artist(Rectangle((0.06, y - 0.075), 0.005, 0.085, transform=fig.transFigure, color=col, alpha=appear(t, s)))
        line(fig, h, t, y, s, x=0.08, size=30, color=INK, weight="semibold")
        line(fig, d, t, y - 0.045, s + 0.2, x=0.08, size=24)


SCENES = [
    ("scene_1_title", 5.0, s1_title),
    ("scene_2_problem", 11.0, s2_problem),
    ("scene_3_calibration", 10.0, s3_calib),
    ("scene_4_costmaps", 11.0, s4_costmaps),
    ("scene_5_mission", 15.0, s5_race),
    ("scene_6_curves", 14.0, s6_curves),
    ("scene_7_missions", 8.0, s7_bars),
    ("scene_8_takeaways", 9.0, s8_takeaways),
]


def stills(D, picks):
    (OUT / "stills").mkdir(exist_ok=True)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    lookup = {n: (d, f) for n, d, f in SCENES}
    for name, t in picks:
        dur, fn = lookup[name]
        frame(fig, fn, D, t, dur)
        fig.savefig(OUT / "stills" / f"{name}_{t:.1f}.png", dpi=100, facecolor=BG)
    plt.close(fig)


def main():
    D = load()
    if "--stills" in sys.argv:
        stills(D, [(n, d - 0.6) for n, d, _ in SCENES] + [("scene_5_mission", 8.0), ("scene_6_curves", 6.5)])
        return
    only = [a for a in sys.argv[1:] if a.startswith("scene_")]
    files = []
    for name, dur, fn in SCENES:
        if not only or name in only:
            render(name, dur, fn, D)
            print("rendered", name, flush=True)
        files.append(OUT / f"{name}.mp4")
    lst = OUT / "concat.txt"
    lst.write_text("".join(f"file '{f.name}'\n" for f in files))
    tmp = OUT / "_full.tmp.mp4"
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                        "-movflags", "+faststart", str(tmp)])
    if r.returncode == 0:
        os.replace(tmp, OUT / "tabpfn_terrain_navigation.mp4")
        print("done", OUT / "tabpfn_terrain_navigation.mp4")


if __name__ == "__main__":
    main()
