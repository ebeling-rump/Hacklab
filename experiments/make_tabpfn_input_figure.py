"""Explainer figure: what one TabPFN row looks like and how the table is used each planning step.

    python experiments/make_tabpfn_input_figure.py   -> results/figures/tabpfn_input_{light,dark}.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# (label, columns, colour, detail)
SEGMENTS = [
    ("contact", 4, "#3B8ED8", "where on the edge\n(point, normal)"),
    ("push", 5, "#F08A24", "direction, angle,\ndistance, speed"),
    ("lever", 1, "#7B61C9", "torque\narm"),
    ("shape", 6, "#8E9AAF", "object descriptor\n(transfer only)"),
    ("ice", 3, "#7FC8E8", "visible ice under\nobject (C8 only)"),
]
TARGET = ("target", 3, "#1A9E8F", "what physics got wrong\nΔx, Δy, Δθ residual")


def draw(theme):
    dark = theme == "dark"
    bg, ink, sub, grid = ("#0F1B2D", "#F5F2EB", "#C3CEDD", "#24364F") if dark else \
                         ("#F5F2EB", "#14213D", "#46536A", "#D9D3C6")
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 192); ax.set_ylim(0, 108); ax.axis("off")
    ax.text(8, 100, "HOW TABPFN IS USED", color="#F08A24", fontsize=20, fontweight="bold")
    ax.text(8, 93.5, "Every push becomes one row. TabPFN reads the whole table in one pass.",
            color=ink, fontsize=27, fontweight="bold")

    cw = 4.2            # column width
    x0, ytop, rh = 24, 74, 4.2
    ncol = sum(s[1] for s in SEGMENTS)
    xt = x0 + ncol * cw + 10           # target block x
    # segment headers
    x = x0
    for lab, n, c, det in SEGMENTS + [TARGET]:
        if lab == "target":
            x = xt
        ax.add_patch(FancyBboxPatch((x, ytop + 3.2), n * cw, 4.2, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc=c, ec="none"))
        ax.text(x + n * cw / 2, ytop + 5.3, lab, ha="center", va="center", fontsize=17 if n > 1 else 12, fontweight="bold",
                color="#0F1B2D")
        ax.text(x + n * cw / 2, ytop + 10.6, det, ha="center", va="bottom", fontsize=13, color=sub)
        x += n * cw
    # rows: context (own pushes), context (other objects), query
    rows = [("push 1", "own"), ("push 2", "own"), ("push 3", "own"), ("…", "gap"),
            ("other object", "prior"), ("other object", "prior"), ("…", "gap"),
            ("candidate 1", "query"), ("candidate 2", "query"), ("…", "gap"), ("candidate 96", "query")]
    y = ytop
    for name, kind in rows:
        y -= rh + (1.2 if kind == "query" and name == "candidate 1" else 0.35)
        if name == "candidate 1":
            ax.plot([x0 - 1, xt + TARGET[1] * cw + 1], [y + rh + 0.75, y + rh + 0.75], color=sub, lw=1.2, ls="--")
        if kind == "gap":
            ax.text(x0 + ncol * cw / 2, y + rh / 2, "⋮", ha="center", va="center", fontsize=20, color=sub)
            continue
        alpha = 0.55 if kind == "prior" else 1.0
        x = x0
        for lab, n, c, _ in SEGMENTS:
            for j in range(n):
                fc = c if not (lab == "shape" and kind == "own" and False) else grid
                ax.add_patch(Rectangle((x + j * cw + 0.15, y), cw - 0.3, rh, fc=fc, ec="none", alpha=alpha))
            x += n * cw
        for j in range(TARGET[1]):
            if kind == "query":
                ax.add_patch(Rectangle((xt + j * cw + 0.15, y), cw - 0.3, rh, fc="none", ec=TARGET[2], lw=2))
                ax.text(xt + j * cw + cw / 2, y + rh / 2, "?", ha="center", va="center", fontsize=16,
                        color=TARGET[2], fontweight="bold")
            else:
                ax.add_patch(Rectangle((xt + j * cw + 0.15, y), cw - 0.3, rh, fc=TARGET[2], ec="none", alpha=alpha))
        ax.text(x0 - 1.2, y + rh / 2, name, ha="right", va="center", fontsize=13, color=sub)
    # brackets / annotations on the right
    ax_r = xt + TARGET[1] * cw + 4
    ax.text(ax_r, 66.5, "Context: this object's pushes so far\n(1 to 20 rows; grows after every push)",
            fontsize=15, color=ink, va="center")
    ax.text(ax_r, 55.5, "+ optional experience: 300 rows\nfrom 40 other objects", fontsize=15, color=ink, va="center")
    ax.text(ax_r, 34, "Query: 96 candidate pushes\nfor the current pose", fontsize=15, color=ink, va="center")
    # bottom pipeline
    yb = 6
    steps = [("Physics model", "predicts the push\n(friction, mass fitted online)", "#F08A24"),
             ("TabPFN", "one forward pass, no training:\nresidual + uncertainty per candidate", "#3B8ED8"),
             ("Planner", "physics + residual, 2 pushes ahead,\npenalise uncertainty → best push", "#1A9E8F"),
             ("Robot", "pushes, observes the outcome,\nadds one row to the table", "#7B61C9")]
    bw, gap = 40, 5.3
    for i, (t, d, c) in enumerate(steps):
        bx = 8 + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((bx, yb), bw, 12, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=bg, ec=c, lw=3))
        ax.text(bx + 2, yb + 8.6, t, fontsize=17, fontweight="bold", color=ink, va="center")
        ax.text(bx + 2, yb + 4.0, d, fontsize=12.5, color=sub, va="center")
        if i < len(steps) - 1:
            ax.annotate("", xy=(bx + bw + gap - 0.6, yb + 6), xytext=(bx + bw + 0.6, yb + 6),
                        arrowprops=dict(arrowstyle="-|>", color=sub, lw=2))
    out = ROOT / "results" / "figures" / f"tabpfn_input_{theme}.png"
    fig.savefig(out, facecolor=bg)
    plt.close(fig)
    print("->", out)


if __name__ == "__main__":
    draw("light")
    draw("dark")
