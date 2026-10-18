"""Generate the method diagram used in the first post (1080x1350 px).

The diagram is purely schematic: feature names are placeholders, no results are shown.
Run:  python assets/make_method_diagram.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).with_name("method-diagram.png")

BG = "#FAFAF7"
INK = "#1F2937"
MUTED = "#6B7280"
C_TRAIN = "#2563EB"   # blue
C_EXPLAIN = "#7C3AED"  # violet
C_COMPARE = "#D97706"  # amber

fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 108)
ax.set_ylim(0, 135)
ax.axis("off")


def box(x, y, w, h, color, alpha=0.10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.2",
                                linewidth=2.2, edgecolor=color, facecolor=color, alpha=alpha))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.2",
                                linewidth=2.2, edgecolor=color, facecolor="none"))


def arrow(x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=34,
                                 linewidth=3, color=MUTED))


# Title
ax.text(54, 124, "Same model.", ha="center", va="center", fontsize=44, fontweight="bold", color=INK)
ax.text(54, 115.5, "Different training data.", ha="center", va="center", fontsize=44, fontweight="bold", color=INK)
ax.text(54, 107, "Same explanation?", ha="center", va="center", fontsize=44, fontweight="bold", color=C_COMPARE)

# Step 1: retrain on different samples
box(8, 76, 92, 24, C_TRAIN)
ax.text(14, 95, "1", fontsize=30, fontweight="bold", color=C_TRAIN, va="center")
ax.text(22, 95, "Retrain on different samples", fontsize=27, fontweight="bold", color=INK, va="center")
for i, (dx, dy) in enumerate([(0, 0), (3.2, -2.2), (6.4, -4.4)]):
    ax.add_patch(FancyBboxPatch((18 + dx, 83 + dy), 20, 7, boxstyle="round,pad=0,rounding_size=1",
                                linewidth=2, edgecolor=C_TRAIN, facecolor="white"))
ax.text(52, 85, "bootstrap samples,\nvarious sizes,\nmany repeats", fontsize=19, color=MUTED, va="center")

arrow(54, 75, 67)

# Step 2: explain the same customers
box(8, 43, 92, 24, C_EXPLAIN)
ax.text(14, 62, "2", fontsize=30, fontweight="bold", color=C_EXPLAIN, va="center")
ax.text(22, 62, "Explain the same customers", fontsize=27, fontweight="bold", color=INK, va="center")
ax.text(22, 54, "SHAP values every time, so only the\ntraining data changes", fontsize=20, color=MUTED, va="center")
ax.text(22, 47, "logistic regression  vs  random forest", fontsize=20, color=C_EXPLAIN, va="center", fontweight="bold")

arrow(54, 42, 34)

# Step 3: compare rankings
box(8, 5.5, 92, 28.5, C_COMPARE)
ax.text(14, 29, "3", fontsize=30, fontweight="bold", color=C_COMPARE, va="center")
ax.text(22, 29, "Does the top-5 move?", fontsize=27, fontweight="bold", color=INK, va="center")

run1 = ["feature A", "feature B", "feature C", "feature D", "feature E"]
run2 = ["feature A", "feature C", "feature B", "feature E", "feature D"]
for col_x, title, ranks in [(16, "run 1", run1), (58, "run 2", run2)]:
    ax.text(col_x + 14, 23, title, fontsize=19, color=MUTED, ha="center", va="center", fontweight="bold")
    for i, name in enumerate(ranks):
        y = 19.2 - i * 2.75
        moved = (title == "run 2" and name != run1[i])
        ax.add_patch(FancyBboxPatch((col_x, y - 1.15), 28, 2.3, boxstyle="round,pad=0,rounding_size=0.8",
                                    linewidth=1.6, edgecolor=C_COMPARE if moved else "#D1D5DB",
                                    facecolor="#FEF3C7" if moved else "white"))
        ax.text(col_x + 2, y, f"{i + 1}.  {name}", fontsize=16, color=INK, va="center")

fig.savefig(OUT, dpi=100, facecolor=BG)
print(f"saved {OUT}")
