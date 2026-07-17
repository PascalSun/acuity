"""Four-panel 3D difficulty surfaces: join pattern x exact predicate count -> six-model mean EX.
Spider / BIRD (academic) + WAMEX / ACYWA (private, controlled protocol).
Data: docs/papers/FlexBench/results/surface_pattern_x_npred{,_bird,_wamex,_acywa}.json
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.interpolate import RectBivariateSpline

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
})

INK = "#282C34"
GRAY = "#6E747D"
RED = "#B3261E"
KS = list(range(1, 8))
ramp = LinearSegmentedColormap.from_list(
    "acc", ["#B3261E", "#E8A79D", "#F0EFED", "#9DC3E0", "#2E6FAC"])
norm = Normalize(vmin=0.30, vmax=1.00)
FLOOR = 0.0

def load_grid(path, patterns):
    data = json.load(open(path))
    P = len(patterns)
    Z = np.full((7, P), np.nan); N = np.zeros((7, P))
    for i, k in enumerate(KS):
        for j, p in enumerate(patterns):
            cell = data.get(f"{p}|{k}")
            if cell: Z[i, j] = cell["acc"]; N[i, j] = cell["n"]
    for i in range(7):
        for j in range(P):
            if np.isnan(Z[i, j]):
                vals, ws = [], []
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        ii, jj = i + di, j + dj
                        if 0 <= ii < 7 and 0 <= jj < P and not np.isnan(Z[ii, jj]):
                            vals.append(Z[ii, jj]); ws.append(max(N[ii, jj], 1))
                Z[i, j] = np.average(vals, weights=ws)
    return Z, N

def shrink(Z, N, n0, solid_min=15):
    # solid cells (n >= solid_min) anchor the sheet exactly; only thin cells
    # are pulled toward the N-weighted 3x3 local mean
    P = Z.shape[1]
    K = np.array([[0.5, 1.0, 0.5], [1.0, 2.0, 1.0], [0.5, 1.0, 0.5]])
    Zloc = np.zeros_like(Z)
    for i in range(7):
        for j in range(P):
            num = den = 0.0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < 7 and 0 <= jj < P:
                        w = K[di + 1, dj + 1] * max(N[ii, jj], 1)
                        num += w * Z[ii, jj]; den += w
            Zloc[i, j] = num / den
    Zs = (N * Z + n0 * Zloc) / (N + n0)
    Zs[N >= solid_min] = Z[N >= solid_min]
    return Zs

def fmt(v):
    return "1.0" if v >= 0.995 else f".{round(v*100):02d}"

def panel(ax, jsonpath, patterns, title, labels, note=None, n0=30.0, zlab=True, kdeg=3):
    P = len(patterns)
    Z, N = load_grid(jsonpath, patterns)
    X, Y = np.meshgrid(np.arange(P), np.arange(7))
    Zs = shrink(Z, N, n0)
    spl = RectBivariateSpline(np.arange(7), np.arange(P), Zs,
                              kx=min(kdeg, 6), ky=min(kdeg, P - 1), s=0)
    fx = np.linspace(0, P - 1, 31 * P); fy = np.linspace(0, 6, 181)
    FX, FY = np.meshgrid(fx, fy)
    FZ = np.clip(spl(fy, fx), FLOOR, 1.0)

    ax.contourf(FX, FY, FZ, zdir="z", offset=FLOOR,
                levels=np.linspace(FLOOR, 1.0, 41), cmap=ramp, norm=norm,
                alpha=0.42, antialiased=True, zorder=0, extend="neither")
    ax.plot_surface(FX, FY, FZ, facecolors=ramp(norm(FZ)),
                    rstride=2, cstride=2, linewidth=0, antialiased=True,
                    shade=False, alpha=0.96, zorder=2)
    ax.contour(FX, FY, FZ, levels=[0.2, 0.4, 0.6, 0.8],
               colors=[INK], linewidths=0.35, alpha=0.4, zorder=3)
    ax.plot(np.zeros_like(fy), fy, np.clip(spl(fy, 0).ravel(), FLOOR, 1), color=INK, lw=0.8, alpha=0.85)
    ax.plot(fx, np.zeros_like(fx), np.clip(spl(0, fx).ravel(), FLOOR, 1), color=INK, lw=0.8, alpha=0.85)
    ax.plot(np.full_like(fy, P - 1), fy, np.clip(spl(fy, P - 1).ravel(), FLOOR, 1), color=INK, lw=0.55, alpha=0.6)
    ax.plot(fx, np.full_like(fx, 6), np.clip(spl(6, fx).ravel(), FLOOR, 1), color=INK, lw=0.55, alpha=0.6)

    solid = N >= 15
    ax.scatter(X[solid], Y[solid], Z[solid] + 0.004, s=5, c=INK,
               edgecolors="white", linewidths=0.3, depthshade=False)
    # thin cells: plotted at the sheet's smoothed estimate, hollow = low-n flag
    ax.scatter(X[~solid], Y[~solid], Zs[~solid] + 0.004, s=6, facecolors="white",
               edgecolors=INK, linewidths=0.45, depthshade=False)

    for (i, j, dx, dy, dz, color, bold) in labels:
        ax.text(j + dx, i + dy, Z[i, j] + dz, fmt(Z[i, j]),
                fontsize=8.6 if bold else 7.2, color=color,
                fontweight="bold" if bold else "normal", ha="center")

    ax.set_xticks(np.arange(P)); ax.set_xticklabels(patterns, fontsize=7.8, color=INK)
    ax.set_yticks(np.arange(7)); ax.set_yticklabels(["1","2","3","4","5","6","7+"], fontsize=7.6, color=INK)
    ax.set_zlim(FLOOR, 1.0)
    ax.set_zticks([0, 0.5, 1.0]); ax.set_zticklabels(["0", ".5", "1"], fontsize=7.4, color=GRAY)
    ax.set_xlabel("join pattern", fontsize=8.2, color=INK, labelpad=-6)
    ax.set_ylabel("filter predicates", fontsize=8.2, color=INK, labelpad=-5)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("execution accuracy" if zlab else "", fontsize=8.2, color=INK, rotation=90, labelpad=-7)
    ax.tick_params(pad=-3)
    ax.view_init(elev=22, azim=-59)
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((0.97, 0.975, 0.98, 1))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update(color="#D8DCE1", linewidth=0.35)
    ax.set_box_aspect((1.45, 1.15, 0.72), zoom=1.16)
    ax.set_title(title, fontsize=9.0, color=INK, y=0.97, fontweight="bold")

R = "docs/papers/FlexBench/results/"
fig = plt.figure(figsize=(7.4, 4.9))
rects = [(-0.03, 0.49, 0.56, 0.51), (0.47, 0.49, 0.56, 0.51),
         (-0.03, -0.005, 0.56, 0.51), (0.47, -0.005, 0.56, 0.51)]
axs = [fig.add_axes(rc, projection="3d") for rc in rects]

panel(axs[0], R + "surface_pattern_x_npred.json", ["0","1p","2p","3p","2i","3i","4i"],
      "Spider — 158 databases, n=6,461",
      [(0,0,0.35,-0.5,0.10,INK,True),(6,6,0.0,0.7,0.09,RED,True)], n0=40, zlab=True)
panel(axs[1], R + "surface_pattern_x_npred_bird.json", ["0","1p","2p","3p","2i","3i","4i"],
      "BIRD — 72 databases, n=2,842",
      [(0,0,0.35,-0.5,0.10,INK,True),(6,2,-0.3,0.8,0.12,RED,True)], n0=40, zlab=False)
panel(axs[2], R + "surface_pattern_x_npred_wamex.json", ["0","1p","2i","3i","4i"],
      "WAMEX (private, Star) — n=488;  2p/3p infeasible",
      [(0,0,0.35,-0.5,0.10,INK,True),(6,3,0.3,0.75,0.14,RED,True)],
      n0=40, zlab=True, kdeg=3)
panel(axs[3], R + "surface_pattern_x_npred_acywa.json", ["0","1p","2p","3p","2i","3i"],
      "ACYWA (private, Snowflake) — n=482;  4i infeasible",
      [(0,0,0.35,-0.5,0.10,INK,True),(6,5,0.45,0.35,0.12,RED,True)],
      n0=40, zlab=False, kdeg=3)

fig.savefig("fig_surface_3d.pdf")
print("saved four-panel")
