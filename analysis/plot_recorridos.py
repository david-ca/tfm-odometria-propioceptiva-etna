"""
plot_recorridos.py -- Figura de los dos recorridos (vista en planta y perfil de altura) a partir del
ground truth DGPS, para el apartado 5.1.2.1 (nota [80] del tutor).
Uso:  python plot_recorridos.py      -> figuras\fig_recorridos.png
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(r"C:\Users\Usuario\Desktop\TFM\extracted")
OUT = Path(r"C:\Users\Usuario\Desktop\TFM\figuras"); OUT.mkdir(exist_ok=True)
COL = {"Run_A": "#2a78d6", "Run_B": "#eb6834"}; NAME = {"Run_A": "Recorrido A", "Run_B": "Recorrido B"}


def load(p):
    with open(p) as f:
        r = csv.reader(f); next(r)
        return np.array([[float(x) for x in row] for row in r])


fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})
for run in ["Run_A", "Run_B"]:
    g = load(BASE / run / "ground_truth.csv")
    t, x, y, z = g[:, 0], g[:, 1], g[:, 2], g[:, 3]
    gaps = np.where(np.diff(t) > 2.0)[0] + 1          # no unir los huecos con una linea
    for seg in np.split(np.arange(len(t)), gaps):
        axs[0].plot(x[seg], y[seg], color=COL[run], lw=1.4)
        axs[1].plot((t[seg] - t[0]) / 60, z[seg], color=COL[run], lw=1.4)
    axs[0].plot(x[0], y[0], "o", color=COL[run], ms=5)
    axs[0].plot([], [], color=COL[run], lw=2, label=NAME[run])
axs[0].set_aspect("equal", adjustable="datalim"); axs[0].set_xlabel("x (m)"); axs[0].set_ylabel("y (m)")
axs[0].set_title("Vista en planta (punto: inicio)", fontsize=9, loc="left"); axs[0].legend(fontsize=8, frameon=False)
axs[1].set_xlabel("Tiempo desde el inicio (min)"); axs[1].set_ylabel("z (m)")
axs[1].set_title("Perfil de altura", fontsize=9, loc="left")
for ax in axs:
    ax.grid(color="#e4e2dc", lw=0.6)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "fig_recorridos.png", dpi=220, bbox_inches="tight")
print("Guardado", OUT / "fig_recorridos.png")
