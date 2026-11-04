#!/usr/bin/env python3
"""
fig_c4.py — Figura de la elección de la distancia de filtrado (C4).

(a) Compromiso: registros conservados contra concentración (fracción de los
    registros en el 1 % de celdas de 5 km más muestreadas), un punto por d.
(b) La distancia que minimiza la concentración, medida con mallas de 2, 5 y
    10 km: sigue al lado de la malla, así que no hay un óptimo universal.

Insectos van fuera de (a): su concentración (64–94 %) aplastaría la escala, y
el texto ya explica por qué filtrar por especie no la reduce.

Lee bench/results/c4_distancias_celda{2,5,10}km.csv (10_c4_distancias.py).
Escribe paper/fig_c4_tradeoff.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "bench" / "results"
COL = {"Birds": "#2a78d6", "Amphibians": "#eb6834", "Plants": "#1baf7a"}   # paleta validada
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.labelcolor": INK})
fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.9), gridspec_kw={"width_ratios": [1.6, 1]})
fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.17, wspace=0.32)

d5 = pd.read_csv(RES / "c4_distancias_celda5km.csv")
for t, c in COL.items():
    g = d5[d5.taxon == t].sort_values("d_km")
    x = [100] + list(100 * g.fraccion_retenida)
    y = [100 * g.antes_frac_1pct.iloc[0]] + list(100 * g.despues_frac_1pct)
    a.plot(x, y, color=c, lw=1.8, marker="o", ms=4, zorder=3)
    dy = {"Birds": 8, "Amphibians": -12, "Plants": 7}[t]
    a.annotate(t, (x[-1], y[-1]), xytext=(0, dy), textcoords="offset points", ha="center",
               fontsize=8, color=c, fontweight="bold")
    if t == "Birds":
        for xi, yi, k in zip(x[1:], y[1:], g.d_km):
            a.annotate(f"{k:g} km", (xi, yi), xytext=(-6, -12) if k >= 50 else (-4, 5) if k >= 25 else (4, -9) if k < 5 else (6, -3),
                       textcoords="offset points", ha="right" if k >= 25 else "left", fontsize=7, color=MUTED)
a.annotate("no thinning", (100, 100 * d5[d5.taxon == "Birds"].antes_frac_1pct.iloc[0]),
           xytext=(5, 2), textcoords="offset points", fontsize=7, color=MUTED)
a.set_xscale("log")
a.invert_xaxis()
a.set_xticks([100, 50, 20, 10, 5, 2], ["100", "50", "20", "10", "5", "2"])
a.set_xlabel("Records kept (%)")
a.set_ylabel("Records in top 1 % of 5 km cells (%)")
a.set_title("(a) Retention vs. concentration", loc="left", fontsize=9, color=INK)
a.grid(color=GRID, lw=0.6)

filas = []
for c in (2, 5, 10):
    d = pd.read_csv(RES / f"c4_distancias_celda{c}km.csv")
    for t in COL:
        g = d[d.taxon == t]
        filas.append({"malla": c, "taxon": t, "d_min": g.loc[g.despues_frac_1pct.idxmin(), "d_km"]})
m = pd.DataFrame(filas)
off = {"Birds": -0.06, "Amphibians": 0.0, "Plants": 0.06}
for t, c in COL.items():
    g = m[m.taxon == t]
    b.scatter(g.malla * (1 + off[t]), g.d_min, color=c, s=22, zorder=3, label=t)
b.plot([1.5, 13], [1.5, 13], color=MUTED, lw=0.8, ls="--")
b.text(11, 8.2, "d = grid side", fontsize=7, color=MUTED, ha="right")
b.set_xscale("log"); b.set_yscale("log")
b.set_xticks([2, 5, 10], ["2", "5", "10"]); b.set_yticks([1, 2, 5, 10, 25, 50], ["1", "2", "5", "10", "25", "50"])
b.set_xlim(1.5, 14); b.set_ylim(0.8, 60)
b.set_xlabel("Grid side used to measure (km)")
b.set_ylabel("d with least concentration (km)")
b.set_title("(b) Minimum follows the grid", loc="left", fontsize=9, color=INK)
b.minorticks_off()
b.grid(color=GRID, lw=0.6)
b.legend(frameon=False, fontsize=7, loc="lower right")
out = ROOT / "paper" / "fig_c4_tradeoff.png"
fig.savefig(out, dpi=300)
print(f"-> {out.relative_to(ROOT)}")
print(m.pivot(index="taxon", columns="malla", values="d_min").to_string())
