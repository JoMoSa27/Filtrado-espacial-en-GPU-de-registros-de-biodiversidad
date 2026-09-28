#!/usr/bin/env python3
"""
fig_thinning.py — Figura del paper: retención y rondas paralelas del filtrado.

Lee data/processed/{tortuga_gt,atun_gt}_por_especie.csv (salida de 04) y
escribe paper/fig_thinning.png. Panel (a): registros retenidos por cada regla
voraz, relativos a la regla de GeoThinneR (mejor de 10 intentos). Panel (b):
rondas paralelas hasta decidir todos los registros (máximo en 10 semillas).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
OUT = ROOT / "paper" / "fig_thinning.png"

# Paleta categórica de referencia (slots 1–3, validados en todos los pares).
C_RANDOM, C_MINDEG, INK, MUTED = "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
SETS = {"tortuga_gt": ("Loggerhead turtle (N = 8,340)", "o", "-"),
        "atun_gt": ("Yellowfin tuna (N = 80,163)", "s", "--")}

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 3.0))
fig.subplots_adjust(left=0.11, right=0.99, top=0.9, bottom=0.33, wspace=0.28)
for lab, (name, mk, ls) in SETS.items():
    d = pd.read_csv(P / f"{lab}_por_especie.csv").sort_values("d_km")
    for col, color, rule in (("ret_random_mejor", C_RANDOM, "random order"),
                             ("ret_mindeg_mejor", C_MINDEG, "min-degree order")):
        rel = 100 * (d[col] / d.ret_maxdeg_mejor - 1)
        a.plot(d.d_km, rel, color=color, marker=mk, ls=ls, lw=2, ms=6,
               label=f"{rule}, {name.split(' (')[0].lower()}")
    b.plot(d.d_km, d.rondas_random_max, color=C_RANDOM, marker=mk, ls=ls, lw=2, ms=6)
    b.plot(d.d_km, d.rondas_mindeg_max, color=C_MINDEG, marker=mk, ls=ls, lw=2, ms=6)
a.axhline(0, color=MUTED, lw=1)
a.text(50, 0.4, "GeoThinneR rule (best of 10)", ha="right", va="bottom", color=MUTED, fontsize=8)
a.set_xlabel("Thinning distance (km)")
a.set_ylabel("Difference vs. GeoThinneR (%)")
a.set_title("(a) Retained records", loc="left", fontsize=9, color=INK)
b.set_xlabel("Thinning distance (km)")
b.set_ylabel("Parallel rounds (max of 10 seeds)")
b.set_title("(b) Rounds to decide every record", loc="left", fontsize=9, color=INK)
b.set_ylim(0, None)
for ax in (a, b):
    ax.set_xticks([10, 25, 50])
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
h, l = a.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7.5,
           bbox_to_anchor=(0.5, 0.0))
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=300)
print(f"-> {OUT.relative_to(ROOT)}")
