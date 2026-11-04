#!/usr/bin/env python3
"""
fig_c6.py — Figura de la especie de gran rango (C6): Turdus grayi antes y
después del filtrado a 10 km, en todo su rango.

Lee data/interim/turdus_grayi.parquet (02_prepare.py: registros únicos) y la
máscara EXACTA del filtrado (05_references.py, regla mindeg, semilla 0, en las
mismas coordenadas float32 que usa la GPU). Ambos paneles con la misma escala
de color: registros por celda hexagonal.

Escribe paper/fig_c6_turdus.png. Con --poster escribe además la versión del
póster al ancho de su espacio (16,4 cm), con letra a escala del póster, en SVG
(vectorial: no pierde calidad al cambiar de tamaño en Canva) y en PNG a 600 dpi:
paper/poster/fig4_turdus.svg y .png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

ROOT = Path(__file__).resolve().parents[1]
m = pd.read_parquet(ROOT / "data" / "interim" / "turdus_grayi.parquet")
mask = np.load(ROOT / "data" / "processed" / "turdus_grayi_thin_mindeg_10km_s0_f32c.npy")
rep = json.loads((ROOT / "data" / "processed" / "turdus_grayi_report.json").read_text())
if len(mask) != len(m):
    raise SystemExit("la máscara no coincide con los registros")

ap = argparse.ArgumentParser()
ap.add_argument("--poster", action="store_true", help="versión del póster (SVG y PNG 600 dpi)")
args = ap.parse_args()

lon, lat = m.lon.values, m.lat.values
ext = (np.percentile(lon, 0.1) - 1, np.percentile(lon, 99.9) + 1,
       np.percentile(lat, 0.1) - 1, np.percentile(lat, 99.9) + 1)
azules = LinearSegmentedColormap.from_list("azules", plt.get_cmap("Blues")(np.linspace(0.25, 1.0, 256)))
kw = dict(gridsize=110, extent=ext, mincnt=1, cmap=azules, linewidths=0)
CM = 1 / 2.54
if args.poster:
    # 16,4 cm de ancho como su espacio en el póster; letra de 14 pt (el pie del póster es de 20 pt)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 14})
    fig, axes = plt.subplots(1, 2, figsize=(16.4 * CM, 16.4 * CM * 3.0 / 6.4), sharey=True)
else:
    plt.rcParams.update({"font.family": "serif", "font.size": 8.5})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
if args.poster:
    fig.subplots_adjust(left=0.01, right=0.82, top=0.88, bottom=0.03, wspace=0.04)
else:
    fig.subplots_adjust(left=0.02, right=0.88, top=0.9, bottom=0.04, wspace=0.04)
hb0 = axes[0].hexbin(lon, lat, **kw)
vmax = hb0.get_array().max()
for h in axes[0].collections:
    h.set_norm(LogNorm(vmin=1, vmax=vmax))
hb1 = axes[1].hexbin(lon[mask], lat[mask], norm=LogNorm(vmin=1, vmax=vmax), **kw)
if args.poster:
    paneles = (f"(a) {len(m):,} records", f"(b) {int(mask.sum()):,} kept, 10 km")
else:
    paneles = (f"(a) Unique records: {len(m):,}", f"(b) After thinning at 10 km: {int(mask.sum()):,}")
for ax, t in zip(axes, paneles):
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect(1 / np.cos(np.radians(np.mean(ext[2:]))))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(t, loc="left", fontsize=None if args.poster else 8.5)
cax = fig.add_axes([0.84, 0.12, 0.02, 0.7] if args.poster else [0.9, 0.12, 0.02, 0.7])
cb = fig.colorbar(hb1, cax=cax)
cb.set_label("Records per cell", fontsize=12 if args.poster else 8)
cb.ax.tick_params(labelsize=11 if args.poster else 7)
if args.poster:
    d = ROOT / "paper" / "poster"
    d.mkdir(exist_ok=True)
    out = d / "fig4_turdus.svg"
    fig.savefig(out)
    fig.savefig(d / "fig4_turdus.png", dpi=600)
else:
    out = ROOT / "paper" / "fig_c6_turdus.png"
    fig.savefig(out, dpi=300)
print(f"-> {out.relative_to(ROOT)}  ({rep['n_raw']:,} descargados, {len(m):,} únicos, "
      f"{int(mask.sum()):,} tras 10 km; lon {ext[0]:.1f}..{ext[1]:.1f}, lat {ext[2]:.1f}..{ext[3]:.1f})")
