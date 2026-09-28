#!/usr/bin/env python3
"""
fig_mapa.py — Mapa de Costa Rica antes y después del filtrado espacial.

Panel (a): todos los registros del taxón, incluidas las presencias repetidas
(lo que se descarga de GBIF). Panel (b): los registros que conserva el filtrado
por especie a la distancia d (máscara exacta de 05_references.py). Ambos
paneles usan la misma escala de color: registros por celda hexagonal.

Además imprime y guarda la concentración del muestreo antes y después, medida
sobre una malla de igual área de lado d: qué fracción de los registros cae en
el 1 % de celdas más muestreadas y cuántas celdas reúnen la mitad de los
registros.

Uso (desde la raíz del repo):
    python3 scripts/fig_mapa.py --label aves_2018 --d-km 5 --regla mindeg
Salidas: paper/fig_mapa_<label>_<d>km.png y data/processed/<label>_concentracion_<d>km.json
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

import geo

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"
OUTLINE = Path(__file__).resolve().parent / "cr_outline.geojson"


def latlon(label):
    pq = INTERIM / f"{label}.parquet"
    m = pd.read_parquet(pq) if pq.exists() else pd.read_csv(pq.with_suffix(".csv.gz"))
    return m.lat.values, m.lon.values


def concentracion(lat, lon, cell_m):
    """Reparto de los registros en una malla de igual área de lado cell_m."""
    x, y = geo.laea(lat, lon, 9.7, -84.2)
    ix = np.floor((x - x.min()) / cell_m).astype(np.int64)
    iy = np.floor((y - y.min()) / cell_m).astype(np.int64)
    _, c = np.unique(ix * (iy.max() + 2) + iy, return_counts=True)
    c = np.sort(c)[::-1]
    top = max(1, int(round(0.01 * len(c))))
    acum = np.cumsum(c) / c.sum()
    return {"registros": int(c.sum()), "celdas_ocupadas": int(len(c)),
            "frac_en_1pct_celdas": float(c[:top].sum() / c.sum()),
            "celdas_mitad": int(np.searchsorted(acum, 0.5) + 1),
            "max_por_celda": int(c[0])}


def contorno(ax):
    g = json.load(open(OUTLINE))["features"][0]["geometry"]
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    for poly in polys:
        ring = np.asarray(poly[0])
        ax.plot(ring[:, 0], ring[:, 1], color="#52514e", lw=0.6)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="aves_2018")
    ap.add_argument("--d-km", type=float, default=5)
    ap.add_argument("--regla", default="mindeg")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--titulo", default="Birds, 2018")
    args = ap.parse_args()
    tag = f"{args.d_km:g}km"

    lat0, lon0 = latlon(f"{args.label}_dup")
    lat1, lon1 = latlon(args.label)
    mask = np.load(PROC / f"{args.label}_thin_{args.regla}_{tag}_s{args.semilla}.npy")
    if len(mask) != len(lat1):
        raise SystemExit("la máscara no coincide con el conjunto limpio")
    lat2, lon2 = lat1[mask], lon1[mask]

    antes = concentracion(lat0, lon0, args.d_km * 1000)
    despues = concentracion(lat2, lon2, args.d_km * 1000)
    res = {"label": args.label, "d_km": args.d_km, "regla": args.regla,
           "antes": antes, "despues": despues}
    (PROC / f"{args.label}_concentracion_{tag}.json").write_text(json.dumps(res, indent=2))
    for k, v in (("antes", antes), ("después", despues)):
        print(f"{k:>8}: {v['registros']:>9,} registros en {v['celdas_ocupadas']:,} celdas de {args.d_km:g} km; "
              f"el 1 % de celdas reúne {100 * v['frac_en_1pct_celdas']:.1f} %; "
              f"la mitad cabe en {v['celdas_mitad']:,} celdas; máximo {v['max_por_celda']:,}")

    ext = (-86.0, -82.5, 8.0, 11.3)          # Costa Rica continental
    azules = LinearSegmentedColormap.from_list(
        "azules", plt.get_cmap("Blues")(np.linspace(0.25, 1.0, 256)))
    kw = dict(gridsize=90, extent=ext, mincnt=1, cmap=azules, linewidths=0,
              norm=LogNorm(vmin=1, vmax=max(1, antes["max_por_celda"])))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), sharey=True)
    fig.subplots_adjust(left=0.02, right=0.88, top=0.9, bottom=0.04, wspace=0.04)
    paneles = ((lon0, lat0, f"(a) All records: {antes['registros']:,}"),
               (lon2, lat2, f"(b) After thinning at {args.d_km:g} km: {despues['registros']:,}"))
    for ax, (x, y, t) in zip(axes, paneles):
        hb = ax.hexbin(x, y, **kw)
        contorno(ax)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_aspect(1 / np.cos(np.radians(9.7)))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(t, fontsize=8.5, loc="left", family="serif")
    cax = fig.add_axes([0.9, 0.12, 0.02, 0.7])
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label("Records per cell", fontsize=8, family="serif")
    cb.ax.tick_params(labelsize=7)
    out = ROOT / "paper" / f"fig_mapa_{args.label}_{tag}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
