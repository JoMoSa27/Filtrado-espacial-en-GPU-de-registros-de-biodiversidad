#!/usr/bin/env python3
"""
03_cluster_stats.py — Predictores de agrupamiento para la tabla del paper.

Para cada radio r (= tamano de celda de la grilla) calcula:
  Morisita 2D   cuadrantes r x r en proyeccion de igual area (Lambert azimutal),
                Q = area del rectangulo de estudio / r^2. ~1 = aleatorio.
                Promedio y cv sobre 5 origenes de malla (control de MAUP).
  hacinamiento  Lloyd (1967): Sum n_i(n_i-1) / N sobre la grilla 3D: cuantos OTROS
                registros comparten celda con un registro al azar (~ trabajo de la grilla)
  n_max, p99    cola de la distribucion de celdas 3D (desbalance entre warps)
  K de Ripley   K(r)/(pi r^2) en 3D con radio de cuerda. ~1 = aleatorio.

Por que Morisita en 2D y no en 3D: en la grilla 3D, Q incluye celdas en el
grosor del casquete (y en el interior de la Tierra para datos globales) donde
nunca puede caer un punto, y el indice sale inflado incluso con datos
aleatorios. Verificado: 3D daba ~366 para aleatorio; 2D de igual area da ~1.

Uso:
    python 03_cluster_stats.py --label aves --radii-km 1 5 10
Lee data/processed/<label>_xyz_f64.npy y data/interim/<label>.parquet|csv.gz
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import geo

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"


def load(label):
    xyz = np.load(PROCESSED / f"{label}_xyz_f64.npy")
    pq, gz = INTERIM / f"{label}.parquet", INTERIM / f"{label}.csv.gz"
    meta = pd.read_parquet(pq) if pq.exists() else pd.read_csv(gz)
    return xyz, meta


def ripley(xyz, r, lam_area, m=20000, seed=0):
    rng = np.random.default_rng(seed)
    q = xyz if len(xyz) <= m else xyz[rng.choice(len(xyz), m, replace=False)]
    k = cKDTree(xyz).query_ball_point(q, r, return_length=True) - 1
    lam = len(xyz) / lam_area
    return float(k.mean() / lam / (np.pi * r * r))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--radii-km", type=float, nargs="+", default=[1, 5, 10])
    ap.add_argument("--offsets", type=int, default=5)
    ap.add_argument("--rect", default="data",
                    help="area de estudio para Q (Morisita) y lambda (Ripley): "
                         "'data' = rectangulo lat/lon de los datos; 'cr' = rectangulo "
                         "comun de Costa Rica (geo.CR_RECT), el que hace comparables "
                         "a los taxones entre si")
    args = ap.parse_args()

    xyz, meta = load(args.label)
    N = len(xyz)
    if args.rect == "cr":
        rect = geo.CR_RECT
    else:
        rect = (meta.lat.min(), meta.lat.max(), meta.lon.min(), meta.lon.max())
    area = geo.rect_area_m2(*rect)
    rows = []
    for km in args.radii_km:
        rm = km * 1000.0
        chord = float(geo.chord_radius(rm))
        mor, cv = geo.morisita_2d(meta.lat.values, meta.lon.values, rm, rect,
                                  n_offsets=args.offsets)
        c, Q3 = geo.cell_counts(xyz, chord)
        rows.append({
            "radio_km": km, "N": N,
            "morisita_2d": mor, "morisita_2d_cv": cv,
            "hacinamiento_medio": float((c * (c - 1.0)).sum() / N),  # Lloyd (1967)
            "n_max": int(c.max()), "p99": float(np.percentile(c, 99)),
            "celdas3d_ocupadas": int(len(c)), "celdas3d_totales": Q3,
            "ripley_norm": ripley(xyz, chord, area),
        })
    out = {"label": args.label, "N": N, "rect": list(map(float, rect)),
           "rect_regla": args.rect, "area_km2": area / 1e6,
           "por_radio": rows}
    (PROCESSED / f"{args.label}_cluster_stats.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.4g}"))
    print(f"\n-> data/processed/{args.label}_cluster_stats.json")


if __name__ == "__main__":
    main()
