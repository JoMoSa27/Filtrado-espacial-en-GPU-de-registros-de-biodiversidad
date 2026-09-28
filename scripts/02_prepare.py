#!/usr/bin/env python3
"""
02_prepare.py — De CSV crudo de GBIF a arreglos listos para los kernels CUDA.

Usa exactamente las mismas funciones que 00_explorar.py (modulo geo.py), asi
que lo que viste en el reporte de exploracion es lo que se guarda aqui.

Salidas en data/processed/:
  <label>_xyz_f64.npy           (N,3) float64 C-contiguo  -> construir y validar
  <label>_xyz_f32_centered.npy  (N,3) float32 sin centroide -> kernels en f32
  <label>_centroid_f64.npy      para volver a coordenadas absolutas
  <label>_report.json           cuantos se descartaron en cada paso

Uso:
    python 02_prepare.py data/raw/gbif_cr_aves/occurrence.txt --label aves
    python 02_prepare.py data/raw/geothinner/thunnus.rda --label tuna
    python 02_prepare.py ... --label aves_dup --keep-duplicates
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import geo

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--label", required=True)
    ap.add_argument("--lat-col")
    ap.add_argument("--lon-col")
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--round-decimals", type=int, default=5)
    ap.add_argument("--max-uncertainty-km", type=float, default=geo.MAX_UNCERTAINTY_M / 1000,
                    help="descarta registros con incertidumbre mayor (0 = no filtrar; "
                         "se usa para dar a GeoThinneR y a la GPU la misma entrada publicada)")
    ap.add_argument("--sample", type=int, default=None,
                    help="submuestra aleatoria de N puntos DESPUES de limpiar "
                         "(para el barrido de N a partir de una sola descarga)")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    frames = [geo.read_any(p, args.lat_col, args.lon_col) for p in args.inputs]
    raw = pd.concat(frames, ignore_index=True)
    max_unc = np.inf if args.max_uncertainty_km <= 0 else args.max_uncertainty_km * 1000
    df, pasos = geo.clean(raw, args.keep_duplicates, args.round_decimals, max_unc)
    if len(df) == 0:
        raise SystemExit("No quedaron registros tras la limpieza.")
    if args.sample and args.sample < len(df):
        # Submuestreo uniforme sin reemplazo con semilla fija: reproducible, y
        # conserva la FORMA del agrupamiento (el indice de Morisita es, en
        # esperanza, independiente de la densidad bajo adelgazamiento aleatorio).
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
        pasos.append({"regla": f"Submuestra aleatoria (semilla {args.seed})",
                      "antes": int(len(raw)), "descartados": 0,
                      "despues": int(len(df)),
                      "razon": "Barrido de N desde una sola descarga."})

    xyz = np.ascontiguousarray(geo.to_cartesian(df.lat.values, df.lon.values))
    err, _ = geo.chord_check(df, xyz)

    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    L = args.label
    np.save(PROCESSED / f"{L}_xyz_f64.npy", xyz)
    centroid = xyz.mean(axis=0)
    np.save(PROCESSED / f"{L}_xyz_f32_centered.npy",
            np.ascontiguousarray((xyz - centroid).astype(np.float32)))
    np.save(PROCESSED / f"{L}_centroid_f64.npy", centroid)

    keep = [c for c in ("species", "scientificName", "verbatimScientificName",
                        "taxonRank", "year", "basisOfRecord", "gbifID",
                        "datasetKey") if c in df.columns]
    meta = df[["lat", "lon"] + keep].copy()
    try:
        meta.to_parquet(INTERIM / f"{L}.parquet", index=False)
    except Exception:
        meta.to_csv(INTERIM / f"{L}.csv.gz", index=False, compression="gzip")

    rep = {
        "label": L, "files": args.inputs, "n_raw": int(len(raw)),
        "n_final": int(len(df)), "pasos": pasos,
        "chord_arc_max_error_m": float(err.max()),
        "bbox": {"lat": [float(df.lat.min()), float(df.lat.max())],
                 "lon": [float(df.lon.min()), float(df.lon.max())]},
        "chord_radii_m": {f"{k}km": float(geo.chord_radius(k * 1000))
                          for k in (1, 5, 10, 25, 50)},
    }
    (PROCESSED / f"{L}_report.json").write_text(json.dumps(rep, indent=2),
                                                 encoding="utf-8")

    print(f"\n[{L}]  crudos {len(raw):,}")
    for p in pasos:
        print(f"  - {p['regla']:<34} {p['descartados']:>10,}")
    print(f"  finales {len(df):,}   error cuerda-arco max {err.max():.2e} m")
    print(f"  -> data/processed/{L}_xyz_f64.npy")


if __name__ == "__main__":
    main()
