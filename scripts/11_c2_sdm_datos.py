#!/usr/bin/env python3
"""
11_c2_sdm_datos.py — Datos del piloto SDM (C2, revisores 1 y 2).

Prepara, para un grupo pequeño y representativo de especies, todo lo que el
piloto en R (bench/sdm_piloto.R) necesita, sin que R tenga que filtrar nada:

  especies   por taxón (aves 2018, anfibios), entre las que tienen al menos
             --min-registros registros únicos, las más cercanas a los
             percentiles 10, 30, 50, 70 y 90 del número de registros. La regla
             es fija y no mira resultados: cubre tamaños de muestra distintos
             sin elegir a mano.
  registros  registros únicos de la especie (presencias repetidas colapsadas,
             como en el paper), con lon/lat.
  filtrado   máscaras EXACTAS del filtrado por especie con la regla de ThinGPU
             (mindeg, thin.py) a 1, 5 y 10 km, semillas 0..9 (las semillas
             solo rompen empates, así que miden la variabilidad del filtrado).
  esfuerzo   todos los registros del taxón con repetidas (<label>_dup): fondo
             de grupo objetivo (Phillips et al. 2009) y mapa de esfuerzo.

Salidas en data/interim/sdm/:
  especies.csv                    especie, taxón, N
  <label>__<otu>.csv.gz           lon, lat, k1_s0 ... k10_s9 (1 = conservado)
  esfuerzo_<label>.csv.gz         lon, lat de todos los registros del taxón

Uso: python3 scripts/11_c2_sdm_datos.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import thin
from fig_mapa import latlon

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "interim" / "sdm"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", default=["aves_2018", "amphibia"])
    ap.add_argument("--min-registros", type=int, default=100)
    ap.add_argument("--percentiles", type=float, nargs="+", default=[10, 30, 50, 70, 90])
    ap.add_argument("--radii-km", type=float, nargs="+", default=[1, 5, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    elegidas = []
    for label in args.labels:
        lat, lon = latlon(label)
        xyz = np.load(PROC / f"{label}_xyz_f64.npy")
        otu = np.load(PROC / f"{label}_otu.npy")
        nombres = pd.read_csv(PROC / f"{label}_otu_nombres.csv").set_index("otu_id").nombre
        if not (len(lat) == len(xyz) == len(otu)):
            raise SystemExit(f"{label}: tamaños distintos entre metadatos, xyz y otu")

        n = pd.Series(otu[otu >= 0]).value_counts()
        n = n[n >= args.min_registros]
        objetivo = np.percentile(n.values, args.percentiles)
        usados = set()
        for q, t in zip(args.percentiles, objetivo):
            orden = (n - t).abs().sort_values(kind="stable").index
            k = next(o for o in orden if o not in usados)
            usados.add(k)
            idx = np.flatnonzero(otu == k)
            cols = {"lon": lon[idx], "lat": lat[idx]}
            for km in args.radii_km:
                ip, ix, _ = thin.neighbor_csr(xyz[idx], km * 1000)
                for s in args.seeds:
                    kept, _, _ = thin.thin_greedy(ip, ix, "mindeg", s, check_rounds=False, ids=idx)
                    cols[f"k{km:g}_s{s}"] = kept.astype(np.int8)
            df = pd.DataFrame(cols)
            df.to_csv(OUT / f"{label}__{k}.csv.gz", index=False)
            elegidas.append({"label": label, "otu": int(k), "especie": nombres[k], "N": int(len(idx)),
                             "percentil": q, **{f"retenidos_{km:g}km_s0": int(df[f"k{km:g}_s0"].sum())
                                                for km in args.radii_km}})
            print(f"{label}: {nombres[k]} (N = {len(idx):,}, percentil {q:g}) -> "
                  + ", ".join(f"{km:g} km: {int(df[f'k{km:g}_s0'].sum())}" for km in args.radii_km))
        lat_d, lon_d = latlon(f"{label}_dup")
        pd.DataFrame({"lon": lon_d, "lat": lat_d}).to_csv(OUT / f"esfuerzo_{label}.csv.gz", index=False)

    pd.DataFrame(elegidas).to_csv(OUT / "especies.csv", index=False)
    print(f"-> {(OUT / 'especies.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
