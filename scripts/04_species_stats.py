#!/usr/bin/env python3
"""
04_species_stats.py — Filtrado POR ESPECIE y predictores de agrupamiento por especie.

Por que existe: el filtrado espacial se aplica por especie (spThin, GeoThinneR
con grupos: "independently within each group"). Un registro de una especie no
elimina a uno de otra. Este script hace eso para cada especie (u OTU: especie,
o BIN de BOLD si no hay especie) de un conjunto ya preparado por 02_prepare.py,
y deja una fila por (especie, distancia) con:

  N, pares a menos de d, grado medio/maximo
  predictores: Morisita 2D (area comun), hacinamiento y n_max en la grilla 3D,
               Ripley normalizado (sale gratis de los pares)
  filtrado de referencia (thin.py): retenidos con la regla de GeoThinneR
               (mejor de 10), voraz aleatorio y voraz de grado menor; rondas
               paralelas; trabajo con salida temprana; tiempos de CPU; validacion

Ademas, con --density, mide en CPU (cKDTree, todos los nucleos asignados) el
conteo de registros de TODO el taxon a menos de d de cada registro: la
densidad de registros del grupo objetivo (esfuerzo de muestreo), la misma
consulta por radio que usa el benchmark en GPU sobre la nube completa.

Salidas en data/processed/:
  <label>_por_especie.csv      una fila por (especie, d)
  <label>_otu.npy              int32 por registro: id de especie (-1 = sin especie)
  <label>_otu_nombres.csv      id -> nombre
  <label>_por_especie.json     resumen por d (+ densidad del grupo si --density)

Uso:
  python 04_species_stats.py --label aves_2018 --radii-km 1 5 10 --rect cr --density
  python 04_species_stats.py --label atun_gt --radii-km 10 25 50 --single-otu "Thunnus albacares"
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import geo
import thin

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"


def load(label):
    xyz = np.load(PROCESSED / f"{label}_xyz_f64.npy")
    pq, gz = INTERIM / f"{label}.parquet", INTERIM / f"{label}.csv.gz"
    meta = pd.read_parquet(pq) if pq.exists() else pd.read_csv(gz)
    if len(meta) != len(xyz):
        raise SystemExit(f"{label}: metadatos ({len(meta)}) y xyz ({len(xyz)}) no coinciden")
    return xyz, meta


def ncpu():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def species_row(sub_xyz, lat, lon, ids, d_m, rect, area, trials, check):
    N = len(sub_xyz)
    chord = float(geo.chord_radius(d_m))
    t = time.perf_counter()
    csr = thin.neighbor_csr(sub_xyz, d_m)
    t_vec = time.perf_counter() - t
    r = thin.run_all(sub_xyz, d_m, trials=trials, check=check, ids=ids, csr=csr)
    r["t_vecinos_s"] = t_vec
    c, _ = geo.cell_counts(sub_xyz, chord)
    r["hacinamiento"] = float((c * (c - 1.0)).sum() / N)  # Lloyd (1967): m* = sum n(n-1)/N
    r["n_max_celda"] = int(c.max())
    if N >= 2:
        mor, cv = geo.morisita_2d(lat, lon, d_m, rect)
        r["morisita_2d"], r["morisita_2d_cv"] = mor, cv
        # Ripley: K(d) = A * 2P / (N (N-1)); normalizado por pi d^2 (~1 aleatorio)
        r["ripley_norm"] = float(area * 2 * r["pares"] / (N * (N - 1)) / (np.pi * d_m ** 2))
    else:
        r["morisita_2d"] = r["morisita_2d_cv"] = r["ripley_norm"] = np.nan
    r["frac_ret_mindeg"] = r["ret_mindeg_mejor"] / N
    r["frac_ret_maxdeg"] = r["ret_maxdeg_mejor"] / N
    r["trabajo_rel_random"] = (r["trabajo_random_media"] / (2 * r["pares"])
                               if r["pares"] else np.nan)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--radii-km", type=float, nargs="+", default=[1, 5, 10])
    ap.add_argument("--rect", default="cr", help="'cr' (comun) o 'data'")
    ap.add_argument("--min-n", type=int, default=2,
                    help="especies con menos registros no se filtran (nada que filtrar)")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--check-max-n", type=int, default=200_000,
                    help="valida (pares < d y maximalidad) hasta este N por especie")
    ap.add_argument("--single-otu", default=None,
                    help="tratar todo el conjunto como una sola especie (tortuga, atun)")
    ap.add_argument("--density", action="store_true",
                    help="medir tambien la densidad de registros de todo el taxon en CPU")
    ap.add_argument("--density-from", default=None,
                    help="etiqueta de la nube para la densidad (p. ej. <label>_dup: el "
                         "esfuerzo cuenta TODOS los registros, incluidas visitas repetidas)")
    args = ap.parse_args()
    if args.density_from:
        args.density = True

    xyz, meta = load(args.label)
    otu = geo.otu_key(meta, args.single_otu)
    codes, names = pd.factorize(otu, use_na_sentinel=True)
    codes = codes.astype(np.int32)
    np.save(PROCESSED / f"{args.label}_otu.npy", codes)
    pd.DataFrame({"otu_id": np.arange(len(names)), "nombre": names}).to_csv(
        PROCESSED / f"{args.label}_otu_nombres.csv", index=False)

    rect = (geo.CR_RECT if args.rect == "cr" else
            (meta.lat.min(), meta.lat.max(), meta.lon.min(), meta.lon.max()))
    area = geo.rect_area_m2(*rect)
    sizes = np.bincount(codes[codes >= 0], minlength=len(names))
    sin_otu = int((codes < 0).sum())
    es_bin = np.array([str(n).startswith("BOLD:") for n in names], dtype=bool)
    print(f"[{args.label}] {len(xyz):,} registros; {len(names):,} OTU "
          f"({int((~es_bin).sum()):,} especies, {int(es_bin.sum()):,} BIN); "
          f"{sin_otu:,} sin especie ni BIN (excluidos del filtrado por especie)")
    if len(sizes):
        print(f"  registros por OTU: mediana {np.median(sizes):.0f}, p90 "
              f"{np.percentile(sizes, 90):.0f}, max {sizes.max():,} ({names[sizes.argmax()]}); "
              f">=100: {(sizes >= 100).sum()}, >=1000: {(sizes >= 1000).sum()}, "
              f">=10000: {(sizes >= 10000).sum()}")

    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(names) + 1))
    rows, resumen = [], {"label": args.label, "n_registros": int(len(xyz)),
                         "n_otu": int(len(names)), "n_bin": int(es_bin.sum()),
                         "sin_otu": sin_otu, "rect": list(map(float, rect)),
                         "por_d": []}
    for km in args.radii_km:
        d_m = km * 1000.0
        t0 = time.perf_counter()
        for k in range(len(names)):
            idx = order[bounds[k]:bounds[k + 1]]
            if len(idx) < args.min_n:
                continue
            r = species_row(xyz[idx], meta.lat.values[idx], meta.lon.values[idx],
                            idx, d_m, rect, area, args.trials,
                            check=len(idx) <= args.check_max_n)
            r.update(otu_id=k, otu=names[k], es_bin=bool(es_bin[k]))
            rows.append(r)
        t_total = time.perf_counter() - t0
        R = pd.DataFrame([x for x in rows if x["d_km"] == km])
        s = {"d_km": km, "otu_filtradas": int(len(R)),
             "registros": int(R.N.sum()), "pares": int(R.pares.sum()),
             "ret_maxdeg": int(R.ret_maxdeg_mejor.sum()),
             "ret_random": int(R.ret_random_mejor.sum()),
             "ret_mindeg": int(R.ret_mindeg_mejor.sum()),
             "malos_total": int(R.filter(like="_malos").sum().sum()),
             "agregables_maxdeg": int(R.maxdeg_agregables.sum()),
             "rondas_max": int(max(R.rondas_random_max.max(), R.rondas_mindeg_max.max())),
             "t_vecinos_s": float(R.t_vecinos_s.sum()),
             "t_maxdeg_s": float(R.t_maxdeg_s.sum()),
             "t_mindeg_s": float(R.t_mindeg_s.sum()),
             "t_total_s": t_total}
        big = R[R.N >= 100]
        corr = {}
        for pred in ("morisita_2d", "hacinamiento", "n_max_celda", "ripley_norm", "N"):
            for tgt in ("frac_ret_mindeg", "rondas_random_max", "trabajo_rel_random"):
                if len(big) >= 10 and big[tgt].notna().sum() >= 10:
                    rho = spearmanr(big[pred], big[tgt], nan_policy="omit").statistic
                    corr[f"{pred}~{tgt}"] = float(rho)
        s["spearman_N>=100"] = corr
        s["n_otu_N>=100"] = int(len(big))

        if args.density:
            cloud = (np.load(PROCESSED / f"{args.density_from}_xyz_f64.npy")
                     if args.density_from else xyz)
            chord = float(geo.chord_radius(d_m))
            t = time.perf_counter()
            tree = cKDTree(cloud)
            t_build = time.perf_counter() - t
            t = time.perf_counter()
            dens = tree.query_ball_point(cloud, chord, return_length=True, workers=ncpu()) - 1
            s["densidad_grupo"] = {
                "nube": args.density_from or args.label, "N": int(len(cloud)),
                "t_build_s": t_build, "t_query_s": time.perf_counter() - t,
                "hilos": ncpu(), "vecinos_total": int(dens.sum()),
                "vecinos_medio": float(dens.mean()), "vecinos_max": int(dens.max()),
                "vecinos_p99": float(np.percentile(dens, 99))}
        resumen["por_d"].append(s)
        print(f"\n  d = {km:g} km: {s['otu_filtradas']:,} OTU, {s['registros']:,} registros, "
              f"{s['pares']:,} pares mismos-OTU")
        print(f"    retenidos  GeoThinneR(mejor de {args.trials}) {s['ret_maxdeg']:,} | "
              f"voraz aleatorio {s['ret_random']:,} | grado menor {s['ret_mindeg']:,}")
        print(f"    validacion: pares a menos de d = {s['malos_total']} ; "
              f"agregables en GeoThinneR = {s['agregables_maxdeg']:,} ; rondas max = {s['rondas_max']}")
        print(f"    CPU total {t_total:.1f} s (vecinos {s['t_vecinos_s']:.1f}, "
              f"GeoThinneR-regla {s['t_maxdeg_s']:.1f}, grado menor {s['t_mindeg_s']:.1f})")
        if corr:
            print("    Spearman (OTU con N>=100): " + ", ".join(
                f"{k} {v:+.2f}" for k, v in corr.items() if k.endswith("frac_ret_mindeg")))
        if args.density:
            g = s["densidad_grupo"]
            print(f"    densidad del grupo ({g['nube']}, N={g['N']:,}, CPU {g['hilos']} hilos): "
                  f"{g['t_query_s']:.2f} s, vecinos medio {g['vecinos_medio']:.1f}, "
                  f"max {g['vecinos_max']:,}")

    pd.DataFrame(rows).to_csv(PROCESSED / f"{args.label}_por_especie.csv", index=False)
    (PROCESSED / f"{args.label}_por_especie.json").write_text(
        json.dumps(resumen, indent=2, default=float), encoding="utf-8")
    print(f"\n-> data/processed/{args.label}_por_especie.csv / .json")


if __name__ == "__main__":
    main()
