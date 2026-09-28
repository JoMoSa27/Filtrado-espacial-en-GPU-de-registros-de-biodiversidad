#!/usr/bin/env python3
"""
05_references.py — Salidas EXACTAS de referencia para validar el benchmark en GPU.

El benchmark (bench/) no se valida "a ojo": debe producir exactamente esto.

  W1  densidad de registros del grupo: para cada registro, cuantos OTROS
      registros de la nube estan a menos de d (geodesica, via cuerda).
      -> data/processed/<label>_count_<d>km.npy          int32 (N,)
  W2  filtrado por especie, voraz por prioridad (thin.py), regla y semilla:
      -> data/processed/<label>_thin_<regla>_<d>km_s<semilla>.npy   bool (N,)
      Registros sin especie/BIN (otu = -1) quedan en False.
      Requiere <label>_otu.npy (lo escribe 04_species_stats.py).

Tambien escribe bench/splitmix_vectors.csv: valores de prueba del hash de
prioridades, para comprobar que la implementacion CUDA da los mismos bits.

Uso:
  python 05_references.py --label e1_aves_2018 --radii-km 1 5 10 --w1
  python 05_references.py --label aves_2018 --radii-km 1 5 10 --w2 --rules random mindeg
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import geo
import thin

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def ncpu():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def write_vectors():
    out = ROOT / "bench" / "splitmix_vectors.csv"
    out.parent.mkdir(exist_ok=True)
    rows = ["seed,id,hash_uint64"]
    for seed in (0, 1, 2026):
        ids = np.array([0, 1, 2, 3, 1000, 123456789, 2**31 - 1], dtype=np.uint64)
        for i, h in zip(ids, thin.record_hash(ids, seed)):
            rows.append(f"{seed},{int(i)},{int(h)}")
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"-> {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--radii-km", type=float, nargs="+", default=[1, 5, 10])
    ap.add_argument("--w1", action="store_true")
    ap.add_argument("--w2", action="store_true")
    ap.add_argument("--rules", nargs="+", default=["random", "mindeg"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--w1-max-n", type=int, default=400_000,
                    help="W1 exacto solo hasta este N (arriba, la GPU se valida por "
                         "acuerdo entre sus tres estructuras)")
    ap.add_argument("--export-csv", action="store_true",
                    help="escribe data/interim/<label>_r.csv.gz (lon, lat, otu) para "
                         "correr GeoThinneR con exactamente la misma entrada")
    args = ap.parse_args()
    write_vectors()

    xyz = np.load(PROCESSED / f"{args.label}_xyz_f64.npy")
    if args.export_csv:
        import pandas as pd
        pq = ROOT / "data" / "interim" / f"{args.label}.parquet"
        meta = pd.read_parquet(pq) if pq.exists() else pd.read_csv(pq.with_suffix(".csv.gz"))
        otu_p = PROCESSED / f"{args.label}_otu.npy"
        otu = np.load(otu_p) if otu_p.exists() else np.zeros(len(meta), np.int32)
        out = pd.DataFrame({"row": np.arange(len(meta)), "lon": meta.lon.values,
                            "lat": meta.lat.values, "otu": otu})
        out = out[out.otu >= 0]
        dest = ROOT / "data" / "interim" / f"{args.label}_r.csv.gz"
        out.to_csv(dest, index=False, compression="gzip")
        print(f"-> {dest.relative_to(ROOT)} ({len(out):,} filas)")
    timing = {}
    for km in args.radii_km:
        d_m = km * 1000.0
        tag = f"{km:g}km"
        if args.w1 and len(xyz) > args.w1_max_n:
            print(f"[{args.label}] W1 d={km:g} km: N={len(xyz):,} > {args.w1_max_n:,}, "
                  f"sin referencia exacta (validar por acuerdo entre estructuras)")
        elif args.w1:
            chord = float(geo.chord_radius(d_m)) * thin._STRICT
            t = time.perf_counter()
            tree = cKDTree(xyz)
            t_build = time.perf_counter() - t
            t = time.perf_counter()
            cnt = tree.query_ball_point(xyz, chord, return_length=True,
                                        workers=ncpu()) - 1
            t_query = time.perf_counter() - t
            np.save(PROCESSED / f"{args.label}_count_{tag}.npy", cnt.astype(np.int32))
            # El tiempo de la referencia es tambien la linea base en CPU (cKDTree).
            timing[tag] = {"N": int(len(xyz)), "d_km": km, "hilos": ncpu(),
                           "build_s": t_build, "query_s": t_query,
                           "vecinos_total": int(cnt.sum()),
                           "vecinos_medio": float(cnt.mean()),
                           "vecinos_max": int(cnt.max())}
            print(f"[{args.label}] W1 d={km:g} km: vecinos {int(cnt.sum()):,} "
                  f"(consulta {t_query:.1f} s, {ncpu()} hilos)")
        if args.w2:
            otu = np.load(PROCESSED / f"{args.label}_otu.npy")
            order = np.argsort(otu, kind="stable")
            ks = otu[order]
            starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
            ends = np.r_[starts[1:], len(ks)]
            for rule in args.rules:
                for seed in args.seeds:
                    t = time.perf_counter()
                    mask = np.zeros(len(xyz), bool)
                    for a, b in zip(starts, ends):
                        if ks[a] < 0:
                            continue
                        idx = order[a:b]
                        if len(idx) == 1:
                            mask[idx] = True
                            continue
                        ip, ix, _ = thin.neighbor_csr(xyz[idx], d_m)
                        k, _, _ = thin.thin_greedy(ip, ix, rule, seed, check_rounds=False, ids=idx)
                        mask[idx[k]] = True
                    np.save(PROCESSED / f"{args.label}_thin_{rule}_{tag}_s{seed}.npy", mask)
                    print(f"[{args.label}] W2 {rule} s{seed} d={km:g} km: "
                          f"retenidos {int(mask.sum()):,} de {int((otu >= 0).sum()):,} "
                          f"({time.perf_counter() - t:.1f} s)")
    if timing:
        import json
        (PROCESSED / f"{args.label}_cpu_w1.json").write_text(
            json.dumps(timing, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
