#!/usr/bin/env python3
"""C1: tiempo de UNA corrida del filtrado en CPU (nuestra implementacion), para la Tabla 1.

04_species_stats.py mide 10 semillas, cada una con el voraz secuencial, las rondas
paralelas y la validacion; sirve para el analisis por especie pero no como tiempo de
una corrida. Aqui se mide solo lo que hace falta para obtener el resultado:

  t_vecinos_s  grafo de vecinos (cKDTree, un hilo)
  t_mindeg_s   voraz con prioridad 'mindeg', una semilla, sin validar

y se comprueba que el resultado sea la referencia f64 de 05_references.py.

Uso:
  python scripts/13_c1_tiempos_cpu.py --label tortuga_gt --radii-km 10 25 50 --reps 5
"""
import argparse
import platform
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd

import thin

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "bench" / "results"


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--radii-km", type=float, nargs="+", default=[10, 25, 50])
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    xyz = np.load(PROCESSED / f"{args.label}_xyz_f64.npy")
    nodo, modelo = socket.gethostname(), cpu_model()
    # calentamiento: la primera llamada compila con numba y no debe contar
    ip, ix, _ = thin.neighbor_csr(xyz[:2000], args.radii_km[0] * 1000)
    thin.thin_greedy(ip, ix, "mindeg", 0, check_rounds=False)
    rows = []
    for km in args.radii_km:
        ref_path = PROCESSED / f"{args.label}_thin_mindeg_{km:g}km_s0.npy"
        ref = np.load(ref_path) if ref_path.exists() else None
        for rep in range(args.reps):
            t = time.perf_counter()
            indptr, indices, pares = thin.neighbor_csr(xyz, km * 1000)
            t_vec = time.perf_counter() - t
            t = time.perf_counter()
            kept, _, _ = thin.thin_greedy(indptr, indices, "mindeg", 0, check_rounds=False)
            t_md = time.perf_counter() - t
            coincide = "sin_ref" if ref is None else str(bool(np.array_equal(kept, ref)))
            rows.append(dict(label=args.label, N=len(xyz), d_km=km, rep=rep, pares=int(pares),
                             t_vecinos_s=t_vec, t_mindeg_s=t_md, retenidos=int(kept.sum()),
                             coincide_ref=coincide, nodo=nodo, cpu=modelo))
            print(f"{args.label} {km:g} km rep {rep}: vecinos {t_vec:.3f} s, mindeg {t_md:.3f} s, "
                  f"{kept.sum():,} retenidos, igual a la referencia: {coincide}", flush=True)
            del indptr, indices
    out = RESULTS / f"c1_cpu_{args.label}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
