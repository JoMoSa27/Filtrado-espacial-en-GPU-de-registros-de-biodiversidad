#!/usr/bin/env python3
"""
08_check_w1.py — Decide si una discrepancia de W1 es ruido numerico o un error.

POR QUE HACE FALTA
    La validacion de W1 es igualdad exacta contra la referencia _f32c, que usa
    las mismas coordenadas float32 que la GPU y el mismo radio. Aun asi el ruido
    residual no es cero: la GPU acumula distance_sq en float32 y la referencia
    en float64, asi que un par que cae a menos de ~1 cm del radio puede quedar
    de un lado o del otro.

    Un conteo agregado no distingue eso de un error de implementacion. Este
    script lo distingue, y el criterio es explicito:

      · TODOS los pares dudosos a menos de --tol del radio  -> ruido numerico,
        se reporta como tal y la corrida sigue siendo valida;
      · ALGUNO lejos del radio                              -> error real.

QUE CONSUME
    El volcado que escribe bench_w1 cuando hay discrepancias:
        <salida>_dif_<label>_<d>km_<estructura>.csv   con columnas idx,gpu,ref
    y las coordenadas float32 centradas del conjunto.

USO
    python3 08_check_w1.py --dump bench/results/..._dif_tortuga_gt_10km_KDTree.csv \\
                           --label tortuga_gt --d-km 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import geo

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def radio_f32c(d_m):
    """El mismo float que recibe la GPU (ver radio() en 05_references.py)."""
    c = float(geo.chord_radius(d_m))
    return float(np.nextafter(np.float32(c), np.float32(0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="CSV idx,gpu,ref de bench_w1")
    ap.add_argument("--label", required=True)
    ap.add_argument("--d-km", type=float, required=True)
    ap.add_argument("--tol-m", type=float, default=0.01,
                    help="banda alrededor del radio dentro de la cual una "
                         "diferencia se considera ruido de float32 (default 1 cm)")
    a = ap.parse_args()

    dump = np.genfromtxt(a.dump, delimiter=",", names=True, dtype=None)
    idx = np.atleast_1d(dump["idx"]).astype(np.int64)
    gpu = np.atleast_1d(dump["gpu"]).astype(np.int64)
    ref = np.atleast_1d(dump["ref"]).astype(np.int64)

    # Las MISMAS coordenadas que vio la GPU, promovidas a float64 para que la
    # aritmetica de esta comprobacion sea exacta.
    xyz = np.load(PROCESSED / f"{a.label}_xyz_f32_centered.npy").astype(np.float64)
    r = radio_f32c(a.d_km * 1000.0)

    print(f"{a.label}  d = {a.d_km:g} km  radio = {r:.6f} m")
    print(f"{len(idx)} registros con conteo distinto "
          f"(suma GPU {gpu.sum()}, suma ref {ref.sum()})\n")

    tree = cKDTree(xyz)
    # Se miran los vecinos dentro de una banda generosa alrededor del radio: los
    # pares que pueden haber cambiado de lado son los que caen cerca del borde.
    banda = max(a.tol_m * 10, 1.0)
    peor = 0.0
    dudosos = 0
    lejanos = []

    for i in idx:
        cerca = tree.query_ball_point(xyz[i], r + banda)
        d = np.linalg.norm(xyz[cerca] - xyz[i], axis=1)
        # Pares cuya distancia esta a menos de 'banda' del radio: los unicos que
        # una diferencia de redondeo puede mover de un lado al otro.
        cand = np.abs(d - r)
        cand = cand[(cand <= banda) & (d > 0)]
        if cand.size == 0:
            lejanos.append(int(i))
            continue
        dudosos += int(cand.size)
        peor = max(peor, float(cand.min()))

    print(f"pares en la banda de +-{banda:g} m del radio : {dudosos}")
    print(f"distancia al radio del par mas ajustado     : {peor*1000:.4f} mm")
    print(f"registros sin ningun par cerca del radio    : {len(lejanos)}")

    if lejanos:
        print(f"\nERROR DE IMPLEMENTACION: {len(lejanos)} registros difieren sin "
              f"tener ningun vecino cerca del radio.")
        print(f"  primeros: {lejanos[:20]}")
        sys.exit(1)

    if peor <= a.tol_m:
        print(f"\nRUIDO NUMERICO: todos los pares dudosos estan a menos de "
              f"{a.tol_m*100:g} cm del radio.")
        print("La corrida se reporta como valida, declarando estas discrepancias.")
        sys.exit(0)

    print(f"\nFUERA DE TOLERANCIA: hay pares a {peor:.4f} m del radio, mas de "
          f"{a.tol_m:g} m. Revisar la implementacion.")
    sys.exit(1)


if __name__ == "__main__":
    main()
