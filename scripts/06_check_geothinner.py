#!/usr/bin/env python3
"""
06_check_geothinner.py — Valida la salida REAL de GeoThinneR con los mismos
criterios que thin.py: pares conservados a menos de d (debe ser 0) y registros
descartados sin ningun vecino conservado de su especie (agregables: si hay,
el conjunto no es maximal).

Lee bench/results/geothinner_mask_<label>_<metodo>_<d>km.csv.gz (filas
conservadas, indices de <label>_xyz_f64.npy) que escribe geothinner_baseline.R.

Uso: python 06_check_geothinner.py
"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

import thin

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

rows = []
for f in sorted(glob.glob(str(ROOT / "bench" / "results" / "geothinner_mask_*.csv.gz"))):
    m = re.search(r"geothinner_mask_(.+)_(local_kd_tree|kd_tree|k_estimation|brute)_([\d.]+)km", f)
    label, metodo, dkm = m.group(1), m.group(2), float(m.group(3))
    xyz = np.load(PROCESSED / f"{label}_xyz_f64.npy")
    otu_p = PROCESSED / f"{label}_otu.npy"
    otu = np.load(otu_p) if otu_p.exists() else np.zeros(len(xyz), np.int32)
    kept_rows = pd.read_csv(f).row.values
    malos = agregables = 0
    for k in np.unique(otu[otu >= 0]):
        idx = np.flatnonzero(otu == k)
        mask = np.isin(idx, kept_rows)
        ip, ix, _ = thin.neighbor_csr(xyz[idx], dkm * 1000)
        b, a = thin.validate(xyz[idx], mask, dkm * 1000, ip, ix)
        malos += b
        agregables += a
    rows.append({"label": label, "metodo": metodo, "d_km": dkm, "retenidos": len(kept_rows),
                 "pares_a_menos_de_d": malos, "agregables": agregables})
    print(rows[-1])
if rows:
    pd.DataFrame(rows).to_csv(ROOT / "bench" / "results" / "geothinner_validacion.csv", index=False)
