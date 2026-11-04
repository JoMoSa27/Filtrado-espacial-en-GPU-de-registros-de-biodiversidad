#!/usr/bin/env python3
"""
09_c1_geothinner.py — Tabla de la comparación con el paquete GeoThinneR (C1).

Junta, para tortuga_gt y atun_gt (y amphibia si existe), solo corridas
guardadas en disco:

  GeoThinneR real   bench/results/geothinner_<label>_<metodo>_<d>km_s<semilla>.csv
                    (geothinner_baseline.R, una fila por corrida: tiempo de
                    pared de thin_points y registros retenidos, mejor de 10
                    intentos)
  memoria pico      bench/results/geothinner_memoria.csv (run_geothinner.sh)
  maximalidad       bench/results/geothinner_validacion.csv (06_check_geothinner.py,
                    sobre la máscara REAL de la semilla 1): pares a menos de d
                    (debe ser 0) y registros descartados que se podrían agregar
  ThinGPU           bench/results/*_NVIDIA_*_w2_<label>.csv (bench/w2): mediana
                    de build_ms + query_ms; tiempo en GPU, sin copiar los datos
                    a la GPU (hay que decirlo en el texto)
  reimplementación  data/processed/<label>_por_especie.csv (thin.py, regla de
                    GeoThinneR replicada): para ver si el paquete se comporta
                    como nuestra copia

Escribe bench/results/c1_geothinner_resumen.csv e imprime la tabla.

Uso: python scripts/09_c1_geothinner.py
"""
import glob
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "bench" / "results"
PROCESSED = ROOT / "data" / "processed"


def gpu(label):
    """Última corrida W2 de la GPU para el conjunto, la estructura más rápida por (d, regla)."""
    files = sorted(glob.glob(str(RES / f"*_NVIDIA_*_w2_{label}.csv")))
    if not files:
        return pd.DataFrame()
    d = pd.read_csv(files[-1])
    if (d.valido != "ok").any():
        raise SystemExit(f"{files[-1]}: hay corridas de GPU no válidas")
    d["t_ms"] = d.build_ms + d.query_ms
    g = (d.groupby(["d_km", "regla", "estructura"])
          .agg(t_ms=("t_ms", "median"), retenidos=("retenidos", "first"), rondas=("rondas", "max"))
          .reset_index())
    best = g.loc[g.groupby(["d_km", "regla"]).t_ms.idxmin()]
    best = best.pivot(index="d_km", columns="regla", values=["t_ms", "retenidos", "estructura", "rondas"])
    best.columns = [f"gpu_{a}_{b}" for a, b in best.columns]
    return best.reset_index()


def geothinner():
    files = [f for f in glob.glob(str(RES / "geothinner_*_s*.csv"))
             if not re.search(r"geothinner_(mask|memoria|validacion)", f)]
    if not files:
        raise SystemExit("no hay corridas de GeoThinneR en bench/results/ (correr bench/run_geothinner.sh)")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    fallidas = d[d.error.fillna("") != ""]
    if len(fallidas):
        print(f"aviso: {len(fallidas)} corridas con error:\n{fallidas[['label', 'metodo', 'd_km', 'rep', 'error']]}")
    d = d[d.error.fillna("") == ""]
    # Repeticiones por conjunto: 10 en la tortuga; 5 en el atún (cada corrida
    # tarda ~7 min). Se usan las MISMAS semillas en todos sus casos (1..5), aunque
    # algunos casos hayan corrido más, para que todas las medianas sean comparables.
    MAX_REPS = {"atun_gt": 5}
    d = d[d.apply(lambda r: r.rep <= MAX_REPS.get(r.label, 10**9), axis=1)]
    g = (d.groupby(["label", "metodo", "n_cores", "d_km"])
          .agg(version=("geothinner", "first"), reps=("rep", "nunique"),
               gt_s_mediana=("seg", "median"), gt_s_q1=("seg", lambda s: s.quantile(.25)),
               gt_s_q3=("seg", lambda s: s.quantile(.75)), gt_ret_mediana=("retenidos", "median"),
               gt_ret_min=("retenidos", "min"), gt_ret_max=("retenidos", "max"))
          .reset_index())
    mem = RES / "geothinner_memoria.csv"
    if mem.exists():
        m = pd.read_csv(mem)
        m = m[m.salida == 0].groupby(["label", "metodo", "n_cores", "d_km"]).max_rss_kb.max().div(1024).rename("gt_mem_mb")
        g = g.merge(m.reset_index(), how="left")
    val = RES / "geothinner_validacion.csv"
    if val.exists():
        v = pd.read_csv(val).rename(columns={"pares_a_menos_de_d": "gt_pares_bajo_d", "agregables": "gt_agregables"})
        g = g.merge(v[["label", "metodo", "d_km", "gt_pares_bajo_d", "gt_agregables"]], how="left")
    return g


def main():
    gt = geothinner()
    filas = []
    for label, sub in gt.groupby("label"):
        ref_p = PROCESSED / f"{label}_por_especie.csv"
        ref = (pd.read_csv(ref_p)[["d_km", "ret_maxdeg_mejor", "ret_maxdeg_media", "maxdeg_agregables",
                                   "ret_mindeg_mejor"]].rename(columns=lambda c: c if c == "d_km" else "ref_" + c)
               if ref_p.exists() else pd.DataFrame({"d_km": sub.d_km.unique()}))
        t = sub.merge(ref, on="d_km", how="left").merge(gpu(label), on="d_km", how="left")
        if "gpu_t_ms_mindeg" in t:
            # Aceleración frente a la configuración más rápida de GeoThinneR en ese d.
            rapido = t.groupby("d_km").gt_s_mediana.transform("min")
            t["aceleracion_mindeg"] = rapido / (t.gpu_t_ms_mindeg / 1000)
            t["mindeg_vs_gt_pct"] = 100 * (t.gpu_retenidos_mindeg / t.gt_ret_mediana - 1)
        filas.append(t)
    out = pd.concat(filas, ignore_index=True)
    dest = RES / "c1_geothinner_resumen.csv"
    out.to_csv(dest, index=False)
    cols = [c for c in ["label", "metodo", "n_cores", "d_km", "reps", "gt_s_mediana", "gt_mem_mb", "gt_ret_mediana",
                        "ref_ret_maxdeg_media", "gt_pares_bajo_d", "gt_agregables", "gpu_t_ms_mindeg",
                        "gpu_retenidos_mindeg", "aceleracion_mindeg", "mindeg_vs_gt_pct"] if c in out]
    print(out[cols].round(2).to_string(index=False))
    print(f"-> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
