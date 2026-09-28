#!/usr/bin/env python3
"""
07_analyze_bench.py — De los CSV del benchmark a las tablas y la figura del paper.

Lee:
  bench/results/*.csv (excepto geothinner_*)   corridas en GPU (esquema de bench/README.md)
  bench/results/geothinner_*.csv               linea base GeoThinneR (W2)
  data/processed/<label>_cluster_stats.json    predictores por radio (03)
  data/processed/<label>_cpu_w1.json           linea base cKDTree (W1, 05)

Escribe en paper/:
  gpu_resumen.csv      mediana e IQR por (label, d, estructura, carga) + validez
  gpu_predictores.csv  Spearman de cada predictor contra ns/vecino, por estructura (W1)
  gpu_ganadores.csv    estructura mas rapida por (label, d) y su predictor
  gpu_aceleracion.csv  GPU (mejor estructura) contra CPU (W1) y contra GeoThinneR (W2)
  fig_gpu.png          ns por vecino verdadero vs hacinamiento medio (W1)

Regla: solo entran a las tablas las filas con valido == "ok". Las filas
"FALLA*" se listan y se excluyen (no se promedian con las validas). Las filas
"no_ejecutable" (p. ej. grilla densa que no cabe) se listan aparte con su
motivo: son un resultado, no un error.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "bench" / "results"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "paper"
PRED = ["morisita_2d", "hacinamiento_medio", "n_max", "p99", "ripley_norm"]


def iqr(x):
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def load_gpu(aceptar=()):
    OUT.mkdir(exist_ok=True)
    files = [f for f in glob.glob(str(RES / "*.csv")) if "geothinner" not in Path(f).name]
    if not files:
        raise SystemExit("No hay CSV de GPU en bench/results/")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["regla"] = df["regla"].fillna("").astype(str)
    # Casos confirmados como ruido de float32 por 08_check_w1.py: se aceptan y
    # quedan marcados en la columna ruido_f32 para declararlos.
    df["ruido_f32"] = False
    for lab, dkm in aceptar:
        m = (df.label == lab) & np.isclose(df.d_km, dkm) & (df.carga == "W1") \
            & (df.valido.astype(str) != "ok") & (df.valido.astype(str) != "no_ejecutable")
        df.loc[m, "valido"] = "ok"
        df.loc[m, "ruido_f32"] = True
        print(f"-- aceptado como ruido float32: {lab} d={dkm:g} km ({int(m.sum())} filas)")
    v = df.valido.astype(str)
    noej = df[v == "no_ejecutable"]
    if len(noej):
        print(f"-- {len(noej)} filas no ejecutables (se reportan, no se miden):")
        print(noej.groupby(["label", "d_km", "estructura", "carga"]).size().to_string())
        noej.to_csv(OUT / "gpu_no_ejecutables.csv", index=False)
    bad = df[(v != "ok") & (v != "no_ejecutable")]
    if len(bad):
        print(f"!! {len(bad)} corridas NO validaron; se excluyen:")
        print(bad.groupby(["label", "d_km", "estructura", "carga", "valido"]).size().to_string())
    return df[v == "ok"].copy()


def predictors():
    rows = []
    for f in glob.glob(str(PROC / "*_cluster_stats.json")):
        j = json.load(open(f))
        for r in j["por_radio"]:
            rows.append({"label": j["label"], "d_km": float(r["radio_km"]),
                         **{p: r.get(p, np.nan) for p in PRED}})
    return pd.DataFrame(rows, columns=["label", "d_km"] + PRED)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aceptar", nargs="*", default=[], metavar="LABEL:D_KM",
                    help="casos W1 que 08_check_w1.py confirmo como ruido de float32")
    args = ap.parse_args()
    aceptar = [(a.split(":")[0], float(a.split(":")[1])) for a in args.aceptar]
    g = load_gpu(aceptar)
    g["ns_por_vecino"] = g.query_ms * 1e6 / g.vecinos_total.replace(0, np.nan)
    keys = ["label", "N", "d_km", "estructura", "carga", "regla"]
    agg = g.groupby(keys).agg(
        build_ms=("build_ms", "median"), build_iqr=("build_ms", iqr),
        query_ms=("query_ms", "median"), query_iqr=("query_ms", iqr),
        ns_por_vecino=("ns_por_vecino", "median"), mem_pico_mb=("mem_pico_mb", "max"),
        rondas=("rondas", "max"), retenidos=("retenidos", "first"),
        vecinos_total=("vecinos_total", "first"), reps=("rep", "count"),
        discrepancias=("discrepancias", "max"), ruido_f32=("ruido_f32", "max")).reset_index()
    agg.to_csv(OUT / "gpu_resumen.csv", index=False)
    print(f"-> paper/gpu_resumen.csv ({len(agg)} filas)")

    w1 = agg[agg.carga == "W1"].merge(predictors(), on=["label", "d_km"], how="left")
    # acuerdo entre estructuras (validacion donde no hay referencia CPU)
    acu = w1.groupby(["label", "d_km"]).vecinos_total.nunique()
    if (acu > 1).any():
        print("!! Las estructuras NO coinciden en vecinos totales en:", acu[acu > 1].to_dict())

    rows = []
    for est, sub in w1.groupby("estructura"):
        for p in PRED:
            ok = sub[[p, "ns_por_vecino"]].dropna()
            if len(ok) >= 4:
                rows.append({"estructura": est, "predictor": p, "n": len(ok),
                             "spearman": spearmanr(ok[p], ok.ns_por_vecino).statistic})
    pd.DataFrame(rows).to_csv(OUT / "gpu_predictores.csv", index=False)
    print("-> paper/gpu_predictores.csv")

    win = (w1.loc[w1.groupby(["label", "d_km"]).query_ms.idxmin(),
                  ["label", "N", "d_km", "estructura", "query_ms"] + PRED])
    win.to_csv(OUT / "gpu_ganadores.csv", index=False)
    print("-> paper/gpu_ganadores.csv")

    acel = []
    for f in glob.glob(str(PROC / "*_cpu_w1.json")):
        label = Path(f).name.replace("_cpu_w1.json", "")
        for tag, c in json.load(open(f)).items():
            b = w1[(w1.label == label) & (np.isclose(w1.d_km, c["d_km"]))]
            if len(b):
                best = b.loc[b.query_ms.idxmin()]
                acel.append({"label": label, "d_km": c["d_km"], "carga": "W1",
                             "cpu": f"cKDTree {c['hilos']} hilos", "cpu_s": c["query_s"],
                             "gpu": best.estructura, "gpu_s": best.query_ms / 1000,
                             "aceleracion": c["query_s"] / (best.query_ms / 1000)})
    gt_files = [f for f in glob.glob(str(RES / "geothinner_*.csv")) if "mask" not in f
                and "validacion" not in f]
    if gt_files:
        gt = pd.concat([pd.read_csv(f) for f in gt_files])
        gt = gt[gt.error.fillna("") == ""].groupby(["label", "d_km", "metodo"]).seg.median().reset_index()
        w2 = agg[agg.carga == "W2"]
        for _, r in gt.iterrows():
            b = w2[(w2.label == r.label) & np.isclose(w2.d_km, r.d_km)]
            if len(b):
                best = b.loc[b.query_ms.idxmin()]
                acel.append({"label": r.label, "d_km": r.d_km, "carga": "W2",
                             "cpu": f"GeoThinneR {r.metodo}", "cpu_s": r.seg,
                             "gpu": best.estructura, "gpu_s": best.query_ms / 1000,
                             "aceleracion": r.seg / (best.query_ms / 1000)})
    pd.DataFrame(acel).to_csv(OUT / "gpu_aceleracion.csv", index=False)
    print("-> paper/gpu_aceleracion.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"KDTree": "#2a78d6", "LinearBVH": "#eb6834", "UniformGrid": "#1baf7a",
              "kdtree": "#2a78d6", "k-d tree": "#2a78d6", "lbvh": "#eb6834",
              "LBVH": "#eb6834", "grid": "#1baf7a", "grilla": "#1baf7a"}
    marks = ["o", "s", "^"]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for i, (est, sub) in enumerate(w1.dropna(subset=["hacinamiento_medio"]).groupby("estructura")):
        ax.scatter(sub.hacinamiento_medio, sub.ns_por_vecino, s=36, marker=marks[i % 3],
                   color=colors.get(est, None), label=est, edgecolor="white", linewidth=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Mean crowding (Lloyd) at query radius")
    ax.set_ylabel("Query time per true neighbor (ns)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_gpu.png", dpi=300)
    print("-> paper/fig_gpu.png")


if __name__ == "__main__":
    main()
