#!/usr/bin/env python3
"""
12_c2_resumen.py — Resumen del piloto SDM (C2).

Lee bench/results/c2_sdm/*.csv (bench/sdm_piloto.R) y, por especie y
tratamiento, promedia las métricas de los 4 bloques espaciales; para los
tratamientos filtrados, además, la media y el rango sobre las 10 semillas.
Expresa cada tratamiento como diferencia contra "sin filtrar" de la MISMA
especie (mismos registros de prueba), que es la comparación que importa.

Salidas: bench/results/c2_sdm_resumen.csv y paper/fig_c2_sdm.png
Uso: python3 scripts/12_c2_resumen.py
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "bench" / "results"
METRICAS = ["auc_prueba", "boyce", "or10", "auc_dif", "rho_esfuerzo"]


def main():
    files = sorted(glob.glob(str(RES / "c2_sdm" / "*.csv")))
    if not files:
        raise SystemExit("no hay resultados en bench/results/c2_sdm/")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    errores = d[d.error.fillna("") != ""]
    if len(errores):
        print(f"aviso: {len(errores)} ajustes sin resultado:\n"
              f"{errores.groupby(['archivo', 'tratamiento', 'error']).size()}")
    d["trat"] = d.tratamiento.where(d.tratamiento != "filtrado", "filtrado_" + d.d_km.map("{:g}km".format))
    cv = d[d.bloque > 0].groupby(["archivo", "trat", "semilla"], dropna=False)[METRICAS[:-1]].mean()
    full = d[d.bloque == 0].groupby(["archivo", "trat", "semilla"], dropna=False)["rho_esfuerzo"].mean()
    por_semilla = cv.join(full).reset_index()
    resumen = (por_semilla.groupby(["archivo", "trat"])[METRICAS]
               .agg(["mean", "min", "max"]))
    resumen.columns = [f"{m}_{s}" for m, s in resumen.columns]
    resumen = resumen.reset_index()
    base = resumen[resumen.trat == "sin_filtrar"].set_index("archivo")
    for m in METRICAS:
        resumen[f"delta_{m}"] = resumen[f"{m}_mean"] - resumen.archivo.map(base[f"{m}_mean"])
    n = d[(d.bloque == 0)].groupby(["archivo", "trat"]).n_entrena.median().rename("n_entrena")
    resumen = resumen.merge(n.reset_index(), how="left")
    dest = RES / "c2_sdm_resumen.csv"
    resumen.to_csv(dest, index=False)
    cols = ["archivo", "trat", "n_entrena"] + [f"{m}_mean" for m in METRICAS]
    print(resumen[cols].round(3).to_string(index=False))
    print(f"-> {dest.relative_to(ROOT)}")

    # Figura: diferencia contra sin filtrar, por métrica, una fila de puntos por especie.
    orden = ["grupo_objetivo", "filtrado_1km", "filtrado_5km", "filtrado_10km"]
    nombres = {"grupo_objetivo": "target-group\nbackground", "filtrado_1km": "thinned\n1 km",
               "filtrado_5km": "thinned\n5 km", "filtrado_10km": "thinned\n10 km"}
    titulos = {"auc_prueba": "Δ test AUC", "boyce": "Δ Boyce index", "rho_esfuerzo": "Δ ρ with effort"}
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.6), sharey=True)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.86, bottom=0.17, wspace=0.12)
    r = resumen[resumen.trat.isin(orden)]
    for ax, m in zip(axes, titulos):
        for i, t in enumerate(orden):
            v = r[r.trat == t][f"delta_{m}"].dropna()
            ax.scatter(v, [i] * len(v), s=14, color="#2a78d6", alpha=0.7, edgecolor="none")
            if len(v):
                ax.plot([v.median()] * 2, [i - 0.3, i + 0.3], color="#0b0b0b", lw=1.5)
        ax.axvline(0, color="#52514e", lw=0.8)
        ax.set_title(titulos[m], fontsize=9, loc="left")
        ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    axes[0].set_yticks(range(len(orden)), [nombres[t] for t in orden])
    axes[0].invert_yaxis()
    out = ROOT / "paper" / "fig_c2_sdm.png"
    fig.savefig(out, dpi=300)
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
