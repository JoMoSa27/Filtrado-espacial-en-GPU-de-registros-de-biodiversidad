#!/usr/bin/env python3
"""
10_c4_distancias.py — Barrido de la distancia de filtrado (C4, revisor 1).

Para cada taxón nacional y cada d, con las máscaras EXACTAS del filtrado por
especie que escribe 05_references.py --w2 (regla mindeg, semilla 0):

  retención      registros conservados / registros limpios (con especie u OTU),
                 y la mediana por especie de esa fracción (especies con >= 2)
  concentración  reparto de los registros conservados en una malla de igual
                 área de lado FIJO (5 km por defecto, la del paper): fracción en
                 el 1 % de celdas más muestreadas y celdas que reúnen la mitad.
                 El lado es fijo a propósito: si la malla cambiara con d, la
                 regla de medición cambiaría junto con lo que se mide.
  rondas         máximo de rondas paralelas (de *_rondas.json)

La referencia "antes" son todos los registros descargados (<label>_dup), como
en la Figura 1.

Salidas: bench/results/c4_distancias_celda<lado>km.csv y paper/fig_c4_distancias_celda<lado>km.png
(el lado de la malla va en el nombre: se corre con varios lados para ver si
el mínimo de concentración depende de la regla de medición).

Uso (desde la raíz del repo, después de generar las máscaras):
  python3 scripts/10_c4_distancias.py --radii-km 1 2 5 10 25 50
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig_mapa import concentracion, latlon

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TAXA = {"aves_2018": "Birds", "amphibia": "Amphibians", "plantae": "Plants", "insecta_2018": "Insects"}
# Paleta categórica validada (dataviz), en orden fijo.
COLORES = {"aves_2018": "#2a78d6", "amphibia": "#eb6834", "plantae": "#1baf7a", "insecta_2018": "#eda100"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radii-km", type=float, nargs="+", default=[1, 2, 5, 10, 25, 50])
    ap.add_argument("--regla", default="mindeg")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--cell-km", type=float, default=5.0)
    args = ap.parse_args()
    cell_m = args.cell_km * 1000

    filas = []
    for label, nombre in TAXA.items():
        lat_dup, lon_dup = latlon(f"{label}_dup")
        antes = concentracion(lat_dup, lon_dup, cell_m)
        lat, lon = latlon(label)
        otu = np.load(PROC / f"{label}_otu.npy")
        validos = otu >= 0
        for km in args.radii_km:
            tag = f"{km:g}km_s{args.semilla}"
            p = PROC / f"{label}_thin_{args.regla}_{tag}.npy"
            if not p.exists():
                print(f"falta {p.name}: se omite")
                continue
            mask = np.load(p)
            if len(mask) != len(lat):
                raise SystemExit(f"{p.name}: la máscara no coincide con {label}")
            despues = concentracion(lat[mask], lon[mask], cell_m)
            # Fracción retenida por especie (solo especies con al menos 2 registros).
            s = pd.DataFrame({"otu": otu[validos], "k": mask[validos]}).groupby("otu").k.agg(["sum", "size"])
            s = s[s["size"] >= 2]
            rj = PROC / f"{label}_thin_{args.regla}_{tag}_rondas.json"
            rondas = json.loads(rj.read_text())["rondas_max"] if rj.exists() else np.nan
            filas.append({
                "label": label, "taxon": nombre, "d_km": km, "regla": args.regla, "celda_km": args.cell_km,
                "registros_limpios": int(validos.sum()), "retenidos": int(mask.sum()),
                "fraccion_retenida": mask.sum() / validos.sum(),
                "mediana_retenida_por_especie": float((s["sum"] / s["size"]).median()),
                "especies_2_o_mas": int(len(s)), "rondas_max": rondas,
                "antes_frac_1pct": antes["frac_en_1pct_celdas"], "despues_frac_1pct": despues["frac_en_1pct_celdas"],
                "antes_celdas_mitad": antes["celdas_mitad"], "despues_celdas_mitad": despues["celdas_mitad"],
                "despues_celdas_ocupadas": despues["celdas_ocupadas"], "antes_celdas_ocupadas": antes["celdas_ocupadas"],
            })
            f = filas[-1]
            print(f"{nombre:>10} d={km:>4g} km: retiene {100 * f['fraccion_retenida']:5.1f} % "
                  f"({f['retenidos']:,}); 1 % de celdas {100 * f['antes_frac_1pct']:.1f} -> "
                  f"{100 * f['despues_frac_1pct']:.1f} %; rondas {rondas}")
    out = pd.DataFrame(filas)
    dest = ROOT / "bench" / "results" / f"c4_distancias_celda{args.cell_km:g}km.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"-> {dest.relative_to(ROOT)}")

    # Figura: retención y concentración contra d, una línea por taxón, etiquetas directas.
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#52514e"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.9))
    fig.subplots_adjust(left=0.1, right=0.86, top=0.88, bottom=0.17, wspace=0.42)
    for label, g in out.groupby("label", sort=False):
        g = g.sort_values("d_km")
        c = COLORES[label]
        a.plot(g.d_km, 100 * g.fraccion_retenida, color=c, lw=2, marker="o", ms=4)
        b.plot(g.d_km, 100 * g.despues_frac_1pct, color=c, lw=2, marker="o", ms=4)
        b.axhline(100 * g.antes_frac_1pct.iloc[0], color=c, lw=1, ls=":")
        for ax, y in ((a, 100 * g.fraccion_retenida.iloc[-1]), (b, 100 * g.despues_frac_1pct.iloc[-1])):
            ax.annotate(TAXA[label], (g.d_km.iloc[-1], y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color="#0b0b0b")
    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_xticks(args.radii_km, [f"{k:g}" for k in args.radii_km])
        ax.set_xlabel("Thinning distance d (km)")
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    a.set_ylabel("Records kept (%)")
    a.set_title("(a) Retention", loc="left", fontsize=9)
    b.set_ylabel(f"Records in top 1 % of {args.cell_km:g} km cells (%)")
    b.set_title("(b) Concentration (dotted: all records)", loc="left", fontsize=9)
    fig_out = ROOT / "paper" / f"fig_c4_distancias_celda{args.cell_km:g}km.png"
    fig.savefig(fig_out, dpi=300)
    print(f"-> {fig_out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
