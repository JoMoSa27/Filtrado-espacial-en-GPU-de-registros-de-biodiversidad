#!/usr/bin/env python3
"""
00_explorar.py — Mira los datos antes de usarlos.

Genera un reporte HTML autocontenido que recorre, paso a paso, lo que le pasa
a un conjunto de registros desde el archivo crudo hasta lo que ve la grilla
en la GPU. Usa las mismas funciones (geo.py) que 02_prepare.py.

  Paso 1  Que trae el archivo           columnas, tipos, filas de ejemplo
  Paso 2  Donde estan los puntos        mapa y densidad
  Paso 3  Calidad y limpieza            embudo de descartes, incertidumbre, años
  Paso 4  Duplicados                    coordenadas repetidas
  Paso 5  Conversion a 3D               ejemplo resuelto con registros reales
  Paso 6  Lo que ve la grilla            conteos por celda vs. aleatorio
  Paso 7  Vecinos por punto             el trabajo real y su desbalance

Uso:
    python 00_explorar.py data/raw/geothinner/thunnus.rda --label tuna --radius-km 25
    python 00_explorar.py data/raw/gbif_cr_aves/occurrence.txt --label aves --radius-km 5

Salida: reports/<label>_explorar.html (abrir en el navegador; funciona sin
internet, las figuras van incrustadas).
"""
import argparse
import base64
import html
import io
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import geo

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# --- Estilo -----------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
DATA = "#2a78d6"      # los datos reales
REF = "#8f8d87"       # la referencia aleatoria (gris: es la linea base)
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "legend.frameon": False,
})


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fmt(n):
    return f"{n:,.0f}".replace(",", " ")


def table(df, max_rows=12):
    return df.head(max_rows).to_html(index=False, border=0, classes="t",
                                     float_format=lambda x: f"{x:,.4g}")


# --- Pasos ------------------------------------------------------------------
def paso1_archivo(raw):
    info = pd.DataFrame({
        "columna": raw.columns,
        "tipo": [str(t) for t in raw.dtypes],
        "no nulos": [f"{raw[c].notna().mean():.0%}" for c in raw.columns],
        "ejemplo": [str(raw[c].dropna().iloc[0])[:40] if raw[c].notna().any()
                    else "" for c in raw.columns],
    })
    return info


def paso2_mapa(df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    sub = df.sample(min(len(df), 200_000), random_state=0)
    ax[0].scatter(sub.lon, sub.lat, s=1.2, c=DATA, alpha=0.35, linewidths=0)
    ax[0].set_title(f"Cada registro es un punto ({fmt(len(sub))} dibujados)")
    hb = ax[1].hexbin(df.lon, df.lat, gridsize=90, cmap=SEQ, bins="log",
                      mincnt=1, linewidths=0.1, edgecolors=SURFACE)
    ax[1].set_title("Densidad: registros por hexagono (escala log)")
    cax = ax[1].inset_axes([1.03, 0.0, 0.035, 1.0])
    cb = fig.colorbar(hb, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label("registros", color=INK2)
    padx = max(0.5, 0.04 * (df.lon.max() - df.lon.min()))
    pady = max(0.5, 0.04 * (df.lat.max() - df.lat.min()))
    for a in ax:
        a.set_xlim(max(-180, df.lon.min() - padx), min(180, df.lon.max() + padx))
        a.set_ylim(max(-90, df.lat.min() - pady), min(90, df.lat.max() + pady))
        a.set_xlabel("longitud (°)")
        a.set_ylabel("latitud (°)")
        a.set_aspect("equal", adjustable="box")
    return fig_to_b64(fig)


def paso3_calidad(raw, pasos):
    embudo = pd.DataFrame(pasos)[["regla", "antes", "descartados", "despues", "razon"]]
    figs = []
    has_unc = "coordinateUncertaintyInMeters" in raw.columns
    has_year = "year" in raw.columns
    if has_unc or has_year:
        n = int(has_unc) + int(has_year)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 3.6), squeeze=False)
        k = 0
        if has_unc:
            a = axes[0, k]; k += 1
            u = pd.to_numeric(raw["coordinateUncertaintyInMeters"], errors="coerce")
            frac_nan = u.isna().mean()
            u = u.dropna()
            u = u[u > 0]
            if len(u):
                bins = np.logspace(np.log10(max(u.min(), 1)), np.log10(u.max()), 45)
                a.hist(u, bins=bins, color=DATA, rwidth=0.88)
                a.set_xscale("log")
                a.axvline(geo.MAX_UNCERTAINTY_M, color=INK, lw=1.2, ls="--")
                a.text(geo.MAX_UNCERTAINTY_M * 1.12, a.get_ylim()[1] * 0.92,
                       "corte: 1 km", color=INK, fontsize=9, va="top")
            a.set_title(f"Incertidumbre reportada ({frac_nan:.0%} sin dato)")
            a.set_xlabel("metros (log)")
            a.set_ylabel("registros")
        if has_year:
            a = axes[0, k]
            y = pd.to_numeric(raw["year"], errors="coerce").dropna()
            y = y[(y > 1700) & (y < 2100)]
            a.hist(y, bins=np.arange(y.min(), y.max() + 2) - 0.5,
                   color=DATA, rwidth=1.0)
            a.set_title("Año del registro")
            a.set_xlabel("año")
            a.set_ylabel("registros")
        figs.append(fig_to_b64(fig))
    return embudo, figs


def _lattice(key):
    """Fraccion de coordenadas que caen en multiplos exactos de 0.05°.
    Datos pesqueros y de atlas se reportan en mallas regulares: eso crea
    empates masivos en una coordenada, malo para las divisiones por mediana
    del k-d tree."""
    frac = lambda v: np.abs(v * 20 - np.round(v * 20)) < 1e-6
    return f"{(frac(key.lat.values) & frac(key.lon.values)).mean():.0%}"


def paso4_duplicados(raw_valid, decimals):
    key = raw_valid[["lat", "lon"]].round(decimals)
    mult = key.value_counts()
    top = mult.head(10).reset_index()
    top.columns = ["lat", "lon", "registros en esta coordenada"]

    # Distribucion de multiplicidad: cuantas coordenadas se repiten k veces
    dist = mult.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    if dist.index.max() <= 30:
        ax.bar(dist.index, dist.values, color=DATA, width=0.7)
        ax.set_xticks(dist.index)
        ax.set_xlabel("veces que se repite una misma coordenada")
    else:
        bins = np.unique(np.logspace(0, np.log10(dist.index.max() + 1), 40).astype(int))
        ax.hist(mult.values, bins=bins, color=DATA, rwidth=0.88)
        ax.set_xscale("log")
        ax.set_xlabel("veces que se repite una misma coordenada (log)")
    ax.set_yscale("log")
    ax.set_ylabel("coordenadas distintas (log)")
    ax.set_title("¿Cuántas veces aparece cada coordenada?")
    # Aporte de los duplicados a Sum n^2 (lo que paga la grilla)
    extra = int((mult.values.astype(np.float64) ** 2).sum() - len(mult))
    stats = {
        "registros validos": len(key),
        "coordenadas distintas": len(mult),
        "registros que comparten coordenada": int(len(key) - len(mult)),
        "maximo en una coordenada": int(mult.max()),
        "coordenadas en una malla regular": _lattice(key),
        "pares extra que solo aportan los duplicados": extra,
    }
    return top, fig_to_b64(fig), stats


def paso5_conversion(df, xyz):
    rng = np.random.default_rng(3)
    idx = rng.choice(len(df), min(3, len(df)), replace=False)
    ej = pd.DataFrame({
        "lat (°)": df.lat.values[idx], "lon (°)": df.lon.values[idx],
        "lat (rad)": np.radians(df.lat.values[idx]),
        "lon (rad)": np.radians(df.lon.values[idx]),
        "x (m)": xyz[idx, 0], "y (m)": xyz[idx, 1], "z (m)": xyz[idx, 2],
        "|xyz| (m)": np.linalg.norm(xyz[idx], axis=1),
    })
    km = np.array([1, 5, 10, 25, 50, 100, 500])
    ch = geo.chord_radius(km * 1000.0)
    radios = pd.DataFrame({
        "distancia en la superficie (km)": km,
        "radio de cuerda para la GPU (m)": ch,
        "diferencia (m)": km * 1000.0 - ch,
    })
    err, arc = geo.chord_check(df, xyz)
    centroid = xyz.mean(axis=0)
    prec = pd.DataFrame({
        "representacion": ["float32 sin centrar", "float32 centrado", "float64"],
        "resolucion cerca del dato (m)": [
            float(np.spacing(np.float32(np.abs(xyz).max()))),
            float(np.spacing(np.float32(np.abs(xyz - centroid).max()))),
            float(np.spacing(np.abs(xyz).max())),
        ],
    })
    return ej, radios, err, arc, prec


def paso6_grilla(df, ref, xyz, ref_xyz, radii_km, main_km):
    rows = []
    rect = (df.lat.min(), df.lat.max(), df.lon.min(), df.lon.max())
    for km in radii_km:
        cell = float(geo.chord_radius(km * 1000))
        c, Q = geo.cell_counts(xyz, cell)
        cr, _ = geo.cell_counts(ref_xyz, cell)
        m, mcv = geo.morisita_2d(df.lat.values, df.lon.values, km * 1000, rect)
        mr, _ = geo.morisita_2d(ref.lat.values, ref.lon.values, km * 1000, rect)
        rows.append({
            "celda (km)": km,
            "Morisita datos": m, "cv (5 orígenes)": mcv,
            "Morisita aleatorio": mr,
            "n_max datos": int(c.max()), "n_max aleatorio": int(cr.max()),
            "celdas 3D totales": f"{Q:.3g}",
            "celdas 3D ocupadas datos": len(c),
            "celdas 3D ocupadas aleatorio": len(cr),
        })
    cell = float(geo.chord_radius(main_km * 1000))
    c, _ = geo.cell_counts(xyz, cell)
    cr, _ = geo.cell_counts(ref_xyz, cell)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    hi = max(c.max(), cr.max())
    bins = np.unique(np.logspace(0, np.log10(hi + 1), 40).astype(int))
    ax.hist(cr, bins=bins, histtype="step", color=REF, lw=2, label="aleatorio uniforme")
    ax.hist(c, bins=bins, histtype="step", color=DATA, lw=2, label="datos reales")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("puntos en una celda (log)")
    ax.set_ylabel("celdas ocupadas (log)")
    ax.set_title(f"Puntos por celda, celda de {main_km:g} km")
    ax.legend(loc="upper right")
    return pd.DataFrame(rows), fig_to_b64(fig)


def paso7_vecinos(xyz, ref_xyz, main_km, max_queries):
    r = float(geo.chord_radius(main_km * 1000))
    rng = np.random.default_rng(1)

    def neigh(X):
        t = cKDTree(X)
        q = X if len(X) <= max_queries else X[rng.choice(len(X), max_queries, replace=False)]
        t0 = time.perf_counter()
        k = t.query_ball_point(q, r, return_length=True) - 1  # sin contarse a si mismo
        return k, time.perf_counter() - t0, len(q)

    k, t, nq = neigh(xyz)
    kr, tr, _ = neigh(ref_xyz)
    N = len(xyz)
    scale = N / nq
    pares = float(k.sum() * scale)
    stats = pd.DataFrame({
        "": ["media de vecinos", "mediana", "percentil 99", "maximo",
             "maximo / media", "eficiencia de warp (orden aleatorio)",
             "pares totales (estimado)",
             "memoria si se guardan los pares (GB, int32 x2)",
             "tiempo CPU 1 hilo, escalado a N (s)"],
        "datos reales": [k.mean(), np.median(k), np.percentile(k, 99), k.max(),
                         k.max() / max(k.mean(), 1e-9), geo.warp_efficiency(k),
                         pares, pares * 8 / 1e9, t * scale],
        "aleatorio uniforme": [kr.mean(), np.median(kr), np.percentile(kr, 99), kr.max(),
                               kr.max() / max(kr.mean(), 1e-9), geo.warp_efficiency(kr),
                               float(kr.sum() * scale), kr.sum() * scale * 8 / 1e9,
                               tr * scale],
    })
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    hi = max(k.max(), kr.max(), 1)
    bins = np.unique(np.concatenate([[0], np.logspace(0, np.log10(hi + 1), 45)]).astype(int))
    ax.hist(kr + 1, bins=bins + 1, histtype="step", color=REF, lw=2, label="aleatorio uniforme")
    ax.hist(k + 1, bins=bins + 1, histtype="step", color=DATA, lw=2, label="datos reales")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"vecinos dentro de {main_km:g} km, +1 (log)")
    ax.set_ylabel("puntos (log)")
    ax.set_title("¿Cuántos vecinos tiene cada punto?")
    ax.legend(loc="upper right")
    return stats, fig_to_b64(fig)


# --- HTML -------------------------------------------------------------------
CSS = """
:root{--s:#fcfcfb;--i:#0b0b0b;--i2:#52514e;--g:#e4e3df;--a:#2a78d6;--card:#f4f3f0}
body{background:var(--s);color:var(--i);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;
max-width:1060px;margin:0 auto;padding:28px 16px 80px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:44px 0 6px;padding-top:18px;border-top:1px solid var(--g)}
.sub{color:var(--i2);margin:0 0 20px}.k{display:inline-block;background:var(--a);color:#fff;border-radius:4px;
padding:1px 8px;font-size:12px;font-weight:600;margin-right:8px;vertical-align:2px}
.box{background:var(--card);border-radius:8px;padding:12px 16px;margin:12px 0}
.box b{display:block;font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--i2);margin-bottom:2px}
img{max-width:100%;height:auto;display:block;margin:10px 0}
.tw{overflow-x:auto}table.t{border-collapse:collapse;font-size:13px;margin:8px 0;font-variant-numeric:tabular-nums}
table.t th{text-align:left;color:var(--i2);font-weight:600;border-bottom:1px solid var(--g);padding:5px 12px 5px 0}
table.t td{border-bottom:1px solid var(--g);padding:5px 12px 5px 0}
.kpi{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}.kpi div{background:var(--card);border-radius:8px;padding:10px 14px;min-width:150px}
.kpi span{display:block;font-size:22px;font-weight:700}.kpi small{color:var(--i2)}
code{background:var(--card);padding:1px 5px;border-radius:4px;font-size:13px}
"""


def box(title, text):
    return f'<div class="box"><b>{title}</b>{text}</div>'


def build_html(label, src, raw, df, info, map_b64, embudo, cal_figs, dup_top,
               dup_b64, dup_stats, ej, radios, err, arc, prec, grid_tab,
               grid_b64, vec_tab, vec_b64, main_km, decimals):
    N = len(df)
    kpis = "".join(f"<div><span>{v}</span><small>{k}</small></div>" for k, v in [
        ("registros en el archivo", fmt(len(raw))),
        ("registros utilizables", fmt(N)),
        ("descartados", f"{1 - N / max(len(raw), 1):.1%}"),
        ("radio de estudio", f"{main_km:g} km"),
    ])
    g = grid_tab.iloc[[0]].to_dict("records")[0] if len(grid_tab) else {}
    dup_kpi = "".join(f"<div><span>{v if isinstance(v, str) else fmt(v)}</span><small>{k}</small></div>"
                      for k, v in dup_stats.items())

    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exploración {html.escape(label)}</title><style>{CSS}</style></head><body>
<h1>Exploración de datos: {html.escape(label)}</h1>
<p class="sub">Fuente: <code>{html.escape(str(src))}</code> · Generado {time.strftime('%Y-%m-%d %H:%M')}</p>
<div class="kpi">{kpis}</div>

<h2><span class="k">1</span>Qué trae el archivo</h2>
{box("Qué estás viendo", "Cada columna del archivo crudo, su tipo, qué fracción tiene dato y un valor de ejemplo. "
"De todo esto la GPU solo usará dos columnas: latitud y longitud. El resto sirve para decidir qué registros son confiables.")}
<div class="tw">{table(info, 60)}</div>

<h2><span class="k">2</span>Dónde están los puntos</h2>
{map_b64 and f'<img src="data:image/png;base64,{map_b64}">'}
{box("Qué estás viendo", "A la izquierda, cada registro como un punto. A la derecha, cuántos registros caen en cada hexágono, en escala logarítmica: "
"un azul oscuro puede tener cientos o miles de veces más registros que un azul claro.")}
{box("Por qué importa", "Si los organismos se muestrearan de forma pareja, el mapa de densidad sería un color casi uniforme. "
"Los cúmulos intensos separados por zonas vacías son el <i>sesgo de muestreo</i>: reflejan dónde observa la gente, no solo dónde vive la especie. "
"Es exactamente la distribución con vacíos que en el paper previo degradó la grilla.")}

<h2><span class="k">3</span>Calidad y limpieza</h2>
{box("Qué estás viendo", "El embudo: cada regla se aplica en orden sobre lo que sobrevivió a la anterior. "
"Así sabes cuánto pierde cada decisión y puedes justificarla en la sección de métodos.")}
<div class="tw">{table(embudo, 20)}</div>
{"".join(f'<img src="data:image/png;base64,{b}">' for b in cal_figs)}
{box("Cómo leer la incertidumbre", "Es el radio de error que el recolector declaró para la coordenada. Los registros a la derecha de la línea "
"tienen un error mayor que el radio más chico que vas a consultar, así que no pueden sostener un filtrado a 1 km. "
"Los que no declaran incertidumbre se conservan, porque en GBIF son muchos y descartarlos vaciaría el conjunto.") if "coordinateUncertaintyInMeters" in raw.columns else box("Nota", "Este archivo no trae incertidumbre de coordenada, así que esa regla no se aplicó. En las descargas de GBIF sí viene.")}

<h2><span class="k">4</span>Duplicados</h2>
<div class="kpi">{dup_kpi}</div>
<img src="data:image/png;base64,{dup_b64}">
<div class="tw">{table(dup_top, 10)}</div>
{box("Qué estás viendo", f"Aquí se cuentan registros que caen en la misma coordenada (redondeada a {decimals} decimales, ≈1 m), SIN importar la especie: es lo que ve la estructura espacial, que no sabe de especies. "
"Ojo: la limpieza solo elimina como duplicado la misma especie en la misma coordenada; dos especies distintas en un mismo punto son registros válidos y se conservan. "
"La barra en x=1 son las coordenadas que aparecen una sola vez; todo lo que está a la derecha son repeticiones. "
"La tabla muestra las coordenadas más repetidas: suelen ser centroides de país, sedes de instituciones o puntos de observación fijos.")}
{box("Por qué importa", "Una coordenada con k copias genera k² pares de vecinos por sí sola. Por eso los duplicados inflan el costo de la grilla "
"y el índice de Morisita más que cualquier otra cosa. El paper debe reportar resultados con y sin ellos, "
"porque un revisor va a sospechar que la aceleración viene de ahí.")}

<h2><span class="k">5</span>Conversión a 3D</h2>
{box("Qué se hace", "Latitud y longitud son ángulos, no distancias: un grado de longitud mide 111 km en el ecuador y 0 km en los polos. "
"Para que una búsqueda por radio euclidiana funcione, cada registro se proyecta a un punto (x, y, z) en metros sobre una esfera de radio "
"R = 6 371 008,8 m:<br><code>x = R·cos(lat)·cos(lon)</code> · <code>y = R·cos(lat)·sin(lon)</code> · <code>z = R·sin(lat)</code>")}
<p><b>Ejemplo resuelto con tres registros reales.</b> La última columna comprueba que todos quedan exactamente sobre la esfera.</p>
<div class="tw">{table(ej)}</div>
{box("El truco del radio de cuerda", "Entre dos puntos sobre la esfera hay dos distancias: el arco por la superficie (la que importa en ecología) "
"y la cuerda recta a través de la Tierra (la que calcula la GPU). La cuerda siempre es un poco más corta, pero crece junto con el arco. "
"Si consultas con radio <code>2R·sin(d/2R)</code> obtienes exactamente los mismos vecinos que a distancia geodésica d. "
"No hace falta un kernel nuevo: sirve la métrica cartesiana que ya tienes.")}
<div class="tw">{table(radios)}</div>
<p><b>Verificación con los datos:</b> en {fmt(len(err))} pares aleatorios, separados entre {arc.min()/1000:,.1f} y {arc.max()/1000:,.0f} km,
la diferencia máxima entre la distancia por cuerda y la geodésica fue de <b>{err.max():.2e} m</b>. Es error de redondeo de punto flotante: la equivalencia se cumple.</p>
{box("Precisión numérica", "Las coordenadas rondan los 6,4 millones de metros. En float32 solo hay unos 7 dígitos significativos, así que "
"la resolución cerca de ese valor es de medio metro. " + (
"Aquí restar el centroide sí recupera precisión, porque los datos cubren una región: el float32 centrado es seguro para los kernels."
if prec.iloc[1, 1] < prec.iloc[0, 1] else
"<b>Aquí centrar no ayuda</b>: los datos cubren medio planeta, así que el centroide cae dentro de la Tierra y las coordenadas centradas siguen siendo del orden de R. "
"Para conjuntos globales, valida en float64; con 0,5–1 m de resolución frente a radios de kilómetros, float32 sigue siendo aceptable para medir tiempos."))}
<div class="tw">{table(prec)}</div>

<h2><span class="k">6</span>Lo que ve la grilla</h2>
<img src="data:image/png;base64,{grid_b64}">
{box("Qué estás viendo", "Se reparte el espacio en celdas cúbicas del tamaño del radio de consulta, igual que tu grilla en GPU, y se cuenta cuántos puntos caen en cada celda. "
"La línea gris es la referencia: los mismos N puntos repartidos al azar en el mismo rectángulo. La azul son los datos reales.")}
{box("Por qué importa", "En la referencia aleatoria casi todas las celdas tienen más o menos los mismos puntos: la curva gris es estrecha. "
"En los datos reales la curva azul se estira hacia la derecha: unas pocas celdas concentran cientos o miles de puntos. "
"Cada punto dentro de una celda compara contra todos los de su celda y las vecinas, así que el trabajo crece con la suma de los cuadrados de los conteos. "
"Y las celdas vacías también se pagan: ocupan memoria y hay que recorrerlas.")}
<div class="tw">{table(grid_tab)}</div>
{box("Cómo leer la tabla", "<b>Morisita</b> se calcula sobre cuadrantes en el plano, con una proyección de igual área (Lambert azimutal), y con Q = área del rectángulo de estudio ÷ área del cuadrante. "
"Así ≈ 1 significa aleatorio y valores mucho mayores que 1, agrupado: la columna de aleatorio debe salir cerca de 1, y eso valida el cálculo. "
"El <b>cv</b> es la variación al mover el origen de la malla 5 veces: si es alto, el índice es frágil para ese conjunto. "
"El valor cambia con el tamaño de celda, así que no existe «el Morisita del conjunto».<br><br>"
"Las <b>celdas 3D</b> son otra cosa: lo que paga la grilla en GPU. Su total incluye el grosor del casquete esférico y, para datos globales, el interior de la Tierra. "
"Por eso el índice no se calcula en 3D: saldría inflado incluso para datos aleatorios.")}

<h2><span class="k">7</span>Vecinos por punto: el trabajo real</h2>
<img src="data:image/png;base64,{vec_b64}">
<div class="tw">{table(vec_tab)}</div>
{box("Qué estás viendo", f"Para cada punto, cuántos otros registros hay a menos de {main_km:g} km. Esto es exactamente lo que calcula la búsqueda por radio en el filtrado.")}
{box("Por qué importa para la GPU", "Hay dos efectos distintos y conviene no mezclarlos. "
"<b>Más trabajo:</b> la media de vecinos. Con el mismo N, los datos agrupados tienen muchos más vecinos que los aleatorios, así que cualquier estructura hace más trabajo. "
"Por eso el benchmark se normaliza por vecino encontrado. "
"<b>Trabajo peor repartido:</b> la eficiencia de warp. En una GPU, 32 hilos avanzan juntos: el warp tarda lo que su hilo más cargado. "
"Una eficiencia de 0,5 significa que la mitad del tiempo de cómputo se va en esperar. Esto sí depende de la estructura y del orden de los puntos. "
"La fila de memoria explica por qué spThin se quedó sin RAM: guardar todos los pares crece mucho más rápido que N.")}
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--label", required=True)
    ap.add_argument("--lat-col")
    ap.add_argument("--lon-col")
    ap.add_argument("--radius-km", type=float, default=5.0,
                    help="radio principal para los pasos 6 y 7")
    ap.add_argument("--radii-km", type=float, nargs="+",
                    help="radios para la tabla de la grilla (por defecto 1 5 10 o 10 25 50)")
    ap.add_argument("--round-decimals", type=int, default=5)
    ap.add_argument("--max-queries", type=int, default=50_000,
                    help="puntos consultados en el paso 7 (se escala a N)")
    args = ap.parse_args()

    t0 = time.time()
    print("[1/7] leyendo ...")
    raw = geo.read_any(args.input, args.lat_col, args.lon_col)
    info = paso1_archivo(raw)

    print("[3/7] limpiando ...")
    df, pasos = geo.clean(raw, round_decimals=args.round_decimals)
    xyz = geo.to_cartesian(df.lat.values, df.lon.values)

    print("[2/7] mapa ...")
    map_b64 = paso2_mapa(df)
    embudo, cal_figs = paso3_calidad(raw, pasos)

    print("[4/7] duplicados ...")
    valid, _ = geo.clean(raw, keep_duplicates=True, round_decimals=args.round_decimals)
    dup_top, dup_b64, dup_stats = paso4_duplicados(valid, args.round_decimals)

    print("[5/7] conversion ...")
    ej, radios, err, arc, prec = paso5_conversion(df, xyz)

    print("[6/7] grilla ...")
    ref = geo.uniform_reference(df, len(df))
    ref_xyz = geo.to_cartesian(ref.lat.values, ref.lon.values)
    radii = args.radii_km or ([1, 5, 10] if args.radius_km <= 10 else [10, 25, 50])
    grid_tab, grid_b64 = paso6_grilla(df, ref, xyz, ref_xyz, radii, args.radius_km)

    print("[7/7] vecinos ...")
    vec_tab, vec_b64 = paso7_vecinos(xyz, ref_xyz, args.radius_km, args.max_queries)

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"{args.label}_explorar.html"
    out.write_text(build_html(args.label, args.input, raw, df, info, map_b64,
                              embudo, cal_figs, dup_top, dup_b64, dup_stats, ej,
                              radios, err, arc, prec, grid_tab, grid_b64,
                              vec_tab, vec_b64, args.radius_km,
                              args.round_decimals), encoding="utf-8")
    print(f"\nListo en {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
