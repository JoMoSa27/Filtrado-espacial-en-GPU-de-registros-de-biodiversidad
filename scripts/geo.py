"""
geo.py — Logica compartida de lectura, limpieza y conversion.

La usan 00_explorar.py (para MOSTRAR cada paso) y 02_prepare.py (para
GENERAR los arreglos). Al vivir en un solo lugar, lo que ves en el reporte
de exploracion es exactamente lo que le pasa a los datos que entran a la GPU.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Radio medio de la Tierra (IUGG), en metros.
R_EARTH = 6_371_008.8

# Umbral de incertidumbre: un registro con +-50 km no sostiene un filtrado a 1 km.
MAX_UNCERTAINTY_M = 1000.0

# Base de registro que se descarta: sus coordenadas son las de la institucion
# (museo, zoologico, jardin botanico), no las del organismo.
BAD_BASIS = {"FOSSIL_SPECIMEN", "LIVING_SPECIMEN"}


# ----------------------------------------------------------------------------
# Lectura
# ----------------------------------------------------------------------------
LAT_NAMES = ("decimallatitude", "lat", "latitude", "y")
LON_NAMES = ("decimallongitude", "lon", "lng", "longitude", "x")
# Columnas de GBIF que usa el pipeline (el resto se ignora al leer).
KEEP_COLS_LOWER = set(LAT_NAMES) | set(LON_NAMES) | {
    "gbifid", "datasetkey", "species", "scientificname", "year",
    "basisofrecord", "coordinateuncertaintyinmeters", "institutioncode",
    "occurrencestatus", "issue", "taxonrank", "verbatimscientificname"}

# Rectangulo de estudio COMUN para los conjuntos de Costa Rica (incluye la Isla
# del Coco). Morisita y Ripley dependen del area de estudio; con un rectangulo
# por conjunto, los taxones no son comparables (Q distinto). Se fija uno solo.
CR_RECT = (5.3, 11.3, -87.2, -82.5)

# BIN de BOLD (Ratnasingham y Hebert 2013): unidad operativa cuando el registro
# no tiene especie (p. ej. muestras de trampas Malaise identificadas por ADN).
_BIN_RE = r"(BOLD:[A-Z]{3}\d{4})"


def read_any(path, lat_col=None, lon_col=None):
    """Lee GBIF SIMPLE_CSV (TSV aunque diga .csv), CSV, parquet o .rda de R.

    Devuelve un DataFrame con las columnas de coordenadas renombradas a
    'lat' y 'lon'. El resto de columnas se conserva tal cual.
    """
    path = Path(path)
    suf = path.suffix.lower()
    if suf in (".rda", ".rdata", ".rds"):
        import pyreadr  # pip install pyreadr
        res = pyreadr.read_r(str(path))
        df = next(iter(res.values()))
    elif suf == ".parquet":
        df = pd.read_parquet(path)
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            head = f.readline()
        sep = "\t" if "\t" in head else ","
        header = [h.strip() for h in head.rstrip("\n").split(sep)]
        # Una descarga de GBIF trae ~50 columnas. Solo se leen las que usa el
        # pipeline: con decenas de millones de filas, leer todo no cabe en RAM.
        wanted = [c for c in header if c.lower() in KEEP_COLS_LOWER]
        has_coords = (any(c.lower() in LAT_NAMES for c in wanted)
                      and any(c.lower() in LON_NAMES for c in wanted))
        usecols = wanted if (has_coords and not (lat_col or lon_col)) else None
        # QUOTE_NONE: GBIF trae comillas sueltas en campos de texto libre.
        df = pd.read_csv(path, sep=sep, usecols=usecols, low_memory=False,
                         on_bad_lines="skip", quoting=3)

    cols = {c.lower().strip(): c for c in df.columns}
    lat = lat_col or next((cols[c] for c in LAT_NAMES if c in cols), None)
    lon = lon_col or next((cols[c] for c in LON_NAMES if c in cols), None)
    if lat is None or lon is None:
        raise SystemExit(f"No encontre columnas de lat/lon en {path.name}. "
                         f"Columnas: {list(df.columns)[:25]}. "
                         f"Usa --lat-col y --lon-col.")
    return df.rename(columns={lat: "lat", lon: "lon"})


# ----------------------------------------------------------------------------
# Limpieza, paso a paso, dejando constancia de cada descarte
# ----------------------------------------------------------------------------
def clean(df, keep_duplicates=False, round_decimals=5,
          max_uncertainty=MAX_UNCERTAINTY_M):
    """Aplica las reglas en orden. Devuelve (df_limpio, pasos).

    'pasos' es una lista de dicts {regla, antes, descartados, despues, razon}
    que el reporte muestra como un embudo.
    """
    pasos = []

    def log(regla, antes, despues, razon):
        pasos.append({"regla": regla, "antes": int(antes),
                      "descartados": int(antes - despues),
                      "despues": int(despues), "razon": razon})

    df = df.copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    n = len(df)
    df = df.dropna(subset=["lat", "lon"])
    log("Coordenadas ausentes", n, len(df),
        "Sin posicion no se puede buscar vecinos.")

    n = len(df)
    df = df[df.lat.between(-90, 90) & df.lon.between(-180, 180)]
    log("Fuera de rango", n, len(df),
        "Latitud fuera de [-90, 90] o longitud fuera de [-180, 180].")

    n = len(df)
    df = df[~((df.lat.abs() < 1e-9) & (df.lon.abs() < 1e-9))]
    log("Isla nula (0, 0)", n, len(df),
        "Valor por defecto de software, no un avistamiento real.")

    if "basisOfRecord" in df.columns:
        n = len(df)
        df = df[~df["basisOfRecord"].isin(BAD_BASIS)]
        log("Fosiles y ejemplares vivos", n, len(df),
            "La coordenada es la de la institucion, no la del organismo.")

    if "coordinateUncertaintyInMeters" in df.columns:
        unc = pd.to_numeric(df["coordinateUncertaintyInMeters"], errors="coerce")
        n = len(df)
        df = df[unc.isna() | (unc <= max_uncertainty)]
        log(f"Incertidumbre > {max_uncertainty/1000:g} km", n, len(df),
            "No sostiene un filtrado a 1 km. Se conservan los NaN: GBIF a "
            "menudo no la reporta.")

    # Presencia repetida = MISMA UNIDAD TAXONOMICA en la misma coordenada
    # (CoordinateCleaner::cc_dupl usa especie + coordenada). La unidad es la
    # misma del filtrado por especie (otu_key): especie; si no hay, BIN de BOLD;
    # si tampoco, scientificName (identificado solo a genero o mas arriba).
    # Dos unidades distintas en el mismo punto (p. ej. una trampa Malaise) NO
    # son repetidas: son dos registros validos.
    key = df[["lat", "lon"]].round(round_decimals)
    taxon_col = None
    if any(c in df.columns for c in ("species", "scientificName", "verbatimScientificName")):
        sp = otu_key(df)
        if "scientificName" in df.columns:
            sp = sp.where(sp.notna(), df["scientificName"])
        key = key.assign(_taxon=sp.astype(str).str.strip())
        taxon_col = "especie/BIN + coordenada"
    dup = key.duplicated(keep="first")
    n = len(df)
    if not keep_duplicates:
        df = df[~dup]
    criterio = taxon_col or "coordenada"
    log(f"Presencias repetidas ({criterio}, {round_decimals} decimales)", n, len(df),
        "Misma especie repetida en la misma coordenada. Inflan Sum n_i^2 y con "
        "ello el costo de la grilla." if not keep_duplicates else
        "CONSERVADOS a proposito (variante 'con duplicados').")

    return df.reset_index(drop=True), pasos


def otu_key(df, single=None):
    """Unidad taxonomica para filtrar POR ESPECIE (como GeoThinneR con grupos).

    Especie si existe; si no, el BIN de BOLD que aparezca en scientificName o
    verbatimScientificName; si no, NaN (identificado solo a genero o mas
    arriba: no se puede filtrar por especie y se excluye, con conteo).
    'single' = nombre para tratar todo el conjunto como una sola especie.
    """
    if single is not None:
        return pd.Series(single, index=df.index, dtype=object)
    out = pd.Series(np.nan, index=df.index, dtype=object)
    if "species" in df.columns:
        sp = df["species"].astype(object)
        ok = sp.notna() & (sp.astype(str).str.strip() != "") & (sp.astype(str) != "nan")
        out[ok] = sp[ok].astype(str).str.strip()
    for col in ("scientificName", "verbatimScientificName"):
        if col in df.columns:
            b = df[col].astype(str).str.extract(_BIN_RE, expand=False)
            fill = out.isna() & b.notna()
            out[fill] = b[fill]
    return out


# ----------------------------------------------------------------------------
# Geometria
# ----------------------------------------------------------------------------
def to_cartesian(lat_deg, lon_deg, radius=R_EARTH):
    """lat/lon en grados -> (N, 3) en metros sobre la esfera. float64."""
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    c = np.cos(lat)
    return np.stack([radius * c * np.cos(lon),
                     radius * c * np.sin(lon),
                     radius * np.sin(lat)], axis=1)


def chord_radius(d_m, radius=R_EARTH):
    """Distancia great-circle d -> radio euclidiano 3D equivalente."""
    return 2.0 * radius * np.sin(np.asarray(d_m, dtype=np.float64) / (2.0 * radius))


def chord_to_arc(c_m, radius=R_EARTH):
    return 2.0 * radius * np.arcsin(np.clip(np.asarray(c_m) / (2.0 * radius), -1, 1))


def haversine_m(lat1, lon1, lat2, lon2, radius=R_EARTH):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def chord_check(df, xyz, pairs=2000, seed=42):
    """Compara distancia por cuerda (lo que calcula la GPU) contra haversine
    (la verdad geodesica) en pares aleatorios. Devuelve los errores en m."""
    rng = np.random.default_rng(seed)
    m = min(pairs, len(df))
    i = rng.choice(len(df), m, replace=False)
    j = rng.choice(len(df), m, replace=False)
    ok = i != j
    i, j = i[ok], j[ok]
    chord = np.linalg.norm(xyz[i] - xyz[j], axis=1)
    arc = haversine_m(df.lat.values[i], df.lon.values[i],
                      df.lat.values[j], df.lon.values[j])
    return np.abs(chord_to_arc(chord) - arc), arc


# ----------------------------------------------------------------------------
# Lo que ve la grilla
# ----------------------------------------------------------------------------
def cell_counts(xyz, cell, origin_shift=(0.0, 0.0, 0.0)):
    """Conteos por celda de una grilla uniforme 3D con celda = 'cell' metros.

    Devuelve (conteos de celdas NO vacias, Q = celdas totales del bbox).
    Q incluye las vacias: es justo lo que la grilla desperdicia.
    """
    mn = xyz.min(axis=0) - np.asarray(origin_shift)
    idx = np.floor((xyz - mn) / cell).astype(np.int64)
    dims = idx.max(axis=0) + 1
    flat = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    # np.unique y no bincount: con extension global y celdas de pocos km,
    # Q llega a miles de millones y un arreglo denso no cabe en memoria.
    # (La grilla densa en GPU tendria el mismo problema: es un dato del paper.)
    _, counts = np.unique(flat, return_counts=True)
    return counts, int(np.prod(dims.astype(np.float64)))


def morisita(counts, Q, N):
    if N < 2:
        return float("nan")
    return float(Q * np.sum(counts * (counts - 1.0)) / (N * (N - 1.0)))


# ----------------------------------------------------------------------------
# Morisita en el plano (la forma correcta)
# ----------------------------------------------------------------------------
# En la grilla 3D, Q (celdas totales del bbox) incluye celdas en el interior
# de la Tierra donde nunca puede caer un punto, asi que Morisita sale inflado
# incluso para datos aleatorios. Los ecologos definen el indice sobre
# cuadrantes en un plano; aqui se usa una proyeccion de IGUAL AREA para que
# cada cuadrante represente la misma superficie real.

def laea(lat_deg, lon_deg, lat0_deg, lon0_deg, radius=R_EARTH):
    """Lambert azimutal de igual area sobre la esfera. Devuelve (x, y) en m."""
    phi, lam = np.radians(lat_deg), np.radians(lon_deg)
    phi0, lam0 = np.radians(lat0_deg), np.radians(lon0_deg)
    dl = lam - lam0
    denom = 1 + np.sin(phi0) * np.sin(phi) + np.cos(phi0) * np.cos(phi) * np.cos(dl)
    k = np.sqrt(2.0 / np.clip(denom, 1e-12, None))
    x = radius * k * np.cos(phi) * np.sin(dl)
    y = radius * k * (np.cos(phi0) * np.sin(phi) - np.sin(phi0) * np.cos(phi) * np.cos(dl))
    return x, y


def rect_area_m2(lat_min, lat_max, lon_min, lon_max, radius=R_EARTH):
    """Area real sobre la esfera del rectangulo lat/lon (area de estudio)."""
    return (radius ** 2 * np.radians(lon_max - lon_min)
            * (np.sin(np.radians(lat_max)) - np.sin(np.radians(lat_min))))


def morisita_2d(lat, lon, cell_m, rect, n_offsets=5, seed=123):
    """Morisita sobre cuadrantes de cell_m x cell_m en proyeccion de igual area.

    rect = (lat_min, lat_max, lon_min, lon_max): el area de estudio. Debe ser
    la MISMA regla para todos los conjuntos (aqui: el rectangulo lat/lon de
    los datos) y declararse en el paper.
    Q = area del rectangulo / area de un cuadrante.

    Se promedia sobre varios desplazamientos del origen (control de MAUP).
    Devuelve (media, coeficiente de variacion).
    """
    lat0 = 0.5 * (rect[0] + rect[1])
    lon0 = 0.5 * (rect[2] + rect[3])
    x, y = laea(np.asarray(lat), np.asarray(lon), lat0, lon0)
    Q = rect_area_m2(*rect) / (cell_m ** 2)
    N = len(x)
    rng = np.random.default_rng(seed)
    vals = []
    for k in range(n_offsets):
        sx, sy = (0.0, 0.0) if k == 0 else rng.uniform(0, cell_m, 2)
        ix = np.floor((x - x.min() + sx) / cell_m).astype(np.int64)
        iy = np.floor((y - y.min() + sy) / cell_m).astype(np.int64)
        _, c = np.unique(ix * (iy.max() + 2) + iy, return_counts=True)
        vals.append(morisita(c, Q, N))
    v = np.asarray(vals)
    return float(v.mean()), float(v.std() / v.mean()) if v.mean() else 0.0


def warp_efficiency(work, warp=32, seed=0):
    """Fraccion del trabajo util si los puntos se asignan a warps de 32 en
    orden aleatorio: cada warp tarda lo que su hilo mas cargado.
    1.0 = carga perfectamente pareja. 'work' = vecinos + 1 (toda consulta
    cuesta al menos una unidad)."""
    w = np.asarray(work, dtype=np.float64) + 1.0
    w = w[np.random.default_rng(seed).permutation(len(w))]
    n = (len(w) // warp) * warp
    if n == 0:
        return float("nan")
    blocks = w[:n].reshape(-1, warp)
    return float(blocks.sum() / (warp * blocks.max(axis=1).sum()))


def uniform_reference(df, n, seed=0):
    """Mismo numero de puntos, uniformes sobre la esfera dentro del mismo
    rectangulo lat/lon. Es la 'hipotesis nula': asi se verian los datos si
    no hubiera sesgo de muestreo."""
    rng = np.random.default_rng(seed)
    z0, z1 = np.sin(np.radians([df.lat.min(), df.lat.max()]))
    lat = np.degrees(np.arcsin(rng.uniform(z0, z1, n)))  # uniforme en area
    lon = rng.uniform(df.lon.min(), df.lon.max(), n)
    return pd.DataFrame({"lat": lat, "lon": lon})
