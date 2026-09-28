#!/bin/bash
#SBATCH --job-name=bip-datos
#SBATCH --partition=kura
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/datos_%j.out
#
# run_pipeline.sh — Corre el pipeline de datos completo, conjunto por conjunto.
#
# Lanzar DESDE LA RAIZ del repo:
#     sbatch scripts/run_pipeline.sh        (en Kabre, como trabajo)
#     bash   scripts/run_pipeline.sh        (en cualquier maquina)
#
# Para cada conjunto que tenga datos en data/raw/:
#   1. 00_explorar.py       reporte HTML en reports/
#   2. 02_prepare.py        <label> sin duplicados y <label>_dup con duplicados
#   3. 03_cluster_stats.py  predictores de agrupamiento
# Y despues arma las entradas de los experimentos:
#   E1  densidad de registros del grupo (esfuerzo de muestreo): los 4 taxones
#       de CR CON registros repetidos (variante _dup: una visita repetida es
#       esfuerzo), submuestreados al MISMO N
#   E2  lo mismo en escala: Aves 2018 _dup en 50k, 100k, 300k, 1M y completo
#   E3  tortuga y atun con la MISMA entrada que publico GeoThinneR (_gt: sin
#       filtro de incertidumbre ni de duplicados)
#   E4  filtrado POR ESPECIE de cada taxon (04_species_stats.py) y densidad
#       del grupo en CPU como linea base
#
# Los conjuntos sin datos se saltan con un aviso: se puede correr hoy con lo
# que haya y volver a correr cuando lleguen las otras descargas.

set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT" || exit 1
mkdir -p logs reports data/interim data/processed
PY="${PYTHON:-python3}"
S=scripts

log() { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }

# Devuelve el archivo de datos de una carpeta de GBIF. Descomprime si hace falta.
gbif_file() {
  local dir="$1"
  if ! ls "$dir"/*.csv >/dev/null 2>&1; then
    local z; z=$(ls "$dir"/*.zip 2>/dev/null | head -1)
    [ -n "$z" ] && unzip -o -q "$z" -d "$dir"
  fi
  ls "$dir"/*.csv 2>/dev/null | head -1
}

# conjunto: etiqueta | carpeta o archivo | radio del reporte (km) | radios | area de estudio
# Area: 'cr' = rectangulo comun de Costa Rica (hace comparables a los taxones);
#       'data' = rectangulo de los propios datos (tortuga y atun, fuera de CR).
DATASETS=(
  "aves_2018|data/raw/gbif_cr_aves_2018|5|1 5 10|cr"
  "insecta_2018|data/raw/gbif_cr_insecta_2018|5|1 5 10|cr"
  "plantae|data/raw/gbif_cr_plantae|5|1 5 10|cr"
  "amphibia|data/raw/gbif_cr_amphibia|5|1 5 10|cr"
  "tortuga|data/raw/geothinner/caretta.rda|10|10 25 50|data"
  "atun|data/raw/geothinner/thunnus.rda|25|10 25 50|data"
)

READY=()
for entry in "${DATASETS[@]}"; do
  IFS='|' read -r label src rkm radii rect <<< "$entry"
  if [ -d "$src" ]; then input=$(gbif_file "$src"); else input="$src"; fi
  if [ -z "${input:-}" ] || [ ! -f "$input" ]; then
    log "SALTO $label: todavia no hay datos en $src"
    continue
  fi
  log "$label  <-  $input"

  $PY $S/00_explorar.py "$input" --label "$label" --radius-km "$rkm" --radii-km $radii \
    || { echo "  !! fallo 00_explorar"; continue; }
  $PY $S/02_prepare.py "$input" --label "$label" || { echo "  !! fallo 02_prepare"; continue; }
  $PY $S/02_prepare.py "$input" --label "${label}_dup" --keep-duplicates >/dev/null \
    || echo "  !! fallo variante con duplicados"
  $PY $S/03_cluster_stats.py --label "$label" --radii-km $radii --rect "$rect" || echo "  !! fallo 03"
  $PY $S/03_cluster_stats.py --label "${label}_dup" --radii-km $radii --rect "$rect" >/dev/null \
    || echo "  !! fallo 03 (_dup)"
  READY+=("$label|$input")
done

# ---------------------------------------------------------------- E1 --------
CR=()
for r in "${READY[@]}"; do
  case "${r%%|*}" in aves_2018|insecta_2018|plantae|amphibia) CR+=("$r");; esac
done
# Un taxon con menos de E1_MIN puntos tras limpiar no sirve para medir en GPU:
# se excluye de E1 (con aviso) en vez de arrastrar a todos a ese tamano.
E1_MIN="${E1_MIN:-10000}"
KEEP=()
for r in "${CR[@]}"; do
  nf=$($PY -c "import json;print(json.load(open('data/processed/${r%%|*}_dup_report.json'))['n_final'])")
  if [ "$nf" -lt "$E1_MIN" ]; then
    log "E1: excluyo ${r%%|*} (N=$nf < $E1_MIN tras limpiar)"
  else
    KEEP+=("$r")
  fi
done
CR=("${KEEP[@]+"${KEEP[@]}"}")
if [ "${#CR[@]}" -ge 2 ]; then
  # N comun = el menor N tras limpiar, redondeado hacia abajo a miles.
  N=$($PY - "${CR[@]}" << 'EOF'
import json, sys
ns = [json.load(open(f"data/processed/{a.split('|')[0]}_dup_report.json"))["n_final"]
      for a in sys.argv[1:]]
print(min(ns) // 1000 * 1000)
EOF
)
  log "E1: ${#CR[@]} taxones de Costa Rica (con repetidos) al mismo N = $N"
  for r in "${CR[@]}"; do
    label="${r%%|*}"; input="${r#*|}"
    $PY $S/02_prepare.py "$input" --label "e1_${label}" --keep-duplicates --sample "$N" >/dev/null \
      && $PY $S/03_cluster_stats.py --label "e1_${label}" --radii-km 1 5 10 --rect cr \
      || echo "  !! fallo E1 $label"
  done
else
  log "E1 pendiente: hacen falta al menos 2 taxones de Costa Rica (hay ${#CR[@]})"
fi

# ---------------------------------------------------------------- E2 --------
AVES=""
for r in "${READY[@]}"; do [ "${r%%|*}" = "aves_2018" ] && AVES="${r#*|}"; done
if [ -n "$AVES" ]; then
  NA=$($PY -c 'import json;print(json.load(open("data/processed/aves_2018_dup_report.json"))["n_final"])')
  log "E2: barrido de N con Aves 2018 con repetidos (completo = aves_2018_dup, $NA)"
  for n in 50000 100000 300000 1000000; do
    if [ "$n" -ge "$NA" ]; then echo "  salto $n: Aves limpio tiene solo $NA"; continue; fi
    $PY $S/02_prepare.py "$AVES" --label "e2_aves_${n}" --keep-duplicates --sample "$n" >/dev/null \
      && $PY $S/03_cluster_stats.py --label "e2_aves_${n}" --radii-km 1 5 10 --rect cr >/dev/null \
      && echo "  e2_aves_${n} listo" || echo "  !! fallo e2_aves_${n}"
  done
else
  log "E2 pendiente: falta Aves 2018"
fi

# ---------------------------------------------------------------- E3 --------
# Misma entrada que publico GeoThinneR: todos los registros, sin filtro de
# incertidumbre ni de duplicados (GeoThinneR colapsa duplicados exactos solo).
for gt in "tortuga_gt|data/raw/geothinner/caretta.rda|Caretta caretta" \
          "atun_gt|data/raw/geothinner/thunnus.rda|Thunnus albacares"; do
  IFS='|' read -r label input name <<< "$gt"
  [ -f "$input" ] || continue
  log "E3: $label (entrada identica a GeoThinneR)"
  $PY $S/02_prepare.py "$input" --label "$label" --keep-duplicates --max-uncertainty-km 0 >/dev/null \
    && $PY $S/03_cluster_stats.py --label "$label" --radii-km 10 25 50 --rect data >/dev/null \
    && $PY $S/04_species_stats.py --label "$label" --radii-km 10 25 50 --rect data --single-otu "$name" \
    && $PY $S/05_references.py --label "$label" --radii-km 10 25 50 --w1 --w2 --export-csv \
    || echo "  !! fallo E3 $label"
done

# ---------------------------------------------------------------- E4 --------
# Filtrado por especie (lo que de verdad hace un biologo), con la referencia
# exacta (W2) que debe reproducir la GPU y el CSV de entrada para GeoThinneR.
for r in "${READY[@]}"; do
  label="${r%%|*}"
  case "$label" in aves_2018|insecta_2018|plantae|amphibia) ;; *) continue;; esac
  log "E4: filtrado por especie en $label"
  $PY $S/04_species_stats.py --label "$label" --radii-km 1 5 10 --rect cr \
    && $PY $S/05_references.py --label "$label" --radii-km 1 5 10 --w2 --export-csv \
    || echo "  !! fallo E4 $label"
done

# ------------------------------------------------ referencias W1 para GPU ---
# Conteos exactos de vecinos (cKDTree, float64). Su tiempo es la linea base en
# CPU (<label>_cpu_w1.json). Sobre N > 400 000 se omite: la GPU se valida ahi
# por acuerdo entre sus tres estructuras.
log "Referencias W1 (conteos exactos de vecinos) para validar la GPU"
for f in data/processed/e1_*_xyz_f64.npy data/processed/e2_*_xyz_f64.npy \
         data/processed/*_dup_xyz_f64.npy; do
  [ -f "$f" ] || continue
  label=$(basename "$f" _xyz_f64.npy)
  case "$label" in tortuga*|atun*) continue;; esac
  $PY $S/05_references.py --label "$label" --radii-km 1 5 10 --w1 || echo "  !! fallo W1 $label"
done

# --------------------------------------------------------------- figuras ---
[ -f data/processed/atun_gt_por_especie.csv ] && [ -f data/processed/tortuga_gt_por_especie.csv ] \
  && $PY $S/fig_thinning.py

# ---------------------------------------------------------------- resumen ---
log "Resumen"
$PY - << 'EOF'
import json, glob, os
rows = []
for f in sorted(glob.glob("data/processed/*_report.json")):
    r = json.load(open(f))
    rows.append((r["label"], r["n_raw"], r["n_final"]))
if not rows:
    print("  (nada procesado todavia)")
for lab, a, b in rows:
    print(f"  {lab:<22} {a:>12,} crudos  ->  {b:>12,} finales")
print("\nReportes HTML:", ", ".join(sorted(os.path.basename(p) for p in glob.glob("reports/*.html"))))
EOF
