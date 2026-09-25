#!/bin/bash
#SBATCH --job-name=bip-datos
#SBATCH --time=02:00:00
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
#   E1  los 4 taxones de CR submuestreados al MISMO N (el minimo tras limpiar)
#   E2  Aves 2018 en 50k, 100k, 300k, 1M (y completo = aves_2018)
#   E3  tortuga y atun tal cual (ya salen del paso 2)
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

# conjunto: etiqueta | carpeta o archivo | radio del reporte (km) | radios de la tabla
DATASETS=(
  "aves_2018|data/raw/gbif_cr_aves_2018|5|1 5 10"
  "insecta_2018|data/raw/gbif_cr_insecta_2018|5|1 5 10"
  "plantae|data/raw/gbif_cr_plantae|5|1 5 10"
  "amphibia|data/raw/gbif_cr_amphibia|5|1 5 10"
  "tortuga|data/raw/geothinner/caretta.rda|10|10 25 50"
  "atun|data/raw/geothinner/thunnus.rda|25|10 25 50"
)

READY=()
for entry in "${DATASETS[@]}"; do
  IFS='|' read -r label src rkm radii <<< "$entry"
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
  $PY $S/03_cluster_stats.py --label "$label" --radii-km $radii || echo "  !! fallo 03"
  READY+=("$label|$input")
done

# ---------------------------------------------------------------- E1 --------
CR=()
for r in "${READY[@]}"; do
  case "${r%%|*}" in aves_2018|insecta_2018|plantae|amphibia) CR+=("$r");; esac
done
if [ "${#CR[@]}" -ge 2 ]; then
  # N comun = el menor N tras limpiar, redondeado hacia abajo a miles.
  N=$($PY - "${CR[@]}" << 'EOF'
import json, sys
ns = [json.load(open(f"data/processed/{a.split('|')[0]}_report.json"))["n_final"]
      for a in sys.argv[1:]]
print(min(ns) // 1000 * 1000)
EOF
)
  log "E1: ${#CR[@]} taxones de Costa Rica al mismo N = $N"
  for r in "${CR[@]}"; do
    label="${r%%|*}"; input="${r#*|}"
    $PY $S/02_prepare.py "$input" --label "e1_${label}" --sample "$N" >/dev/null \
      && $PY $S/03_cluster_stats.py --label "e1_${label}" --radii-km 1 5 10 \
      || echo "  !! fallo E1 $label"
  done
else
  log "E1 pendiente: hacen falta al menos 2 taxones de Costa Rica (hay ${#CR[@]})"
fi

# ---------------------------------------------------------------- E2 --------
AVES=""
for r in "${READY[@]}"; do [ "${r%%|*}" = "aves_2018" ] && AVES="${r#*|}"; done
if [ -n "$AVES" ]; then
  NA=$($PY -c 'import json;print(json.load(open("data/processed/aves_2018_report.json"))["n_final"])')
  log "E2: barrido de N con Aves 2018 (completo = $NA)"
  for n in 50000 100000 300000 1000000; do
    if [ "$n" -ge "$NA" ]; then echo "  salto $n: Aves limpio tiene solo $NA"; continue; fi
    $PY $S/02_prepare.py "$AVES" --label "e2_aves_${n}" --sample "$n" >/dev/null \
      && echo "  e2_aves_${n} listo" || echo "  !! fallo e2_aves_${n}"
  done
else
  log "E2 pendiente: falta Aves 2018"
fi

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
