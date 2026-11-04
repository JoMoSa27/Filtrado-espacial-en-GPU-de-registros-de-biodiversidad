#!/bin/bash
#SBATCH --job-name=bip-geothinner
#SBATCH --partition=kura
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/geothinner_%j.out
#
# Linea base en CPU: GeoThinneR con la MISMA entrada que la GPU.
#
# Comparacion justa con el resto del paper (respuesta a los revisores, C1):
#   * 4 nucleos, los mismos que la linea base de SciPy (n_cores = 4 en
#     local_kd_tree; los demas metodos de GeoThinneR son de un solo hilo).
#     local_kd_tree se corre ademas con 1 nucleo (el valor por defecto del
#     paquete): con conjuntos chicos, levantar 4 procesos de R cuesta mas que
#     lo que se gana, y la comparacion se hace contra la configuracion MAS
#     RAPIDA de GeoThinneR;
#   * mejor de 10 intentos (trials = 10), como en el paper de GeoThinneR;
#   * 10 repeticiones por caso (semillas 1..10): se reporta la mediana;
#   * mismo radio terrestre que ThinGPU (ver geothinner_baseline.R).
#
# R con GeoThinneR: el R del sistema en Kabre no tiene GDAL/GEOS/PROJ, que
# necesitan sf y terra. Se usa el entorno de conda /work/$USER/r-geothinner
# (Miniforge en /work/$USER/miniforge3), activado UNA vez aqui, y Rscript se
# llama directo: con "conda run", timeout mataria a conda y dejaria a R
# corriendo huerfano.
# Antes, desde la raiz del repo (una vez por conjunto):
#   python3 scripts/05_references.py --label tortuga_gt --radii-km 10 25 50 --export-csv
#   python3 scripts/05_references.py --label atun_gt    --radii-km 10 25 50 --export-csv
#
# Corte fijado DE ANTEMANO: 30 min por corrida. Si la primera repeticion de un
# caso no termina, se reporta "no completo en 30 min" y no se intentan las
# demas; no se extrapola.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs bench/results
# Python del sistema (tiene numba, que usa 06_check_geothinner.py), fijado antes
# de activar conda para que el entorno de R no lo tape.
PY="$(command -v python3)"
# Los scripts de activacion de conda usan variables sin definir: sin set -u.
set +u
eval "$(/work/$USER/miniforge3/bin/conda shell.bash hook)"
conda activate /work/$USER/r-geothinner
set -u
R_CMD="${R_CMD:-Rscript}"
# Memoria pico y tiempo de pared de cada corrida, medidos desde afuera del
# proceso de R. Va a bench/results/ (que se sube) y no solo al log de SLURM.
MEM=bench/results/geothinner_memoria.csv
[ -f "$MEM" ] || echo "label,metodo,n_cores,d_km,semilla,max_rss_kb,pared_s,salida" > "$MEM"
TMP=$(mktemp)
# GNU time: los nodos kura (AlmaLinux) no traen /usr/bin/time; se instala en el
# mismo entorno de conda (paquete "time" de conda-forge). Sin el, no se corre:
# un trabajo que se salta todas las corridas no sirve de nada.
TIME_BIN="${TIME_BIN:-/work/$USER/r-geothinner/bin/time}"
[ -x "$TIME_BIN" ] || { echo "no encuentro GNU time en $TIME_BIN" >&2; exit 1; }
CORTE=1800
REPS="${REPS:-10}"

echo "== nodo: $(hostname)"; lscpu | grep -E "Model name|^CPU\(s\)"
$R_CMD -e 'cat(R.version.string, "\n"); cat("GeoThinneR", as.character(packageVersion("GeoThinneR")), "\n")'

corre() {  # label distancias metodos nucleos: una llamada por (metodo, d, repeticion)
  local label=$1 nucleos=$4 metodo d r
  for metodo in ${3//,/ }; do
    for d in ${2//,/ }; do
      for r in $(seq 1 $REPS); do
        # Retomar: si la corrida ya tiene resultado, no se repite.
        if [ -f "bench/results/geothinner_${label}_${metodo}_${nucleos}c_${d}km_s${r}.csv" ]; then
          echo "== $label $metodo ($nucleos nucleos) $d km semilla $r: ya hecha"; continue
        fi
        echo "== $label $metodo ($nucleos nucleos) $d km semilla $r"
        # %x = salida del comando: 124 si timeout corto la corrida.
        "$TIME_BIN" -o "$TMP" -f "%M,%e,%x" timeout $CORTE $R_CMD bench/geothinner_baseline.R \
             "$label" "$d" "$metodo" 10 "$nucleos" 1 "$r"
        echo "$label,$metodo,$nucleos,$d,$r,$(tail -1 "$TMP")" >> "$MEM"
        if [ "$(tail -1 "$TMP" | cut -d, -f3)" != "0" ]; then
          echo "   (no completo en ${CORTE}s o fallo: se omiten las demas repeticiones)"
          break
        fi
      done
    done
  done
}

# kd_tree y brute arman una matriz n x n: solo caben con la tortuga (8 340).
# SOLO=<label> (con DISTS) corre solo ese conjunto con local_kd_tree a 1 y 4
# nucleos; lo usa C6 (Turdus grayi): REPS=3 SOLO=turdus_grayi DISTS=10,25,50
# CASO="label|distancias|metodos|nucleos" corre solo ese caso (para repartir
# los casos lentos, como el atun, en trabajos paralelos). Sin validacion al
# final: la corre un trabajo aparte cuando terminan todos.
if [ -n "${CASO:-}" ]; then
  IFS='|' read -r c_label c_dists c_metodos c_nucleos <<< "$CASO"
  corre "$c_label" "$c_dists" "$c_metodos" "$c_nucleos"
  rm -f "$TMP"; exit 0
elif [ -n "${SOLO:-}" ]; then
  corre "$SOLO" "${DISTS:-10,25,50}" local_kd_tree 1
  corre "$SOLO" "${DISTS:-10,25,50}" local_kd_tree 4
else
  corre tortuga_gt 10,25,50 local_kd_tree 4
  corre tortuga_gt 10,25,50 local_kd_tree,k_estimation,kd_tree,brute 1
  corre atun_gt    10,25,50 local_kd_tree 4
  corre atun_gt    10,25,50 local_kd_tree,k_estimation 1
  # Subconjunto nacional por especie (opcional, si existe la entrada).
  [ -f "data/interim/amphibia_r.csv.gz" ] && corre amphibia 1,5,10 local_kd_tree 4
fi

rm -f "$TMP"
"$PY" scripts/06_check_geothinner.py
