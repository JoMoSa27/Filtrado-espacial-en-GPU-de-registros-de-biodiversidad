#!/bin/bash
#SBATCH --job-name=bip-gpu
#SBATCH --partition=nukwa-l40s
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/gpu_%j.out
# DOS TRABAJOS SEPARADOS, porque tienen costes muy distintos y comparten el
# limite de 4 h: W2 reconstruye la estructura y el CSR en cada repeticion, asi
# que es mucho mas caro que W1. Mandarlos juntos arriesga perder los dos.
#
#   CARGA=W1 sbatch bench/run_gpu.sh              tortuga_gt, atun_gt, e1_*, e2_aves_*
#   CARGA=W2 sbatch bench/run_gpu.sh              tortuga_gt, atun_gt
#   SOLO=tortuga_gt CARGA=W2 sbatch bench/run_gpu.sh    un solo conjunto
#
# Variables:
#   DATA_STRUCTURES_DIR  repo de las estructuras (default: $HOME/...)
#   CARGA                W1 | W2 | ambas (default)
#   SOLO                 correr solo ese conjunto
#   ARCH                 arquitectura CUDA (L40S = 89)
#   MOD_CUDA / MOD_GCC   nombres de modulo, por si cambian

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs bench/results

DATA_STRUCTURES_DIR="${DATA_STRUCTURES_DIR:-$HOME/asistencia_cenat_josue_morera/projects/Data_Structures}"
ARCH="${ARCH:-89}"                 # L40S = sm_89
CARGA="${CARGA:-ambas}"
SOLO="${SOLO:-}"
MOD_CUDA="${MOD_CUDA:-cuda/12.4.0}"

case "$CARGA" in W1|W2|ambas) ;; *) echo "CARGA debe ser W1, W2 o ambas"; exit 2;; esac

if [ ! -f "$DATA_STRUCTURES_DIR/structures/common/include/SpatialTags.cuh" ]; then
  echo "ERROR: DATA_STRUCTURES_DIR no apunta al repo de las estructuras."
  echo "  esperado: \$DATA_STRUCTURES_DIR/structures/common/include/SpatialTags.cuh"
  echo "  recibido: $DATA_STRUCTURES_DIR"
  exit 1
fi

# Modulos
if command -v module >/dev/null 2>&1; then
  module load "$MOD_CUDA" 2>/dev/null || module load cuda 2>/dev/null || true
fi
command -v nvcc >/dev/null || {
  echo "ERROR: no hay nvcc. Ver: module avail cuda"; exit 1; }

# Compilador de host
# Comprobado en el nodo de login de Kabre: el gcc por defecto ya es 11.5.0, que
# sirve tal cual. Esta seccion es una red de seguridad, no un arreglo: el nodo
# de computo podria traer otra imagen, y el trabajo no deberia enterarse a mitad
# de un nvcc con un error de plantillas ilegible.
#
# La banda es [8, 13]:
#   minimo 8  -> el codigo es C++17 y en gcc 6 esta incompleto
#   maximo 13 -> CUDA 12.4 soporta hasta gcc 13.2. OJO: el modulo gcc/13.4.0
#                EXISTE en Kabre y esta POR ENCIMA de ese tope, asi que nvcc
#                lo rechazaria. Por eso no esta entre los candidatos.
GCC_MIN=8
GCC_MAX=13
# Devuelve 0 si no hay gcc o si la version no se puede leer, para que las
# comparaciones de abajo esten siempre bien definidas y el fallo sea explicito
# en vez de que el script siga de largo.
gcc_major() {
  local v
  command -v gcc >/dev/null 2>&1 || { echo 0; return; }
  v=$(gcc -dumpversion 2>/dev/null | cut -d. -f1)
  case "$v" in (''|*[!0-9]*) echo 0;; (*) echo "$v";; esac
}

if [ "$(gcc_major)" -lt "$GCC_MIN" ]; then
  echo "== gcc $(gcc -dumpversion) es viejo para CUDA 12.4 + C++17; buscando modulo"
  if command -v module >/dev/null 2>&1; then
    # Nombres reales de `module avail gcc` en Kabre, ordenados de mas a menos
    # conveniente. Se deja fuera gcc/13.4.0 a proposito: supera el tope de
    # CUDA 12.4 (13.2) y nvcc lo rechazaria.
    for m in ${MOD_GCC:-} gcc/11.1.0 gcc/12.4.0 gcc/9.3.0; do
      [ -n "$m" ] || continue
      if module load "$m" 2>/dev/null; then
        echo "   cargado: $m -> gcc $(gcc -dumpversion)"
        break
      fi
    done
  fi
fi

if [ "$(gcc_major)" -lt "$GCC_MIN" ] || [ "$(gcc_major)" -gt "$GCC_MAX" ]; then
  echo "ERROR: gcc $(gcc -dumpversion) fuera de la banda [$GCC_MIN, $GCC_MAX]."
  echo "  nvcc 12.4 acepta gcc 6.x-13.2; el codigo necesita C++17 completo."
  echo "  Modulos disponibles en Kabre: gcc/9.3.0  gcc/11.1.0  gcc/12.4.0"
  echo "  (gcc/13.4.0 existe pero supera el tope de CUDA 12.4.)"
  echo "  Usar:  MOD_GCC=gcc/11.1.0 $0"
  exit 1
fi

echo "== nvcc: $(nvcc --version | tail -1)"
echo "== gcc : $(gcc --version | head -1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Compilacion
# CMAKE_CUDA_HOST_COMPILER explicito: si el gcc bueno vino por modulo, CMake
# podria quedarse con el del sistema si ya hay cache de una corrida anterior.
echo "== Compilando (CARGA=$CARGA${SOLO:+, SOLO=$SOLO})"
cmake -S bench -B build \
      -DDATA_STRUCTURES_DIR="$DATA_STRUCTURES_DIR" \
      -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
      -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++)" \
      --log-level=WARNING
cmake --build build -j "${SLURM_CPUS_PER_TASK:-8}"

FECHA=$(date +%Y-%m-%d)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_' | tr -d '()')
GPU="${GPU:-gpu}"
OUT="bench/results/${FECHA}_${GPU}"

# Conjuntos
# Formato: etiqueta:radios en km.
#   W1  los *_gt validan todo, E1 mide el agrupamiento a N fijo y E2 la escala.
#       E2 se corta en 300k: los tamannos mayores no caben en 4 h junto a lo
#       demas y van en su propio trabajo si sobra tiempo.
#   W2  solo los *_gt: una sola especie, y son los que tienen referencia de
#       thin.py contra la cual comparar el conjunto retenido y las rondas.
W1_SETS=(
  "tortuga_gt:10 25 50"
  "atun_gt:10 25 50"
  "e1_aves_2018:1 5 10"
  "e1_insecta_2018:1 5 10"
  "e1_plantae:1 5 10"
  "e1_amphibia:1 5 10"
  "e2_aves_50000:1 5 10"
  "e2_aves_100000:1 5 10"
  "e2_aves_300000:1 5 10"
)
W2_SETS=(
  "tortuga_gt:10 25 50"
  "atun_gt:10 25 50"
)

falta() { [ ! -f "data/processed/$1_xyz_f32_centered.npy" ]; }

# Con SOLO=<label> se filtra la lista. Si el conjunto no esta en la lista de esa
# carga se avisa, en vez de correr en silencio sin hacer nada.
filtrar() {
  local encontrado=0 cfg
  for cfg in "$@"; do
    [ -z "$SOLO" ] || [ "${cfg%%:*}" = "$SOLO" ] || continue
    encontrado=1
    echo "$cfg"
  done
  if [ -n "$SOLO" ] && [ "$encontrado" = "0" ]; then
    echo "   SOLO=$SOLO no esta en esta carga" >&2
  fi
}

if [ "$CARGA" = "W1" ] || [ "$CARGA" = "ambas" ]; then
  echo "== W1"
  while read -r cfg; do
    [ -n "$cfg" ] || continue
    label="${cfg%%:*}"; radios="${cfg#*:}"
    if falta "$label"; then echo "   SALTO $label: faltan los .npy"; continue; fi
    echo "-- $label ($radios km)"
    ./build/bench_w1 --label "$label" --radii-km $radios \
                     --processed data/processed \
                     --lib "$DATA_STRUCTURES_DIR" \
                     --out "${OUT}_w1_${label}.csv"
  done < <(filtrar "${W1_SETS[@]}")
fi

if [ "$CARGA" = "W2" ] || [ "$CARGA" = "ambas" ]; then
  echo "== W2"
  while read -r cfg; do
    [ -n "$cfg" ] || continue
    label="${cfg%%:*}"; radios="${cfg#*:}"
    if falta "$label"; then echo "   SALTO $label: faltan los .npy"; continue; fi
    echo "-- $label ($radios km)"
    ./build/bench_w2 --label "$label" --radii-km $radios \
                     --rules random mindeg --seeds 0 \
                     --processed data/processed \
                     --vectores bench/splitmix_vectors.csv \
                     --lib "$DATA_STRUCTURES_DIR" \
                     --out "${OUT}_w2_${label}.csv"
  done < <(filtrar "${W2_SETS[@]}")
fi

echo "== Listo. Resultados en bench/results/"
ls -la bench/results/ | tail -20
