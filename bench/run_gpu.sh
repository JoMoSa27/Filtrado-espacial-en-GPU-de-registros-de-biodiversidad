#!/bin/bash
#SBATCH --job-name=bip-gpu
#SBATCH --partition=nukwa-l40s
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/gpu_%j.out
#
# En Kabre no esta documentado si la particion nukwa-l40s necesita --gres para
# entregar la GPU. Se deja comentado a proposito: si el trabajo arranca y
# nvidia-smi ve el dispositivo, no hace falta. Confirmar con:
#     sinfo -o "%20P %10N %15G %6t"
# y si la columna GRES muestra gpu:l40s:N, descomentar la linea siguiente.
##SBATCH --gres=gpu:1
#
# run_gpu.sh — Benchmark en GPU del poster BIP 2026.
#
# DOS TRABAJOS SEPARADOS, porque tienen costes muy distintos y comparten el
# limite de 4 h: W2 reconstruye la estructura y el CSR en cada repeticion, asi
# que es mucho mas caro que W1. Mandarlos juntos arriesga perder los dos.
#
#   CARGA=W1 sbatch bench/run_gpu.sh     tortuga_gt, atun_gt, e1_*, e2_aves_*
#   CARGA=W2 sbatch bench/run_gpu.sh     tortuga_gt, atun_gt
#
# Sin CARGA corre los dos, en ese orden.
#
# Variables:
#   DATA_STRUCTURES_DIR  repo de las estructuras (obligatoria)
#   CARGA                W1 | W2 | ambas (default)
#   ARCH                 arquitectura CUDA (L40S = 89)

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs bench/results

: "${DATA_STRUCTURES_DIR:?Falta DATA_STRUCTURES_DIR (repo de las estructuras)}"
ARCH="${ARCH:-89}"
CARGA="${CARGA:-ambas}"
case "$CARGA" in W1|W2|ambas) ;; *) echo "CARGA debe ser W1, W2 o ambas"; exit 2;; esac

# El nombre exacto del modulo cambia entre instalaciones. Ver: module avail cuda
if command -v module >/dev/null 2>&1; then
  module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true
fi
command -v nvcc >/dev/null || { echo "ERROR: no hay nvcc. Revisar: module avail cuda"; exit 1; }
echo "== nvcc: $(nvcc --version | tail -1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "== Compilando (CARGA=$CARGA)"
cmake -S bench -B build \
      -DDATA_STRUCTURES_DIR="$DATA_STRUCTURES_DIR" \
      -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
      --log-level=WARNING
cmake --build build -j 8

FECHA=$(date +%Y-%m-%d)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_' | tr -d '()')
GPU="${GPU:-gpu}"
OUT="bench/results/${FECHA}_${GPU}"

# Conjuntos por carga. Formato: etiqueta:radios en km.
#   W1  los *_gt validan todo, E1 mide el efecto del agrupamiento a N fijo y
#       E2 el escalamiento. E2 se corta en 300k: los tamannos mayores no caben
#       en 4 h junto a lo demas y van en su propio trabajo si sobra tiempo.
#   W2  solo los *_gt: una sola especie y con referencia de thin.py contra la
#       cual comparar el conjunto retenido y las rondas.
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

falta() {   # true si no estan los .npy del conjunto
  [ ! -f "data/processed/$1_xyz_f32_centered.npy" ]
}

if [ "$CARGA" = "W1" ] || [ "$CARGA" = "ambas" ]; then
  echo "== W1"
  for cfg in "${W1_SETS[@]}"; do
    label="${cfg%%:*}"; radios="${cfg#*:}"
    if falta "$label"; then echo "   SALTO $label: faltan los .npy"; continue; fi
    echo "-- $label ($radios km)"
    ./build/bench_w1 --label "$label" --radii-km $radios \
                     --processed data/processed \
                     --lib "$DATA_STRUCTURES_DIR" \
                     --out "${OUT}_w1_${label}.csv"
  done
fi

if [ "$CARGA" = "W2" ] || [ "$CARGA" = "ambas" ]; then
  echo "== W2"
  for cfg in "${W2_SETS[@]}"; do
    label="${cfg%%:*}"; radios="${cfg#*:}"
    if falta "$label"; then echo "   SALTO $label: faltan los .npy"; continue; fi
    echo "-- $label ($radios km)"
    ./build/bench_w2 --label "$label" --radii-km $radios \
                     --rules random mindeg --seeds 0 \
                     --processed data/processed \
                     --vectores bench/splitmix_vectors.csv \
                     --lib "$DATA_STRUCTURES_DIR" \
                     --out "${OUT}_w2_${label}.csv"
  done
fi

echo "== Listo. Resultados en bench/results/"
ls -la bench/results/ | tail -20
