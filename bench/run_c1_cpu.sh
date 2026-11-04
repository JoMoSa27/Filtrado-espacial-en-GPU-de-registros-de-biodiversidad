#!/bin/bash
#SBATCH --job-name=c1-cpu
#SBATCH --partition=kura
#SBATCH --cpus-per-task=1
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=logs/c1_cpu_%j.out
# C1: tiempo de una corrida del filtrado en CPU (scripts/13_c1_tiempos_cpu.py), en un
# nucleo del mismo tipo de nodo donde corrio GeoThinneR. Turdus a 50 km usa ~1.200
# millones de pares, de ahi la memoria.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs bench/results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "== nodo: $(hostname)"
cd scripts
python3 13_c1_tiempos_cpu.py --label tortuga_gt --reps 5
python3 13_c1_tiempos_cpu.py --label atun_gt --reps 5
python3 13_c1_tiempos_cpu.py --label turdus_grayi --reps 3
