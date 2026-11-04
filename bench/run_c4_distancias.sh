#!/bin/bash
#SBATCH --job-name=bip-c4-dist
#SBATCH --partition=kura
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/c4_distancias_%j.out
#
# C4 (revisor 1): barrido de la distancia de filtrado sobre los cuatro taxones
# nacionales. Las máscaras de 1, 5 y 10 km ya existen (corrida del paper); aquí
# se agregan 2, 25 y 50 km con la misma regla (mindeg) y semilla (0), en float64,
# y después se mide retención y concentración para todas las distancias.
#
# Uso (desde la raíz del repo): sbatch bench/run_c4_distancias.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs bench/results
echo "== nodo: $(hostname)"
for L in amphibia insecta_2018 aves_2018 plantae; do
  for d in 1 2 5 10 25 50; do
    if [ ! -f "data/processed/${L}_thin_mindeg_${d}km_s0.npy" ]; then
      echo "-- $L $d km"
      python3 scripts/05_references.py --label "$L" --radii-km "$d" --w2 --rules mindeg --seeds 0
    fi
  done
done
# El mínimo de concentración puede depender del lado de la malla con que se
# mide: se mide con 2, 5 (la del paper) y 10 km.
for c in 2 5 10; do
  python3 scripts/10_c4_distancias.py --radii-km 1 2 5 10 25 50 --cell-km $c
done
