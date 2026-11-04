#!/bin/bash
#SBATCH --job-name=bip-c6-turdus
#SBATCH --partition=kura
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/c6_turdus_%j.out
#
# C6 (revisor 2): una especie de gran rango, Turdus grayi en todo su rango
# (GBIF.org, 3 Nov 2026, doi:10.15468/dl.js3buy; 949 010 registros con los
# mismos filtros que las descargas nacionales).
#
# Parte CPU: preparar, referencias exactas (f32c para validar la GPU, f64 como
# verdad) y GeoThinneR a 1 y 4 nucleos con el mismo corte de 30 min. La parte
# GPU va aparte (nukwa-l40s) y depende de esta:
#   A=$(sbatch --parsable bench/run_c6_turdus.sh)
#   SOLO=turdus_grayi CARGA=W2 DATA_STRUCTURES_DIR=$HOME/ds_bip/projects/Data_Structures \
#     sbatch --dependency=afterany:$A bench/run_gpu.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs bench/results
# Datos en data/raw/gbif_turdus_grayi/ (procedencia y DOI en su CITATION.txt).
RAW=data/raw/gbif_turdus_grayi
CSV=$RAW/0023623-261023074918325.csv
L=turdus_grayi
echo "== nodo: $(hostname)"
[ -f "$CSV" ] || { echo "falta $CSV: ver KABRE.md para bajarla"; exit 1; }

python3 scripts/02_prepare.py "$CSV" --label $L || exit 1
# Una sola especie: id 0 para todos (lo mismo que 04_species_stats.py --single-otu).
python3 - <<PY || exit 1
import numpy as np, pandas as pd
n = len(np.load("data/processed/${L}_xyz_f64.npy"))
np.save("data/processed/${L}_otu.npy", np.zeros(n, np.int32))
pd.DataFrame({"otu_id": [0], "nombre": ["Turdus grayi"]}).to_csv("data/processed/${L}_otu_nombres.csv", index=False)
print(f"{n:,} registros unicos de Turdus grayi")
PY
python3 scripts/05_references.py --label $L --export-csv

# Una distancia por llamada: si 50 km no cabe en memoria, 10 y 25 km quedan.
for d in 10 25 50; do
  python3 scripts/05_references.py --label $L --radii-km $d --w2 --rules random mindeg --seeds 0 --coords f32c \
    || echo "   referencia f32c a $d km fallo"
done

REPS=3 SOLO=$L DISTS=10,25,50 bash bench/run_geothinner.sh

for d in 10 25 50; do
  python3 scripts/05_references.py --label $L --radii-km $d --w2 --rules mindeg --seeds 0 \
    || echo "   referencia f64 a $d km fallo"
done
