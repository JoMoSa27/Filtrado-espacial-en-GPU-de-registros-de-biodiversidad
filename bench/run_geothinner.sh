#!/bin/bash
#SBATCH --job-name=bip-geothinner
#SBATCH --partition=kura
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/geothinner_%j.out
#
# Linea base en CPU: GeoThinneR con la MISMA entrada que la GPU.
# Requisitos (una vez, en el nodo login, que tiene internet):
#   R_ENV=/work/$USER/r-geothinner   (ver KABRE: R no es modulo documentado)
#   conda create -y -p $R_ENV -c conda-forge r-base=4.4 r-sf r-terra r-nabor \
#         r-fields r-matrixstats r-data.table r-doparallel r-foreach
#   conda run -p $R_ENV Rscript -e 'install.packages("GeoThinneR", repos="https://cloud.r-project.org")'
# y antes, desde la raiz del repo:
#   python3 scripts/05_references.py --label atun_gt --radii-km 10 25 50 --export-csv
#
# Corte fijado DE ANTEMANO: 30 min por corrida (timeout). Si no termina, se
# reporta "no completo en 30 min", no se extrapola.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs bench/results
R_ENV="${R_ENV:-/work/$USER/r-geothinner}"
RUN="conda run -p $R_ENV"
CORTE=1800

corre() {  # label distancias metodos trials n_cores reps
  echo "== $*"
  /usr/bin/time -v timeout $CORTE $RUN Rscript bench/geothinner_baseline.R "$@" \
    2> >(grep -E "Maximum resident|Elapsed|Exit status" >&2) \
    || echo "   (no completo en ${CORTE}s o fallo)"
}

corre tortuga_gt 10,25,50 local_kd_tree,k_estimation,kd_tree,brute 10 6 3
corre atun_gt    10,25,50 local_kd_tree,k_estimation 10 6 3
for L in amphibia aves_2018 plantae insecta_2018; do
  [ -f "data/interim/${L}_r.csv.gz" ] && corre "$L" 1,5,10 local_kd_tree 10 6 1
done
