#!/bin/bash
#SBATCH --job-name=bip-c2-sdm
#SBATCH --partition=kura
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/c2_sdm_%j.out
#
# C2 (revisores 1 y 2): piloto SDM con y sin filtrar. Ver bench/sdm_piloto.R
# para el diseño. Cuatro especies a la vez (una por núcleo).
#
# Requisitos: entorno conda /work/$USER/r-geothinner con r-terra, r-maxnet y
# r-glmnet; capa /work/$USER/sdm/CRI_wc2.1_30s_bio.tif (WorldClim 2.1, 30").
# Uso (desde la raíz del repo): sbatch bench/run_c2_sdm.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs bench/results/c2_sdm
PY="$(command -v python3)"
set +u
eval "$(/work/$USER/miniforge3/bin/conda shell.bash hook)"
conda activate /work/$USER/r-geothinner
set -u
BIO=/work/$USER/sdm/CRI_wc2.1_30s_bio.tif
echo "== nodo: $(hostname)"; sha256sum "$BIO"
Rscript -e 'cat(R.version.string, "| maxnet", as.character(packageVersion("maxnet")), "| terra", as.character(packageVersion("terra")), "\n")'

"$PY" scripts/11_c2_sdm_datos.py || exit 1
ls data/interim/sdm/*__*.csv.gz | xargs -P 4 -I{} Rscript bench/sdm_piloto.R {} "$BIO" bench/results/c2_sdm
"$PY" scripts/12_c2_resumen.py
