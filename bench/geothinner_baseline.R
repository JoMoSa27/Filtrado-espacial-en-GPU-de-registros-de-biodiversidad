#!/usr/bin/env Rscript
# geothinner_baseline.R — Linea base en CPU con GeoThinneR (Mestre-Tomas 2026).
#
# Misma entrada que la GPU: data/interim/<label>_r.csv.gz (lon, lat, otu), que
# escribe scripts/05_references.py --export-csv. Una fila por (metodo, d, rep):
# tiempo de pared, puntos retenidos. La memoria pico se mide desde afuera con
# /usr/bin/time -v (ver bench/run_geothinner.sh).
#
# Uso: Rscript bench/geothinner_baseline.R <label> <d_km,...> <search_type,...> [trials] [n_cores] [reps]
#   Rscript bench/geothinner_baseline.R atun_gt 10,25,50 local_kd_tree,k_estimation 10 6 3
suppressPackageStartupMessages(library(GeoThinneR))
a <- commandArgs(trailingOnly = TRUE)
label   <- a[1]
dists   <- as.numeric(strsplit(a[2], ",")[[1]])
methods <- strsplit(a[3], ",")[[1]]
trials  <- if (length(a) >= 4) as.integer(a[4]) else 10L
ncores  <- if (length(a) >= 5) as.integer(a[5]) else 1L
reps    <- if (length(a) >= 6) as.integer(a[6]) else 3L

x <- read.csv(gzfile(file.path("data", "interim", paste0(label, "_r.csv.gz"))))
grupos <- length(unique(x$otu))
gcol <- if (grupos > 1) "otu" else NULL
cat(sprintf("[%s] %d registros, %d especies, trials=%d, n_cores=%d\n",
            label, nrow(x), grupos, trials, ncores))

out <- file.path("bench", "results", sprintf("geothinner_%s_%s.csv", label,
                 format(Sys.time(), "%Y%m%d_%H%M%S")))
dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
rows <- list()
for (m in methods) for (d in dists) for (r in seq_len(reps)) {
  t <- system.time(res <- tryCatch(
    thin_points(x, lon_col = "lon", lat_col = "lat", group_col = gcol,
                method = "distance", search_type = m, thin_dist = d,
                trials = trials, seed = r, n_cores = ncores),
    error = function(e) e))
  ok <- !inherits(res, "error")
  kept <- if (ok) sum(res$retained[[1]]) else NA
  
  # Mascara de la primera repeticion: 05/validar confirma sobre la salida REAL
  # de GeoThinneR si quedan registros agregables (conjunto no maximal).
  if (ok && r == 1) {
    mk <- file.path("bench", "results", sprintf("geothinner_mask_%s_%s_%gkm.csv.gz", label, m, d))
    write.csv(data.frame(row = x$row[res$retained[[1]]]), gzfile(mk), row.names = FALSE)
  }
  rows[[length(rows) + 1]] <- data.frame(
    label = label, N = nrow(x), especies = grupos, metodo = m, d_km = d, rep = r,
    trials = trials, n_cores = ncores, seg = unname(t["elapsed"]), retenidos = kept,
    error = if (ok) "" else conditionMessage(res))
  cat(sprintf("  %-14s d=%4.0f km rep %d: %8.2f s  retenidos %s %s\n", m, d, r,
              t["elapsed"], kept, if (ok) "" else conditionMessage(res)))
  write.csv(do.call(rbind, rows), out, row.names = FALSE)
}
cat("->", out, "\n")
