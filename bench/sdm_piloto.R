#!/usr/bin/env Rscript
# sdm_piloto.R — Piloto SDM (C2): ¿cambia el modelo al filtrar con ThinGPU?
#
# Una especie por llamada. Entrada: data/interim/sdm/<label>__<otu>.csv.gz
# (11_c2_sdm_datos.py) y la capa WorldClim 2.1 bioclimática de 30" de Costa Rica.
#
# Diseño (todo fijado antes de ver resultados):
#   predictores  WorldClim 2.1 (Fick y Hijmans 2017), selección voraz en un
#                orden a priori, sin pares con |r| >= 0,7 (Dormann et al. 2013),
#                sobre 10 000 celdas de fondo al azar. Igual para todas las especies.
#   modelo       MaxEnt con maxnet (Phillips et al. 2017), clases por defecto
#                según n y regmult = 1, salida cloglog.
#   tratamientos sin filtrar (registros únicos); filtrado a 1, 5 y 10 km con la
#                regla de ThinGPU (semillas 0..9); y sin filtrar con fondo de
#                grupo objetivo (Phillips et al. 2009), la otra corrección.
#   validación   bloques espaciales tipo ENMeval "block" (Muscarella et al.
#                2014; Radosavljevic y Anderson 2014): 4 bloques por la mediana
#                de latitud y luego de longitud de TODOS los registros únicos.
#                El conjunto de prueba de cada bloque es el MISMO para todos los
#                tratamientos (todos los registros únicos del bloque, contra
#                fondo al azar del bloque); solo cambia el entrenamiento.
#   métricas     AUC de prueba, AUC de entrenamiento menos AUC de prueba,
#                omisión con el umbral del percentil 10 de entrenamiento (OR10),
#                índice de Boyce continuo (Hirzel et al. 2006) y, con el modelo
#                ajustado a todos los datos, la correlación de Spearman entre la
#                idoneidad predicha y el esfuerzo de muestreo (registros del
#                taxón a ~5 km) en el fondo: cuánto "aprendió" el esfuerzo.
#   salvedad     los registros de prueba tienen el mismo sesgo que los de
#                entrenamiento: una métrica mejor o peor en ellos no prueba un
#                modelo mejor o peor. Por eso es un piloto.
#
# Uso: Rscript bench/sdm_piloto.R <csv de la especie> <tif WorldClim> [salida]
suppressPackageStartupMessages({ library(terra); library(maxnet) })
a <- commandArgs(trailingOnly = TRUE)
archivo <- a[1]
bio_tif <- a[2]
out_dir <- if (length(a) >= 3) a[3] else file.path("bench", "results", "c2_sdm")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
base <- sub("\\.csv\\.gz$", "", basename(archivo))
label <- sub("__.*$", "", base)

# ------------------------------------------------------------ predictores
env <- rast(bio_tif)
names(env) <- paste0("bio", as.integer(sub(".*_bio_", "", names(env))))
set.seed(2026)
bg_xy <- spatSample(env[[1]], 10000, "random", na.rm = TRUE, xy = TRUE)[, c("x", "y")]
bg_all <- extract(env, bg_xy, ID = FALSE)
orden <- paste0("bio", c(1, 12, 4, 15, 2, 7, 5, 6, 13, 14, 3, 8, 9, 10, 11, 16, 17, 18, 19))
vars <- character(0)
for (v in orden) {
  if (length(vars) == 0 || max(abs(cor(bg_all[[v]], bg_all[, vars, drop = FALSE]))) < 0.7) vars <- c(vars, v)
}
env <- env[[vars]]
bg_env <- bg_all[, vars]

# ------------------------------------------------- esfuerzo y fondo de grupo objetivo
esf <- read.csv(gzfile(file.path(dirname(archivo), paste0("esfuerzo_", label, ".csv.gz"))))
conteo <- rasterize(vect(esf, geom = c("lon", "lat"), crs = "EPSG:4326"), env[[1]], fun = "length")
conteo[is.na(conteo)] <- 0
esfuerzo <- focal(conteo, w = 5, fun = "sum", na.rm = TRUE)   # 5 x 5 celdas de 30" (~4,6 km)
bg_esfuerzo <- extract(esfuerzo, bg_xy, ID = FALSE)[[1]]
tg <- esf[sample.int(nrow(esf), min(10000, nrow(esf))), c("lon", "lat")]
tg_env <- extract(env, tg, ID = FALSE)
ok_tg <- complete.cases(tg_env)
tg <- tg[ok_tg, ]; tg_env <- tg_env[ok_tg, ]

# ------------------------------------------------------------- presencias
p <- read.csv(gzfile(archivo))
p_env <- extract(env, p[, c("lon", "lat")], ID = FALSE)
ok <- complete.cases(p_env)
p <- p[ok, ]; p_env <- p_env[ok, ]

# Bloques ENMeval "block": mediana de latitud y, dentro de cada mitad, de longitud.
lat_m <- median(p$lat)
lon_n <- median(p$lon[p$lat >= lat_m]); lon_s <- median(p$lon[p$lat < lat_m])
bloque <- function(lon, lat) ifelse(lat >= lat_m, ifelse(lon >= lon_n, 1L, 2L), ifelse(lon >= lon_s, 3L, 4L))
p_b <- bloque(p$lon, p$lat)
bg_b <- bloque(bg_xy$x, bg_xy$y)
tg_b <- bloque(tg$lon, tg$lat)

auc <- function(pos, neg) {
  r <- rank(c(pos, neg))
  (sum(r[seq_along(pos)]) - length(pos) * (length(pos) + 1) / 2) / (length(pos) * length(neg))
}
boyce <- function(fondo, obs, res = 100) {
  # Índice de Boyce continuo (Hirzel et al. 2006): ventana móvil de 1/10 del rango.
  lo <- min(fondo); hi <- max(fondo); w <- (hi - lo) / 10
  centros <- seq(lo + w / 2, hi - w / 2, length.out = res)
  pe <- sapply(centros, function(c) {
    P <- mean(obs >= c - w / 2 & obs <= c + w / 2)
    E <- mean(fondo >= c - w / 2 & fondo <= c + w / 2)
    if (E == 0) NA else P / E
  })
  keep <- !is.na(pe)
  if (sum(keep) < 3) return(NA_real_)
  suppressWarnings(cor(centros[keep], pe[keep], method = "spearman"))
}
ajusta <- function(pres_env, fondo_env) {
  y <- c(rep(1, nrow(pres_env)), rep(0, nrow(fondo_env)))
  d <- rbind(pres_env, fondo_env)
  maxnet(y, d, f = maxnet.formula(y, d, classes = "default"), regmult = 1)
}
predice <- function(m, d) as.numeric(predict(m, d, type = "cloglog"))

# ------------------------------------------------------------ tratamientos
trat <- list(list(nombre = "sin_filtrar", d_km = 0, semilla = NA, sel = rep(TRUE, nrow(p)), tgb = FALSE),
             list(nombre = "grupo_objetivo", d_km = 0, semilla = NA, sel = rep(TRUE, nrow(p)), tgb = TRUE))
for (col in grep("^k[0-9.]+_s[0-9]+$", names(p), value = TRUE)) {
  km <- as.numeric(sub("^k([0-9.]+)_s.*", "\\1", col)); s <- as.integer(sub(".*_s", "", col))
  trat[[length(trat) + 1]] <- list(nombre = "filtrado", d_km = km, semilla = s, sel = p[[col]] == 1, tgb = FALSE)
}

filas <- list()
for (t in trat) {
  fondo_env <- if (t$tgb) tg_env else bg_env
  fondo_b <- if (t$tgb) tg_b else bg_b
  for (k in 0:4) {
    # k = 0: modelo con todos los datos del tratamiento (correlación con el esfuerzo).
    tr <- if (k == 0) t$sel else t$sel & p_b != k
    te <- if (k == 0) rep(FALSE, nrow(p)) else p_b == k
    fila <- data.frame(archivo = base, label = label, tratamiento = t$nombre, d_km = t$d_km,
                       semilla = t$semilla, bloque = k, n_entrena = sum(tr), n_prueba = sum(te),
                       auc_prueba = NA, auc_entrena = NA, auc_dif = NA, or10 = NA, boyce = NA,
                       rho_esfuerzo = NA, error = "")
    if (sum(tr) < 10 || (k > 0 && sum(te) < 5)) { fila$error <- "pocos registros"; filas[[length(filas) + 1]] <- fila; next }
    m <- tryCatch(ajusta(p_env[tr, ], fondo_env[if (k == 0) TRUE else fondo_b != k, ]), error = function(e) e)
    if (inherits(m, "error")) { fila$error <- conditionMessage(m); filas[[length(filas) + 1]] <- fila; next }
    if (k == 0) {
      fila$rho_esfuerzo <- suppressWarnings(cor(predice(m, bg_env), bg_esfuerzo, method = "spearman"))
    } else {
      s_tr <- predice(m, p_env[tr, ]); s_te <- predice(m, p_env[te, ])
      s_bg <- predice(m, bg_env[bg_b == k, ]); s_bgtr <- predice(m, bg_env[bg_b != k, ])
      fila$auc_prueba <- auc(s_te, s_bg)
      fila$auc_entrena <- auc(s_tr, s_bgtr)
      fila$auc_dif <- fila$auc_entrena - fila$auc_prueba
      fila$or10 <- mean(s_te < quantile(s_tr, 0.1))
      fila$boyce <- boyce(s_bg, s_te)
    }
    filas[[length(filas) + 1]] <- fila
  }
}
res <- do.call(rbind, filas)
out <- file.path(out_dir, paste0(base, ".csv"))
write.csv(res, out, row.names = FALSE)
writeLines(vars, file.path(out_dir, "predictores.txt"))
cat(sprintf("[%s] %d presencias, predictores %s -> %s\n", base, nrow(p), paste(vars, collapse = ","), out))
