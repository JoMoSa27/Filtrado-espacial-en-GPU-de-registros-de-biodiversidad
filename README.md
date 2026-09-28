# Filtrado espacial en GPU de registros de biodiversidad

**Póster para BIP 2026** — IEEE International Conference on BioInspired
Processing. Fecha límite de envío: 28 de septiembre de 2026.

Este repositorio contiene el pipeline de datos y el benchmark de un trabajo que
mide **qué estructura espacial en GPU filtra más rápido registros de
biodiversidad de Costa Rica**, y si el grado de agrupamiento de esos registros
permite predecirlo antes de construir la estructura.

---

## 1. El problema

Los modelos de distribución de especies (SDM) se entrenan con registros de
presencia de [GBIF](https://www.gbif.org): "se observó esta especie en estas
coordenadas". Esos registros tienen **sesgo de muestreo**. Se amontonan donde la
gente observa —ciudades, carreteras, estaciones biológicas, rutas de
observación de aves— y dejan vacíos enormes donde nadie fue. Entrenar con ellos
tal cual enseña al modelo dónde muestrea la gente, no dónde vive la especie.

La corrección estándar es el **filtrado espacial** (*spatial thinning*):
descartar registros que estén a menos de una distancia mínima `d` de otro
registro ya conservado, para que la densidad de la muestra deje de reflejar el
esfuerzo de observación. Computacionalmente son dos pasos:

1. Para cada punto, una **búsqueda por radio** `d`: encontrar sus vecinos.
2. Una **selección voraz** que conserva un subconjunto sin pares a menos de `d`.

El paso 1 domina el costo. Y es, exactamente, la consulta que ya teníamos
implementada en GPU.

## 2. De dónde parte: el trabajo previo

Este póster es la continuación directa de un artículo **[aceptado/enviado]**
<!-- VERIFICAR: confirmar el estado antes de publicar --> en **CONCAPAN 2026**:

> *Measuring Distributional Degradation in GPU Spatial Indices for Nearest-Neighbor Search*


Ese trabajo implementó tres estructuras espaciales en CUDA —**k-d tree**,
**LBVH** (jerarquía de volúmenes envolventes lineal) y **grilla uniforme**— y
las comparó en un diseño factorial controlado sobre distribuciones sintéticas,
verificando cada resultado contra fuerza bruta en CPU. Los hallazgos que
importan aquí:

- La **grilla construye mucho más rápido** que el k-d tree — hasta 71× a un
  millón de puntos — porque su construcción es un `count + scan + fill` sin
  ordenamiento comparativo.
  <!-- VERIFICAR: cifra 71x y el N al que corresponde -->
- Pero **su consulta se derrumba cuando los datos están agrupados**. Con puntos
  uniformes el tiempo por consulta es prácticamente constante en N; con una
  mezcla de gaussianas pasa a crecer más rápido que lineal. La grilla deja de
  ser sublineal: es un cambio de régimen, no ruido.
  <!-- VERIFICAR: exponentes del ajuste de potencia que sostienen "sublineal" y
       "más rápido que lineal" -->
- El **k-d tree se mantiene estable** frente al mismo cambio de distribución
  <!-- VERIFICAR: margen de estabilidad reportado para el k-d tree -->,
  porque se adapta a la densidad: sus cortes siguen a los datos, mientras que
  la retícula de la grilla es ciega a ellos.

**La conexión con este póster:** el agrupamiento con vacíos que hace *necesario*
el filtrado espacial es el mismo tipo de distribución que degradó la grilla en
aquel trabajo. Si ese resultado se sostiene, la elección de estructura para
filtrar registros de biodiversidad no es indiferente — y debería poder
anticiparse midiendo el agrupamiento de la entrada.

Aquel trabajo usó datos sintéticos. Este los lleva a **datos biológicos reales**.

## 3. El hueco

| Herramienta | Enfoque | Límite |
|---|---|---|
| `spThin` (R, 2015) | fuerza bruta | se quedó sin memoria en una máquina de 128 GB con 80 163 registros, según Mestre-Tomás (2026) |
| `GeoThinneR` (R Journal, 2026) | k-d trees en **CPU** | su benchmark simulado llegó a 50 000 puntos; su método `k_estimation` se degrada con agrupamiento fuerte |

> J. Mestre-Tomás, "GeoThinneR: An R Package for Efficient Spatial Thinning of
> Species Occurrences and Point Data," *The R Journal*, vol. 18, no. 1,
> pp. 299–314, 2026, doi: 10.32614/RJ-2026-006.

**No encontramos ninguna herramienta publicada de filtrado espacial en GPU.**
Ese es el hueco.

## 4. Objetivos

**General.** Determinar si la distribución espacial real de los registros decide
qué estructura en GPU filtra más rápido, y si un estadístico de agrupamiento
permite predecirlo **antes** de construir la estructura.

| | |
|---|---|
| **OE1** | Verificar con datos reales que la grilla se degrada al aumentar el agrupamiento mientras el k-d tree se mantiene estable. |
| **OE2** | Comparar predictores de agrupamiento (Morisita, ocupación máxima por celda, hacinamiento medio, K de Ripley) contra el costo medido en GPU. |
| **OE3** | Comparar tiempo, memoria y puntos retenidos contra GeoThinneR. |
| **OE4** | Filtrar **por especie** con un algoritmo paralelo exacto (voraz por prioridad, Blelloch *et al.* 2012) cuya salida en GPU se valida conjunto por conjunto contra la referencia en CPU. |

La consulta por radio sirve a **dos** pasos estándar de los flujos de SDM, y
cada uno define una carga de trabajo:

- **Filtrado espacial por especie** (spThin, GeoThinneR con grupos): un
  registro solo elimina a otros de su misma especie.
- **Densidad de registros del grupo objetivo** (esfuerzo de muestreo; fondo
  *target-group* de Phillips *et al.* 2009): cuántos registros de todo el
  taxón hay a menos de `d`. Aquí se cuentan **todos** los registros, incluidas
  las visitas repetidas, porque cada visita es esfuerzo.

El trabajo se plantea como **comparación de predictores**, no como defensa de
ninguno. Es posible que la ocupación máxima prediga mejor que Morisita —en GPU
un warp se atasca en la peor celda, así que puede mandar la cola y no el
promedio— y ese sería un resultado igual de publicable.

El aporte es **computacional, no ecológico**. No afirmamos que filtrar mejore
los modelos de distribución de especies: Ten Caten *et al.* (2023, *Ecosphere*)
encuentran que a menudo los empeora. Afirmamos que *si* se va a filtrar, así se
hace rápido.

## 5. Decisión técnica que conviene conocer de entrada

Las coordenadas geográficas se convierten a **cartesianas 3D sobre la esfera**
(R = 6 371 008,8 m, radio medio WGS84). Para consultar a una distancia geodésica
`d` se usa el **radio de cuerda**:

```
chord(d) = 2 · R · sin(d / 2R)
```

que devuelve **exactamente** los mismos vecinos que la distancia great-circle,
porque la cuerda crece monótonamente con el arco. Esto permite reutilizar la
métrica euclidiana que las tres estructuras ya tienen, sin escribir un kernel
haversine — a cambio de que todo el trabajo pase a ser tridimensional.

**Precisión.** float32 cerca de 6 371 km resuelve apenas ~0,5 m. Se construye y
se valida en float64; `02_prepare.py` emite además una versión float32 con el
centroide restado, que recupera la precisión para datos regionales como Costa
Rica. Para datos globales centrar no ayuda (el centroide cae dentro de la
Tierra), así que ahí manda el float64.

## 6. Los tres experimentos

| | Qué aísla | Entrada |
|---|---|---|
| **E1** | El efecto del agrupamiento a N fijo (densidad del grupo) | Los 4 taxones **con registros repetidos** submuestreados al mismo N (lo fija Amphibia). Como N es igual, lo único que cambia entre taxones es la forma del agrupamiento. |
| **E2** | El escalamiento: dónde se justifica la GPU | Aves 2018 con repetidos en 50 k, 100 k, 300 k, 1 M y completo (1,59 M). |
| **E3** | La comparación con el baseline publicado | Tortuga y atún con **exactamente** la entrada que usó GeoThinneR (`*_gt`: sin filtro de incertidumbre ni de duplicados), a sus mismos radios. |
| **E4** | Filtrado por especie a escala nacional | Cada especie (o BIN de BOLD) de los 4 taxones, a 1, 5 y 10 km: retenidos por regla, rondas paralelas, validación, costo en CPU. |

Radios: **1, 5 y 10 km** para Costa Rica; **10, 25 y 50 km** para el atún.
Morisita y Ripley de los taxones de Costa Rica usan un **área de estudio común**
(`geo.CR_RECT`); con un rectángulo por conjunto, los taxones no serían
comparables.

## 7. Datos

| Conjunto | N aprox. | Papel |
|---|---|---|
| Aves 2018 (GBIF Costa Rica) | 1,60 M | Volumen y barrido de N |
| Plantae (GBIF Costa Rica) | 1,05 M | Segundo conjunto grande, otra fuente (herbarios) |
| Insecta 2018 (GBIF Costa Rica) | 248 k | Gradiente de agrupamiento |
| Amphibia (GBIF Costa Rica) | 89 k | Extremo menos denso; fija el N común de E1 |
| Tortuga boba, atún aleta amarilla | 8 k / 80 k | Comparación directa con GeoThinneR |

Se espera que los taxones den niveles de agrupamiento muy distintos: Aves está
dominado por eBird y debería concentrarse en rutas de observación, mientras que
Insecta y Amphibia, al venir de colectas, deberían salir más dispersos. **Es una
hipótesis, no un dato**: el agrupamiento real de cada conjunto se mide con
Morisita y se verifica en E1. Ese contraste, si se confirma, *es* la variable
del estudio.

Las descargas de GBIF **no** están en el repositorio: se reconstruyen desde sus
DOI con la sección siguiente. Cada carpeta de `data/raw/` lleva su
`CITATION.txt` con la consulta exacta y el DOI. Los dos conjuntos de GeoThinneR
sí van versionados (558 KB) porque son la entrada exacta con la que ese paper
publicó sus tiempos; su procedencia y SHA-256 están en
`data/raw/geothinner/SOURCE.md`.

## 8. Reproducir los datos

### 8.1 Los filtros de GBIF

Comunes a las cuatro descargas, aplicados del lado del servidor:

| Filtro | Valor |
|---|---|
| Country or area | Costa Rica |
| Location | *Including coordinates* y *Exclude known geospatial issues* |
| Occurrence status | *Present* |
| Basis of record | los 7 que **no** son *Fossil specimen* ni *Living specimen* |
| Scientific name | un solo taxón por descarga |
| Year | 2018 en Aves e Insecta; sin filtro en Plantae y Amphibia |

Se excluyen `FOSSIL_SPECIMEN` y `LIVING_SPECIMEN` porque sus coordenadas son las
del museo o el zoológico, y crearían cúmulos artificiales gigantes en un solo
punto — que es justo la variable que se mide.

Un taxón por descarga, nunca varios juntos: cada taxón es un conjunto distinto
en el estudio, y bajarlos juntos mezclaría el DOI.

Aves supera los 3 M de registros, así que se restringió a un año. Se eligió
**2018**: deja Aves en 1,60 M, es anterior a la pandemia (que redujo el
aviturismo que alimenta eBird en 2020–2021) y es un año cerrado, sin registros
tardíos por llegar. El filtro queda dentro de la consulta de GBIF, así que el
DOI describe exactamente lo que se usó.

### 8.2 Las cuatro descargas

Pedidas a GBIF el 25 de septiembre de 2026; ya tienen DOI asignado, así que
**no hay que volver a pedirlas**.

| Conjunto | Registros | Clave GBIF | DOI |
|---|---|---|---|
| Aves 2018 | 1 597 958 | `0007871-260921141020460` | `10.15468/dl.xdw99k` |
| Plantae | 1 052 323 | `0007891-260921141020460` | `10.15468/dl.mbz7g2` |
| Insecta 2018 | 247 581 | `0007865-260921141020460` | `10.15468/dl.e5pzwg` |
| Amphibia | 89 058 | `0007872-260921141020460` | `10.15468/dl.4awycg` |

Son ~383 MB comprimidos. Se bajan **donde se vayan a procesar** — en un clúster,
desde el nodo de login, nunca dentro de un trabajo de SLURM: los nodos de
cómputo normalmente no tienen salida a internet.

```bash
cd data/raw
B=https://api.gbif.org/v1/occurrence/download/request

wget -O gbif_cr_aves_2018/descarga.zip    "$B/0007871-260921141020460.zip"
wget -O gbif_cr_plantae/descarga.zip      "$B/0007891-260921141020460.zip"
wget -O gbif_cr_insecta_2018/descarga.zip "$B/0007865-260921141020460.zip"
wget -O gbif_cr_amphibia/descarga.zip     "$B/0007872-260921141020460.zip"

for d in gbif_cr_*/; do (cd "$d" && unzip -o -q descarga.zip && rm descarga.zip); done
```

### 8.3 Verificar antes de seguir

```bash
wc -l gbif_cr_*/*.csv
```

Tiene que dar **registros + 1** (la cabecera):

| Conjunto | Esperado |
|---|---|
| Aves 2018 | 1 597 959 |
| Plantae | 1 052 324 |
| Insecta 2018 | 247 582 |
| Amphibia | 89 059 |

Si alguno no cuadra, la descarga se cortó: hay que volver a bajar ese conjunto.
El archivo de datos se llama `<clave>.csv` y, pese a la extensión, es TSV — que
es lo que `02_prepare.py` espera.

**Caducidad.** GBIF conserva el archivo de estas descargas hasta el **25 de
marzo de 2027**. El dueño de la cuenta puede pedir una extensión desde
`https://www.gbif.org/occurrence/download/<clave>`. En la práctica GBIF suele
conservar indefinidamente las descargas cuyo DOI ha sido citado.

Ojo: **relanzar la consulta no sirve para recuperar el archivo** — crea una
descarga nueva, con otra clave y otro DOI, y por tanto un conjunto distinto del
que cita el paper.

### 8.4 Ambiente

```bash
python -m venv ~/venvs/bip && source ~/venvs/bip/bin/activate
pip install numpy pandas scipy matplotlib pyarrow pyreadr requests numba
```

`pyarrow` es opcional: sin él, `02_prepare.py` escribe `.csv.gz` en vez de
parquet.

### 8.5 Correr el pipeline

```bash
source ~/venvs/bip/bin/activate
cd bip2026

sbatch scripts/run_pipeline.sh     # en un clúster con SLURM
bash   scripts/run_pipeline.sh     # en cualquier otra máquina
```

Recorre cada conjunto que tenga datos —explorar, convertir con y sin
duplicados, calcular predictores— y arma las entradas de E1, E2 y E3. Salta con
un aviso los conjuntos que todavía no estén, así que se puede correr con lo que
haya y repetir cuando lleguen los demás.

Paso a paso, si hace falta mirar un conjunto suelto:

```bash
cd scripts

# 1. Mirar — reporte HTML autocontenido en ../reports/
python 00_explorar.py ../data/raw/gbif_cr_aves_2018/*.csv --label aves_2018 --radius-km 5

# 2. Convertir — con y sin duplicados
python 02_prepare.py ../data/raw/gbif_cr_aves_2018/*.csv --label aves_2018
python 02_prepare.py ../data/raw/gbif_cr_aves_2018/*.csv --label aves_2018_dup --keep-duplicates

# 3. Predictores de agrupamiento
python 03_cluster_stats.py --label aves_2018 --radii-km 1 5 10
```

`00_explorar.py` produce un reporte que recorre qué trae el archivo, dónde están
los puntos, qué descarta cada regla de limpieza, los duplicados, la conversión a
3D con un ejemplo resuelto, lo que ve la grilla y cuántos vecinos tiene cada
punto. Usa las mismas funciones que `02_prepare.py`, así que lo que se ve es lo
que se convierte.

Sobre los conjuntos de millones de filas, `00_explorar.py` y
`03_cluster_stats.py` conviene correrlos en un trabajo de SLURM corto, no en el
nodo de login.

### 8.6 Qué queda en `data/processed/`

| Archivo | Uso |
|---|---|
| `<label>_xyz_f64.npy` | `(N,3)` float64 C-contiguo, entrada de los kernels |
| `<label>_xyz_f32_centered.npy` | float32 con el centroide restado |
| `<label>_centroid_f64.npy` | el centroide, para volver a coordenadas absolutas |
| `<label>_report.json` | cuántos se descartaron en cada paso, bbox, radios de cuerda |
| `<label>_cluster_stats.json` | predictores de agrupamiento por radio (`03`) |
| `<label>_por_especie.csv/.json` | filtrado y predictores por especie (`04`) |
| `<label>_otu.npy` | id de especie/BIN por registro (-1 = sin especie) |
| `<label>_count_<d>km.npy` | referencia W1: vecinos exactos por registro (`05`) |
| `<label>_thin_<regla>_<d>km_s<semilla>.npy` | referencia W2: máscara exacta del filtrado por especie (`05`) |

## 9. Organización

```
bip2026/
├── README.md
├── scripts/
│   ├── geo.py               lectura, limpieza, proyección, Morisita 2D
│   ├── 00_explorar.py       reporte HTML que muestra cada paso con datos reales
│   ├── 01_download_gbif.py  pedir descargas nuevas a la API de GBIF
│   ├── 02_prepare.py        genera los .npy que consumen los kernels
│   ├── 03_cluster_stats.py  predictores de agrupamiento
│   ├── thin.py              filtrado de referencia en CPU (3 reglas, validación)
│   ├── 04_species_stats.py  filtrado por especie + predictores por especie
│   ├── 05_references.py     salidas exactas para validar la GPU
│   ├── 06_check_geothinner.py valida la salida real de GeoThinneR (pares < d, maximalidad)
│   ├── fig_thinning.py      figura del paper (retención y rondas)
│   ├── 07_analyze_bench.py  CSV del benchmark -> tablas, Spearman y figura GPU
│   └── run_pipeline.sh      corre todo, conjunto por conjunto
├── bench/                   diseño del benchmark CUDA y línea base GeoThinneR
├── data/raw/                descargas sin tocar — inmutable
├── data/interim/            metadatos limpios
├── data/processed/          .npy listos para los kernels
├── results/                 CSV de las corridas
└── paper/                   el manuscrito
```

Dos reglas que gobiernan el repositorio:

- **`data/raw/` es inmutable.** Todo lo derivado se regenera con
  `scripts/run_pipeline.sh`.
- **Ningún número del paper puede venir de otro lado que no sea una corrida
  guardada en disco.** Si un resultado contradice la hipótesis, se reporta tal
  cual.

La lógica de lectura, limpieza y proyección vive solo en `scripts/geo.py`: los
demás scripts la importan, no la duplican. Las estructuras espaciales viven en
**otro repositorio** y aquí se consumen como biblioteca.

## 10. Estado

| | |
|---|---|
| Pipeline de datos | completo (E1–E4 y referencias) |
| Descargas de GBIF | en Kabre |
| Filtrado de referencia en CPU (`thin.py`) | hecho y validado |
| Benchmark en GPU (`bench/`) | diseñado (`bench/README.md`), por implementar |
| Línea base GeoThinneR | script listo (`bench/run_geothinner.sh`) |
| Manuscrito | pendiente |

Formato de envío: 4–6 páginas, plantilla de *Tecnología en Marcha*, resumen y
palabras clave en español e inglés, referencias IEEE, ORCID requerido.
