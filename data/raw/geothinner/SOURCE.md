# Procedencia de los archivos de esta carpeta

Descargados el 2026-09-24 desde el repositorio del paquete GeoThinneR:
https://raw.githubusercontent.com/jmestret/GeoThinneR/main/data/<archivo>.rda

| Archivo | Especie | Filas | Extensión | Años | SHA-256 |
|---|---|---|---|---|---|
| caretta.rda | Caretta caretta (tortuga boba) | 8 340 | Mediterráneo (lat 30,5–45,8; lon −5,9–35,8) | 1962–2025 | d42ba189e5af65922644eaf55dff6bc182494d34908f920ffbd1f22b1af1a6df |
| thunnus.rda | Thunnus albacares (atún aleta amarilla) | 80 163 | Global (lat −42,8–52,9) | 1950–2025 | a399876fcbfff1e253536e1bc37fb8abad17b500c8a63c45f6141ff3d261e787 |

## Origen de los datos

Son registros de GBIF que el autor de GeoThinneR descargó y **ya limpió**
(no son descargas crudas de GBIF: traen solo 3–5 columnas y la tortuga ya
viene sin duplicados). DOIs de las descargas originales, según el paper:

- Tortuga: GBIF.org (2025) GBIF Occurrence Download, https://doi.org/10.15468/dl.9jcjrm
- Atún: GBIF.org (2025) GBIF Occurrence Download, https://doi.org/10.15468/dl.xsyrkh

Referencia: J. Mestre-Tomás, "GeoThinneR: An R Package for Efficient Spatial
Thinning of Species Occurrences and Point Data," The R Journal, vol. 18,
no. 1, pp. 299–314, 2026, doi: 10.32614/RJ-2026-006.

## Para qué se usan

Únicamente como **conjuntos de comparación con GeoThinneR**: son exactamente
los datos con los que ese paper reporta sus tiempos, así que ambos métodos se
miden sobre la misma entrada. No aportan volumen (69 k puntos tras quitar
duplicados) ni datos de Costa Rica (0 y 9 registros dentro del país).
