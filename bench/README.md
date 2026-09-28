# bench/ — Benchmark en GPU (diseño)

Mide la consulta por radio geodésico en tres estructuras espaciales en GPU
(**k-d tree, LBVH, grilla uniforme**) sobre registros reales, en las dos
tareas de los flujos de SDM que se apoyan en esa consulta. Las estructuras
vienen de la biblioteca del trabajo previo (CONCAPAN 2026), enlazada como
biblioteca externa: **aquí no se modifica su código**.

## Cargas de trabajo

| | Tarea biológica | Consulta | Entrada | Salida exacta de referencia |
|---|---|---|---|---|
| **W1** | Densidad de registros del grupo objetivo (esfuerzo de muestreo; *target-group*) | contar vecinos a < d, todos los registros | `<label>_xyz_f64.npy` de nubes `_dup`, `e1_*`, `e2_*`, `*_gt` | `<label>_count_<d>km[_f32c].npy` (int32) |
| **W2** | Filtrado espacial **por especie** | voraz paralelo por prioridad (abajo) | `<label>_xyz_f64.npy` + `<label>_otu.npy` (conjuntos sin repetidas, `*_gt`) | `<label>_thin_<regla>_<d>km_s<semilla>.npy` (bool) |

Las referencias las genera `scripts/05_references.py` en CPU. La biblioteca solo
trabaja en float32, así que hay **dos** referencias: `--coords f64` (la verdad
geodésica) y `--coords f32c` (sufijo `_f32c`: las mismas coordenadas float32
centradas que recibe la GPU, promovidas a float64). Contra `_f32c` se exige
igualdad exacta (errores de implementación); contra f64 se **reporta** la
diferencia (efecto de la cuantización float32). Medido en CPU: atún a 10/25/50
km, 36/40/59 conteos distintos de 80 163 y **cero** máscaras W2 distintas;
tortuga, 2 conteos a 10 km.

Vecinos si d < r estricto. La biblioteca compara `<=`, y en float32 escalar el
radio por (1 − 1e-12) no cambia nada: pasar `nextafterf(r, 0.f)` y confirmarlo
contra `_f32c`. Los pares a distancia 0 (repetidos) sí son vecinos; en W1 se
resta solo el propio punto.

Grilla densa: calcular el número de celdas en 64 bits **antes** de construir.
Si supera 2³²−1 o la memoria libre, registrar la fila como `no ejecutable` con
el número de celdas (es un resultado: con datos globales la grilla densa
reserva memoria para el interior de la Tierra). El atún a 10 km son ~2,07×10⁹
celdas (~16,6 GB): cabe en la L40S (48 GB) y se reporta su memoria. La
validación es **igualdad exacta** de conteos (W1) y de conjuntos (W2), no de
promedios. Para N > 400 000 no hay referencia W1 en CPU (tardaría horas con
registros apilados en la misma coordenada): ahí se exige que las **tres
estructuras den conteos idénticos entre sí**. El tiempo de la referencia W1
(`<label>_cpu_w1.json`, cKDTree con todos los núcleos asignados) es la línea
base en CPU para la densidad; GeoThinneR (`run_geothinner.sh`) es la línea
base para el filtrado.

## Distancia

Coordenadas cartesianas 3D sobre la esfera (R = 6 371 008,8 m). Vecinos si la
distancia euclidiana es **estrictamente menor** que el radio de cuerda
`2R·sin(d/2R)`, que equivale exactamente a la distancia great-circle `d`
(radios precalculados en `<label>_report.json`). Si la estructura trabaja en
float32, usar `<label>_xyz_f32_centered.npy` y reportar cuántos conteos
difieren de la referencia (en datos globales, como el atún, centrar no ayuda:
la resolución float32 cerca de R es ~0,5 m).

## W2: filtrado voraz paralelo por prioridad

Filtrar a distancia d es elegir un conjunto independiente **maximal** en el
grafo "a menos de d", por especie. El voraz secuencial en un orden fijo se
calcula en paralelo por rondas y da **el mismo conjunto** (Blelloch, Fineman y
Shun, SPAA 2012):

```
estado[v] = SIN_DECIDIR para todo v con otu[v] >= 0
repetir hasta que no quede SIN_DECIDIR:
  A) para cada v SIN_DECIDIR (un hilo por punto):
       raíz[v] = no existe u SIN_DECIDIR, otu[u] == otu[v], dist(u,v) < d,
                 con clave(u) > clave(v)          (salida temprana al hallar uno)
  B) para cada v con raíz[v]: estado[v] = DENTRO
  C) para cada v SIN_DECIDIR:
       si existe u DENTRO, otu[u] == otu[v], dist(u,v) < d: estado[v] = FUERA
                                                  (salida temprana)
  rondas += 1
```

Claves (comparación lexicográfica, sin ordenar nada en GPU):

- `random`: `(h(v), v)`, mayor primero
- `mindeg`: `(-grado(v), h(v), v)`, mayor primero; `grado(v)` = vecinos de la
  **misma especie** a < d (un conteo tipo W1 con filtro de especie)

`h(v) = splitmix64(v XOR splitmix64(semilla))`, con `v` = fila global en el
arreglo del conjunto. `splitmix_vectors.csv` trae valores de prueba: la
implementación CUDA debe reproducirlos bit a bit.

```
splitmix64(x): x += 0x9E3779B97F4A7C15
               z = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
               z = (z ^ (z >> 27)) * 0x94D049BB133111EB
               return z ^ (z >> 31)
```

Si la biblioteca no permite visitar vecinos con salida temprana, la
alternativa aceptable es materializar las listas de vecinos de la misma
especie en CSR (conteo + llenado con la biblioteca) y correr las rondas sobre
el CSR. Declarar cuál de las dos variantes se usó: cambia la memoria.

## Qué se mide

Por (conjunto, d, estructura, carga): tiempo de construcción, tiempo de
consulta (W1) o total y rondas (W2), memoria GPU pico, vecinos totales,
**ns por vecino verdadero** (consulta / vecinos: sin esta normalización, más
agrupamiento parece "más lento" solo porque devuelve más vecinos), retenidos
(W2) y validación. 1 corrida de calentamiento + 10 medidas; reportar mediana e
IQR. Celda de la grilla = radio de consulta.

Salida: `bench/results/<fecha>_<gpu>.csv` con columnas

```
label,N,d_km,estructura,carga,regla,semilla,rep,build_ms,query_ms,rondas,
vecinos_total,retenidos,mem_pico_mb,valido,discrepancias,precision,
variante_w2,lib_commit,gpu,cuda
```

`lib_commit` = hash de la biblioteca de estructuras usada (reproducibilidad).
En W2, `query_ms` es el tiempo **total** del filtrado (todas las rondas, sin la
construcción). `scripts/07_analyze_bench.py` convierte estos CSV en las tablas y
la figura del paper.

## Orden de ejecución (límite de 4 h en nukwa-l40s)

1. `tortuga_gt`, `atun_gt` a 10/25/50 km (W1 + W2): los más chicos; validan todo.
2. `e1_*` a 1/5/10 km (W1): efecto del agrupamiento a N fijo.
3. `e2_aves_*` y `aves_2018_dup` a 1/5/10 km (W1): escala.
4. W2 por especie en `aves_2018`, `amphibia`, `plantae`, `insecta_2018`.

## Validar la salida de GeoThinneR

`run_geothinner.sh` guarda la máscara de la primera repetición de cada
(método, d). `python3 scripts/06_check_geothinner.py` le aplica los mismos
criterios que a la referencia: cero pares a menos de d y cuántos registros
descartados no tienen ningún vecino conservado (si hay, el conjunto no es
maximal). En nuestra reimplementación de su regla aparecen entre 0 y 155 por
corrida en tortuga y atún; esto confirma si pasa también con el paquete.
