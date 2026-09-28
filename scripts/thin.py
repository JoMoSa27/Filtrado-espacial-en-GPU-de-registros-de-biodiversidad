"""
thin.py — Filtrado espacial de REFERENCIA en CPU (exacto, validable).

Sirve para tres cosas:
  1. Replicar la regla de GeoThinneR/spThin para comparar puntos retenidos.
  2. Definir el resultado EXACTO que debe producir la version en GPU: el
     filtrado voraz por orden de prioridad da un conjunto unico, asi que la GPU
     se valida comparando conjuntos, no solo conteos.
  3. Medir cuantas rondas paralelas necesita ese filtrado con datos reales.

Filtrar a distancia d = elegir un conjunto de registros sin pares a menos de d
en el que no se pueda agregar ningun otro (conjunto independiente maximal en el
grafo "estan a menos de d"). Reglas implementadas:

  maxdeg    GeoThinneR::max_thinning_algorithm (y spThin): quitar repetidamente
            el punto con MAS vecinos vivos (empates al azar) hasta que nadie
            tenga vecinos. Se repite `trials` veces y se queda el que retiene mas.
  random    Voraz en orden aleatorio: recorrer los puntos en orden aleatorio y
            conservar cada uno que no tenga un vecino ya conservado. Blelloch,
            Fineman y Shun (SPAA 2012): se paraleliza en O(log^2 n) rondas y da
            EXACTAMENTE el mismo conjunto que el secuencial en ese orden.
  mindeg    Igual, pero el orden prioriza a los de MENOS vecinos (empates al
            azar), como ECL-MIS (Burtscher et al. 2018), que retiene mas.

Distancia: la misma que GeoThinneR (great-circle, vecinos si d_ij < d), via el
radio de cuerda 2R sin(d/2R) sobre coordenadas cartesianas 3D.
"""
import time

import numba as nb
import numpy as np
from scipy.spatial import cKDTree

from geo import chord_radius

# Vecinos si la distancia es ESTRICTAMENTE menor que d (como GeoThinneR brute).
_STRICT = 1.0 - 1e-12


# ----------------------------------------------------------------------------
# Grafo de vecinos
# ----------------------------------------------------------------------------
def neighbor_csr(xyz, d_m):
    """Pares a menos de d (geodesica) -> grafo en formato CSR (indptr, indices).

    Devuelve (indptr int64 [n+1], indices int32 [2P], P = pares no ordenados).
    """
    n = len(xyz)
    if n < 2:
        return np.zeros(n + 1, np.int64), np.zeros(0, np.int32), 0
    r = float(chord_radius(d_m)) * _STRICT
    pairs = cKDTree(xyz).query_pairs(r, output_type="ndarray")
    P = len(pairs)
    if P == 0:
        return np.zeros(n + 1, np.int64), np.zeros(0, np.int32), 0
    i = np.concatenate([pairs[:, 0], pairs[:, 1]]).astype(np.int32)
    j = np.concatenate([pairs[:, 1], pairs[:, 0]]).astype(np.int32)
    del pairs
    order = np.argsort(i, kind="stable")
    indices = j[order]
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(i, minlength=n), out=indptr[1:])
    return indptr, indices, P


# ----------------------------------------------------------------------------
# Regla de GeoThinneR (maxdeg) con cola de cubetas O(n + P) por intento
# ----------------------------------------------------------------------------
@nb.njit(cache=True)
def _dec(u, deg, start, order, pos):
    """Baja el grado de u en 1 moviendolo a la frontera de su cubeta."""
    c = deg[u]
    first = start[c]
    w = order[first]
    pu = pos[u]
    order[first] = u
    order[pu] = w
    pos[u] = first
    pos[w] = pu
    start[c] = first + 1
    deg[u] = c - 1


@nb.njit(cache=True)
def _maxdeg_trial(indptr, indices, seed):
    np.random.seed(seed)
    n = indptr.size - 1
    deg = np.empty(n, np.int64)
    maxd = 0
    for v in range(n):
        deg[v] = indptr[v + 1] - indptr[v]
        if deg[v] > maxd:
            maxd = deg[v]
    cnt = np.zeros(maxd + 2, np.int64)
    for v in range(n):
        cnt[deg[v]] += 1
    start = np.zeros(maxd + 2, np.int64)
    s = 0
    for c in range(maxd + 1):
        start[c] = s
        s += cnt[c]
    start[maxd + 1] = n
    fill = start.copy()
    order = np.empty(n, np.int64)
    pos = np.empty(n, np.int64)
    for v in range(n):
        p = fill[deg[v]]
        order[p] = v
        pos[v] = p
        fill[deg[v]] += 1
    alive = np.ones(n, np.bool_)
    cur = maxd
    while True:
        while cur > 0 and start[cur] == start[cur + 1]:
            cur -= 1
        if cur == 0:
            break
        k = start[cur] + np.random.randint(0, start[cur + 1] - start[cur])
        v = order[k]
        alive[v] = False
        for e in range(indptr[v], indptr[v + 1]):
            u = indices[e]
            if alive[u] and deg[u] > 0:
                _dec(u, deg, start, order, pos)
        while deg[v] > 0:
            _dec(v, deg, start, order, pos)
    return alive


def thin_maxdeg(indptr, indices, trials=10, seed=0, keep_all=False):
    """GeoThinneR: mejor de `trials` intentos (el que retiene mas; en empate,
    el primero). Devuelve (mascara, lista de retenidos por intento[, todas])."""
    best, counts, todas = None, [], []
    for t in range(trials):
        kept = _maxdeg_trial(indptr, indices, seed + t)
        c = int(kept.sum())
        counts.append(c)
        if keep_all:
            todas.append(kept)
        if best is None or c > best.sum():
            best = kept
    return (best, counts, todas) if keep_all else (best, counts)


# ----------------------------------------------------------------------------
# Voraz por prioridad: secuencial y por rondas paralelas (mismo resultado)
# ----------------------------------------------------------------------------
@nb.njit(cache=True)
def _greedy_sequential(indptr, indices, order):
    n = indptr.size - 1
    state = np.zeros(n, np.int8)          # 0 sin decidir, 1 dentro, 2 fuera
    for v in order:
        if state[v] == 0:
            state[v] = 1
            for e in range(indptr[v], indptr[v + 1]):
                u = indices[e]
                if state[u] == 0:
                    state[u] = 2
    return state == 1


@nb.njit(cache=True)
def _greedy_rounds(indptr, indices, prio):
    """Simula la version paralela: en cada ronda entra todo punto sin decidir
    que no tenga un vecino sin decidir de mayor prioridad; luego salen sus
    vecinos. Devuelve (mascara, rondas, trabajo = aristas revisadas)."""
    n = indptr.size - 1
    state = np.zeros(n, np.int8)
    mark = np.zeros(n, np.bool_)
    rounds = 0
    work = 0
    left = n
    while left > 0:
        rounds += 1
        for v in range(n):
            mark[v] = False
            if state[v] == 0:
                ok = True
                for e in range(indptr[v], indptr[v + 1]):
                    work += 1
                    u = indices[e]
                    if state[u] == 0 and prio[u] > prio[v]:
                        ok = False
                        break
                mark[v] = ok
        for v in range(n):
            if mark[v]:
                state[v] = 1
        for v in range(n):
            if state[v] == 0:
                for e in range(indptr[v], indptr[v + 1]):
                    work += 1
                    if state[indices[e]] == 1:
                        state[v] = 2
                        break
        left = 0
        for v in range(n):
            if state[v] == 0:
                left += 1
    return state == 1, rounds, work


_GOLD = np.uint64(0x9E3779B97F4A7C15)


def splitmix64(x):
    """Hash splitmix64 (Steele et al. 2014) sobre uint64. Es la fuente de
    prioridades 'aleatorias': la GPU implementa la misma funcion y obtiene el
    MISMO orden, asi que su salida se valida comparando conjuntos."""
    with np.errstate(over="ignore"):
        z = np.asarray(x, dtype=np.uint64) + _GOLD
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def record_hash(ids, seed):
    """h(id, seed) = splitmix64(id XOR splitmix64(seed))."""
    return splitmix64(np.asarray(ids, dtype=np.uint64) ^ splitmix64(np.uint64(seed)))


def priorities(indptr, rule, seed=0, ids=None):
    """Prioridad unica por punto como rango (mayor = se decide primero).

    Orden definido para que la GPU lo reproduzca sin ordenar nada, comparando
    claves por pares:
      random : h(id) mayor primero; empate -> id mayor primero
      mindeg : grado menor primero; empate -> h(id) mayor; empate -> id mayor
    'ids' = identificador global del registro (fila en el arreglo del conjunto).
    """
    n = indptr.size - 1
    ids = np.arange(n, dtype=np.uint64) if ids is None else np.asarray(ids, np.uint64)
    h = record_hash(ids, seed)
    if rule == "random":
        order = np.lexsort((ids, h))[::-1]
    elif rule == "mindeg":
        deg = np.diff(indptr).astype(np.int64)
        order = np.lexsort((ids, h, -deg))[::-1]
    else:
        raise ValueError(rule)
    prio = np.empty(n, np.int64)
    prio[order] = np.arange(n, 0, -1, dtype=np.int64)
    return prio


def thin_greedy(indptr, indices, rule="random", seed=0, check_rounds=True, ids=None):
    """Voraz por prioridad. Devuelve (mascara, rondas, trabajo).
    Si check_rounds, comprueba que las rondas paralelas den el MISMO conjunto
    que el secuencial (propiedad de Blelloch et al. 2012). 'trabajo' = aristas
    revisadas por la version paralela con salida temprana, sumando rondas."""
    prio = priorities(indptr, rule, seed, ids)
    order = np.argsort(-prio, kind="stable")
    kept = _greedy_sequential(indptr, indices, order)
    rounds = work = -1
    if check_rounds:
        kept_p, rounds, work = _greedy_rounds(indptr, indices, prio)
        if not np.array_equal(kept, kept_p):
            raise AssertionError("rondas paralelas != secuencial")
    return kept, rounds, work


# ----------------------------------------------------------------------------
# Validacion
# ----------------------------------------------------------------------------
def validate(xyz, kept, d_m, indptr, indices):
    """(pares conservados a menos de d, puntos agregables).
    Un filtrado correcto da 0 en lo primero. Lo segundo cuenta descartados sin
    ningun vecino conservado: se podrian agregar sin violar d (no maximal)."""
    r = float(chord_radius(d_m)) * _STRICT
    bad = len(cKDTree(xyz[kept]).query_pairs(r)) if kept.sum() > 1 else 0
    addable = _addable(indptr, indices, kept)
    return bad, addable


@nb.njit(cache=True)
def _addable(indptr, indices, kept):
    n = indptr.size - 1
    c = 0
    for v in range(n):
        if not kept[v]:
            has = False
            for e in range(indptr[v], indptr[v + 1]):
                if kept[indices[e]]:
                    has = True
                    break
            if not has:
                c += 1
    return c


# ----------------------------------------------------------------------------
# Todo junto para un conjunto de puntos
# ----------------------------------------------------------------------------
def run_all(xyz, d_m, trials=10, seed=0, check=True, ids=None, csr=None):
    """Corre las tres reglas y devuelve un dict con tiempos y retenidos.
    'trabajo_*' = aristas revisadas con salida temprana (todas las rondas);
    comparar con 2*pares, que es lo que cuesta materializar las listas."""
    out = {"N": int(len(xyz)), "d_km": d_m / 1000.0}
    t = time.perf_counter()
    indptr, indices, P = csr if csr is not None else neighbor_csr(xyz, d_m)
    out["t_vecinos_s"] = time.perf_counter() - t
    out["pares"] = int(P)
    deg = np.diff(indptr)
    out["grado_medio"] = float(deg.mean()) if len(deg) else 0.0
    out["grado_max"] = int(deg.max(initial=0))
    out["aislados"] = int((deg == 0).sum())

    t = time.perf_counter()
    k_md, per_trial, todas = thin_maxdeg(indptr, indices, trials, seed, keep_all=True)
    out["t_maxdeg_s"] = time.perf_counter() - t
    out["ret_maxdeg_mejor"] = int(k_md.sum())
    out["ret_maxdeg_media"] = float(np.mean(per_trial))
    if check:
        # el del mejor intento (lo que devuelve GeoThinneR) y el rango en todos
        out["maxdeg_malos"], out["maxdeg_agregables"] = validate(xyz, k_md, d_m, indptr, indices)
        ag = [validate(xyz, k, d_m, indptr, indices)[1] for k in todas]
        out["maxdeg_agregables_min"], out["maxdeg_agregables_max"] = int(min(ag)), int(max(ag))

    for rule in ("random", "mindeg"):
        rets, rnds, works = [], [], []
        t = time.perf_counter()
        malos = agreg = 0
        for s in range(trials):
            k, rounds, work = thin_greedy(indptr, indices, rule, seed + s,
                                          check_rounds=check, ids=ids)
            rets.append(int(k.sum()))
            rnds.append(rounds)
            works.append(work)
            if check:                       # se valida CADA semilla, no solo una
                b, a = validate(xyz, k, d_m, indptr, indices)
                malos += b
                agreg = max(agreg, a)
        out[f"t_{rule}_s"] = time.perf_counter() - t
        out[f"ret_{rule}_mejor"] = int(max(rets))
        out[f"ret_{rule}_media"] = float(np.mean(rets))
        out[f"rondas_{rule}_max"] = int(max(rnds))
        out[f"rondas_{rule}_media"] = float(np.mean(rnds))
        out[f"trabajo_{rule}_media"] = float(np.mean(works))
        if check:
            out[f"{rule}_malos"], out[f"{rule}_agregables"] = malos, agreg
    return out
