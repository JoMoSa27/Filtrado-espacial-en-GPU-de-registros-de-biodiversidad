/**
 * @file w2.cu
 * @brief W2 — filtrado espacial por especie: voraz paralelo por prioridad.
 *
 * Filtrar a distancia d es elegir un conjunto independiente MAXIMAL en el grafo
 * "estan a menos de d", por especie. El voraz secuencial en un orden fijo se
 * calcula por rondas en paralelo y da EXACTAMENTE el mismo conjunto (Blelloch,
 * Fineman y Shun, SPAA 2012), asi que la GPU se valida comparando conjuntos con
 * la referencia de scripts/thin.py, no promedios.
 *
 * VARIANTE: CSR MATERIALIZADO
 *   El diseno admite dos caminos y este usa el segundo, porque el primero no es
 *   posible: la biblioteca no expone visitor ni salida temprana por vecino, solo
 *   count_in_radius (conteo exacto) y range_query (lista, con tope). Asi que el
 *   grafo se materializa una vez y las rondas corren sobre el:
 *
 *     1. cnt[v] = count_in_radius(v, r)      conteo exacto, sin tope
 *     2. scan exclusivo                       -> desplazamientos del CSR
 *     3. range_query(v, r, csr+off[v], cnt[v]) sin truncamiento: la capacidad
 *                                              sale del conteo, no de una
 *                                              estimacion
 *     4. compactar en sitio: fuera el propio v y los de otra especie -> deg[v]
 *
 *   El paso 4 compacta sobre el mismo buffer. Es seguro: deg[v] <= cnt[v], asi
 *   que el desplazamiento nuevo de cada fila nunca supera al viejo, y la
 *   escritura de la fila v termina antes de donde empieza la lectura de v+1.
 *
 *   Lo que se paga: el CSR guarda todas las aristas por adelantado, mientras
 *   que la version con salida temprana revisa entre 63 y 80 % de ellas (medido
 *   en CPU). Queda declarado en la columna variante_w2.
 *
 * UNA SOLA ESTRUCTURA, NO UN BOSQUE
 *   Las especies tienen tamanos dispares (Plantae: 15 307 especies, mediana 19;
 *   Aves: maximo 27 450), y KDForest exige arboles del MISMO tamano y es 2D.
 *   Asi que se construye una estructura sobre todos los puntos y el filtro
 *   otu[u] == otu[v] se aplica al compactar. En los conjuntos *_gt hay una sola
 *   especie, asi que el filtro no descarta nada.
 *
 * PRIORIDADES SIN ORDENAR NADA
 *   h(v) = splitmix64(v XOR splitmix64(semilla)), con v = fila GLOBAL, igual
 *   que en la referencia (05_references.py pasa ids=idx, los indices globales).
 *   Las claves se comparan por pares, lexicograficamente:
 *     random : (h(v), v)            mayor primero
 *     mindeg : (-deg(v), h(v), v)   mayor primero
 *
 * USO
 *   bench_w2 --label tortuga_gt --radii-km 10 25 50 --rules random mindeg \
 *            --seeds 0 --processed ../data/processed --out ../bench/results/x.csv
 */

#include <algorithm>
#include <string>
#include <vector>

#include <cub/cub.cuh>

#include "bench_common.cuh"
#include "npy.hpp"

#include "SpatialTags.cuh"
#include "Metric.cuh"
#include "KDTree2D.cuh"
#include "UniformGrid2D.cuh"
#include "LinearBVH.cuh"

using Metrica = Cartesian<3>;

/// 1 calentamiento + 10 medidas, igual que W1 (bench/README.md).
static constexpr int REPS_W2 = 10;

// ── splitmix64 ───────────────────────────────────────────────────────────────
// Steele, Lea y Flood (2014). Se valida bit a bit contra bench/splitmix_vectors.csv.

__host__ __device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

__host__ __device__ __forceinline__ uint64_t record_hash(uint64_t id, uint64_t seed) {
    return splitmix64(id ^ splitmix64(seed));
}

// ── Estado del voraz ─────────────────────────────────────────────────────────

enum : int8_t { SIN_DECIDIR = 0, DENTRO = 1, FUERA = 2 };

/**
 * clave(u) > clave(v), lexicografica, sin materializar la clave.
 *
 * random : (h, id)        mayor primero
 * mindeg : (-deg, h, id)  mayor primero  ==  grado menor primero
 */
__device__ __forceinline__ bool clave_mayor(uint32_t u, uint32_t v,
                                            const uint64_t* h, const int32_t* deg,
                                            bool usar_grado)
{
    if (usar_grado && deg[u] != deg[v]) return deg[u] < deg[v];  // -deg mayor
    if (h[u] != h[v]) return h[u] > h[v];
    return u > v;
}

// ── Kernels del CSR ──────────────────────────────────────────────────────────

template <typename DS>
__global__ void contar_k(DS ds, CoordPtrs<3> pts, float radio,
                         uint32_t* cnt, uint32_t* n_overflow, uint32_t n)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    float q[3] = {pts.c[0][i], pts.c[1][i], pts.c[2][i]};
    bool ov = false;
    cnt[i] = ds.count_in_radius(q, radio, &ov);
    if (ov) atomicAdd(n_overflow, 1u);
}

template <typename DS>
__global__ void llenar_k(DS ds, CoordPtrs<3> pts, float radio,
                         const uint64_t* off, const uint32_t* cnt,
                         uint32_t* csr, uint32_t n)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    float q[3] = {pts.c[0][i], pts.c[1][i], pts.c[2][i]};
    ds.range_query(q, radio, csr + off[i], cnt[i]);
}

/**
 * Compacta cada fila en sitio: fuera el propio v y los de otra especie.
 * Escribe deg[v]. Los desplazamientos finales se obtienen con otro scan.
 */
__global__ void compactar_k(const uint64_t* off, const uint32_t* cnt,
                            const int32_t* otu, uint32_t* csr,
                            int32_t* deg, uint32_t n)
{
    const uint32_t v = threadIdx.x + blockIdx.x * blockDim.x;
    if (v >= n) return;
    const uint64_t o = off[v];
    const uint32_t c = cnt[v];
    const int32_t  ov = otu[v];
    uint32_t k = 0;
    for (uint32_t j = 0; j < c; j++) {
        const uint32_t u = csr[o + j];
        if (u != v && otu[u] == ov) csr[o + k++] = u;
    }
    deg[v] = static_cast<int32_t>(k);
}

/** Mueve cada fila compactada a su desplazamiento definitivo. */
__global__ void reubicar_k(const uint64_t* off_viejo, const uint64_t* off_nuevo,
                           const int32_t* deg, const uint32_t* src,
                           uint32_t* dst, uint32_t n)
{
    const uint32_t v = threadIdx.x + blockIdx.x * blockDim.x;
    if (v >= n) return;
    const uint64_t a = off_viejo[v], b = off_nuevo[v];
    const int32_t  d = deg[v];
    for (int32_t j = 0; j < d; j++) dst[b + j] = src[a + j];
}

// ── Kernels de las rondas ────────────────────────────────────────────────────

__global__ void hash_k(uint64_t* h, uint64_t seed, uint32_t n) {
    const uint32_t v = threadIdx.x + blockIdx.x * blockDim.x;
    if (v >= n) return;
    h[v] = record_hash(v, seed);          // v = fila global, como la referencia
}

/** Ronda A: marca los que no tienen vecino SIN_DECIDIR de mayor clave. */
__global__ void ronda_a_k(const uint64_t* off, const int32_t* deg,
                          const uint32_t* csr, const int8_t* estado,
                          const uint64_t* h, const int32_t* grado,
                          bool usar_grado, uint8_t* raiz, uint32_t n)
{
    const uint32_t v = threadIdx.x + blockIdx.x * blockDim.x;
    if (v >= n) return;
    raiz[v] = 0;
    if (estado[v] != SIN_DECIDIR) return;
    const uint64_t o = off[v];
    const int32_t  d = deg[v];
    for (int32_t j = 0; j < d; j++) {           // salida temprana
        const uint32_t u = csr[o + j];
        if (estado[u] == SIN_DECIDIR && clave_mayor(u, v, h, grado, usar_grado)) return;
    }
    raiz[v] = 1;
}

/** Ronda B: los marcados entran. */
__global__ void ronda_b_k(const uint8_t* raiz, int8_t* estado, uint32_t n) {
    const uint32_t v = threadIdx.x + blockIdx.x * blockDim.x;
    if (v >= n) return;
    if (raiz[v]) estado[v] = DENTRO;
}

/** Ronda C: sale todo SIN_DECIDIR con un vecino DENTRO. Cuenta los que quedan. */
__global__ void ronda_c_k(const uint64_t* off, const int32_t* deg,
                          const uint32_t* csr, int8_t* estado,
                          uint32_t* n_sin_decidir, uint32_t n)
{
    const uint32_t v = threadIdx.x + blockIdx.x * blockDim.x;
    if (v >= n) return;
    if (estado[v] != SIN_DECIDIR) return;
    const uint64_t o = off[v];
    const int32_t  d = deg[v];
    for (int32_t j = 0; j < d; j++) {           // salida temprana
        if (estado[csr[o + j]] == DENTRO) { estado[v] = FUERA; return; }
    }
    atomicAdd(n_sin_decidir, 1u);
}

/** Excluye del filtrado los registros sin especie ni BIN (otu = -1). */
__global__ void marcar_sin_otu_k(const int32_t* otu, int8_t* estado, uint32_t n) {
    const uint32_t v = threadIdx.x + blockIdx.x * blockDim.x;
    if (v >= n) return;
    if (otu[v] < 0) estado[v] = FUERA;
}

// ── Carga ────────────────────────────────────────────────────────────────────

struct Nube {
    uint32_t           n = 0;
    std::vector<float> x, y, z;
    std::vector<int32_t> otu;
    float              mn[3]{}, mx[3]{};
    CoordPtrs<3>       dev{};
    float*             dev_raw[3]{};
    int32_t*           d_otu = nullptr;

    void subir() {
        for (int d = 0; d < 3; d++) CUDA_CHECK(cudaMalloc(&dev_raw[d], n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dev_raw[0], x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_raw[1], y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_raw[2], z.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        for (int d = 0; d < 3; d++) dev.c[d] = dev_raw[d];
        CUDA_CHECK(cudaMalloc(&d_otu, n * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(d_otu, otu.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
    }
    void liberar() {
        for (int d = 0; d < 3; d++) if (dev_raw[d]) cudaFree(dev_raw[d]);
        if (d_otu) cudaFree(d_otu);
    }
};

static Nube cargar(const std::string& processed, const std::string& label) {
    size_t filas = 0, cols = 0;
    std::vector<float> v =
        npy::load<float>(processed + "/" + label + "_xyz_f32_centered.npy", &filas, &cols);
    if (cols != 3) throw std::runtime_error("w2: se esperaba (N,3)");

    Nube c;
    c.n = static_cast<uint32_t>(filas);
    c.x.resize(filas); c.y.resize(filas); c.z.resize(filas);
    for (int d = 0; d < 3; d++) { c.mn[d] = 1e30f; c.mx[d] = -1e30f; }
    for (size_t i = 0; i < filas; i++) {
        const float t[3] = {v[3 * i], v[3 * i + 1], v[3 * i + 2]};
        c.x[i] = t[0]; c.y[i] = t[1]; c.z[i] = t[2];
        for (int d = 0; d < 3; d++) {
            c.mn[d] = std::min(c.mn[d], t[d]);
            c.mx[d] = std::max(c.mx[d], t[d]);
        }
    }
    const std::string po = processed + "/" + label + "_otu.npy";
    if (npy::exists(po)) {
        size_t f2 = 0, c2 = 0;
        c.otu = npy::load<int32_t>(po, &f2, &c2);
        if (c.otu.size() != filas) throw std::runtime_error("w2: _otu.npy no cuadra con N");
    } else {
        // Sin archivo de especies se asume una sola, que es el caso de *_gt.
        c.otu.assign(filas, 0);
        std::fprintf(stderr, "[w2] aviso: no hay %s, se asume una sola especie\n", po.c_str());
    }
    return c;
}

// ── El filtrado completo sobre una estructura ────────────────────────────────

struct Resultado {
    double   build_ms = 0, query_ms = 0;
    int      rondas = 0;
    long long retenidos = 0, aristas = 0;
    uint32_t overflow = 0;
    double   mem_pico_mb = 0;
    std::vector<uint8_t> mascara;
};

template <typename DS, typename Construir>
static Resultado filtrar(const Nube& c, float radio, bool usar_grado,
                         uint64_t semilla, Construir construir)
{
    Resultado R;
    MemProbe mem; mem.begin();
    const uint32_t n = c.n;
    const uint32_t TPB = 256, blk = (n + TPB - 1) / TPB;

    DS ds;
    GpuTimer t;
    t.start();
    construir(ds);
    R.build_ms = t.stop();
    CUDA_CHECK(cudaGetLastError());
    mem.sample();

    t.start();   // desde aqui se mide el filtrado, sin la construccion

    uint32_t *d_cnt = nullptr, *d_ov = nullptr;
    uint64_t *d_off = nullptr, *d_off2 = nullptr;
    int32_t  *d_deg = nullptr;
    int8_t   *d_estado = nullptr;
    uint8_t  *d_raiz = nullptr;
    uint64_t *d_h = nullptr;
    // n+1: el scan exclusivo recorre n+1 entradas para dejar el total en
    // la posicion n. El elemento extra va a cero.
    CUDA_CHECK(cudaMalloc(&d_cnt, ((size_t)n + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_ov,  sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_off, ((size_t)n + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_off2,((size_t)n + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_deg, ((size_t)n + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_estado, (size_t)n * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_raiz,   (size_t)n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_h,      (size_t)n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_ov, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_cnt + n, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_deg + n, 0, sizeof(int32_t)));

    // 1) conteo exacto
    contar_k<DS><<<blk, TPB>>>(ds, c.dev, radio, d_cnt, d_ov, n);
    CUDA_CHECK(cudaGetLastError());

    // 2) scan exclusivo -> desplazamientos
    auto scan = [&](const uint32_t* in, uint64_t* out) {
        void*  tmp = nullptr; size_t bytes = 0;
        cub::DeviceScan::ExclusiveSum(tmp, bytes, in, out, n + 1);
        CUDA_CHECK(cudaMalloc(&tmp, bytes));
        cub::DeviceScan::ExclusiveSum(tmp, bytes, in, out, n + 1);
        CUDA_CHECK(cudaFree(tmp));
    };
    scan(d_cnt, d_off);

    uint64_t total = 0;
    CUDA_CHECK(cudaMemcpy(&total, d_off + n, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    uint32_t* d_csr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_csr, total * sizeof(uint32_t)));
    mem.sample();

    // 3) llenado sin truncamiento (capacidad = conteo exacto)
    llenar_k<DS><<<blk, TPB>>>(ds, c.dev, radio, d_off, d_cnt, d_csr, n);
    CUDA_CHECK(cudaGetLastError());

    // 4) compactar: fuera el propio v y las otras especies
    compactar_k<<<blk, TPB>>>(d_off, d_cnt, c.d_otu, d_csr, d_deg, n);
    CUDA_CHECK(cudaGetLastError());

    // desplazamientos definitivos y reubicacion
    {
        void*  tmp = nullptr; size_t bytes = 0;
        cub::DeviceScan::ExclusiveSum(tmp, bytes, d_deg, d_off2, n + 1);
        CUDA_CHECK(cudaMalloc(&tmp, bytes));
        cub::DeviceScan::ExclusiveSum(tmp, bytes, d_deg, d_off2, n + 1);
        CUDA_CHECK(cudaFree(tmp));
    }
    uint64_t aristas = 0;
    CUDA_CHECK(cudaMemcpy(&aristas, d_off2 + n, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    R.aristas = static_cast<long long>(aristas);

    uint32_t* d_g = nullptr;
    CUDA_CHECK(cudaMalloc(&d_g, std::max<uint64_t>(aristas, 1) * sizeof(uint32_t)));
    reubicar_k<<<blk, TPB>>>(d_off, d_off2, d_deg, d_csr, d_g, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_csr));
    mem.sample();

    // 5) prioridades y rondas
    hash_k<<<blk, TPB>>>(d_h, semilla, n);
    CUDA_CHECK(cudaMemset(d_estado, SIN_DECIDIR, (size_t)n * sizeof(int8_t)));
    marcar_sin_otu_k<<<blk, TPB>>>(c.d_otu, d_estado, n);
    CUDA_CHECK(cudaGetLastError());

    uint32_t* d_left = nullptr;
    CUDA_CHECK(cudaMalloc(&d_left, sizeof(uint32_t)));
    uint32_t left = 1;
    R.rondas = 0;
    while (left > 0) {
        ++R.rondas;
        // d_deg va dos veces a proposito: es la longitud de la fila del CSR
        // y, a la vez, grado(v) = vecinos de la MISMA especie a < d, que es lo
        // que mindeg ordena.
        ronda_a_k<<<blk, TPB>>>(d_off2, d_deg, d_g, d_estado, d_h, d_deg,
                                usar_grado, d_raiz, n);
        ronda_b_k<<<blk, TPB>>>(d_raiz, d_estado, n);
        CUDA_CHECK(cudaMemset(d_left, 0, sizeof(uint32_t)));
        ronda_c_k<<<blk, TPB>>>(d_off2, d_deg, d_g, d_estado, d_left, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&left, d_left, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (R.rondas > 10000) {           // no puede pasar; si pasa, es un bug
            std::fprintf(stderr, "[w2] mas de 10000 rondas: abortando\n");
            break;
        }
    }
    R.query_ms = t.stop();
    mem.sample();

    // salida
    std::vector<int8_t> est(n);
    CUDA_CHECK(cudaMemcpy(est.data(), d_estado, (size_t)n * sizeof(int8_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&R.overflow, d_ov, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    R.mascara.resize(n);
    R.retenidos = 0;
    for (uint32_t i = 0; i < n; i++) {
        R.mascara[i] = (est[i] == DENTRO) ? 1 : 0;
        R.retenidos += R.mascara[i];
    }
    R.mem_pico_mb = mem.pico_mb();

    cudaFree(d_cnt); cudaFree(d_ov); cudaFree(d_off); cudaFree(d_off2);
    cudaFree(d_deg); cudaFree(d_estado); cudaFree(d_raiz); cudaFree(d_h);
    cudaFree(d_g); cudaFree(d_left);
    ds.clear();
    return R;
}

// ── Validacion ───────────────────────────────────────────────────────────────

/** Discrepancias contra la mascara de referencia, o -1 si no esta. */
static long long comparar_mascara(const std::string& ruta,
                                  const std::vector<uint8_t>& got) {
    if (!npy::exists(ruta)) return -1;
    size_t f = 0, c = 0;
    std::vector<uint8_t> ref = npy::load<uint8_t>(ruta, &f, &c);
    if (ref.size() != got.size()) return -1;
    long long dif = 0;
    for (size_t i = 0; i < ref.size(); i++) if ((ref[i] != 0) != (got[i] != 0)) ++dif;
    return dif;
}

/**
 * Lee "rondas_max" del JSON que acompana a la mascara de referencia.
 * Devuelve -1 si el archivo no esta o no trae el campo.
 *
 * POR QUE EL MAXIMO
 *   La referencia filtra especie por especie, asi que tiene un numero de rondas
 *   por especie. La GPU corre todas a la vez sobre un unico grafo: su bucle
 *   termina cuando NO queda ningun punto sin decidir, o sea cuando acaba la
 *   ultima especie. Ese total es el maximo sobre especies, no la suma.
 *
 *   Se parsea a mano, sin dependencia de JSON: el archivo lo escribe
 *   05_references.py con una forma fija y buscar la clave alcanza.
 */
static int leer_rondas_ref(const std::string& ruta) {
    std::FILE* f = std::fopen(ruta.c_str(), "r");
    if (!f) return -1;
    std::string txt;
    char buf[512];
    while (std::fgets(buf, sizeof(buf), f)) txt += buf;
    std::fclose(f);
    const std::string clave = "\"rondas_max\"";
    size_t p = txt.find(clave);
    if (p == std::string::npos) return -1;
    p = txt.find(':', p);
    if (p == std::string::npos) return -1;
    return std::atoi(txt.c_str() + p + 1);
}

/** Comprueba splitmix64 contra los vectores de prueba. */
static bool verificar_splitmix(const std::string& ruta) {
    std::FILE* f = std::fopen(ruta.c_str(), "r");
    if (!f) { std::fprintf(stderr, "[w2] no se pudo abrir %s\n", ruta.c_str()); return false; }
    char linea[256];
    if (!std::fgets(linea, sizeof(linea), f)) { std::fclose(f); return false; }  // cabecera
    long long n = 0, malos = 0;
    while (std::fgets(linea, sizeof(linea), f)) {
        unsigned long long seed = 0, id = 0, esperado = 0;
        if (std::sscanf(linea, "%llu,%llu,%llu", &seed, &id, &esperado) != 3) continue;
        ++n;
        if (record_hash(id, seed) != esperado) {
            ++malos;
            std::fprintf(stderr, "[w2] splitmix difiere: seed=%llu id=%llu\n", seed, id);
        }
    }
    std::fclose(f);
    std::printf("splitmix64: %lld vectores, %lld fallos\n", n, malos);
    return malos == 0 && n > 0;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string label, processed = "../data/processed", out = "w2.csv", lib_dir,
                vectores = "../bench/splitmix_vectors.csv";
    std::vector<double> radios;
    std::vector<std::string> reglas{"random", "mindeg"};
    std::vector<int> semillas{0};

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto lista = [&](auto& v, auto conv) {
            v.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') v.push_back(conv(argv[++i]));
        };
        if      (a == "--label"     && i + 1 < argc) label     = argv[++i];
        else if (a == "--processed" && i + 1 < argc) processed = argv[++i];
        else if (a == "--out"       && i + 1 < argc) out       = argv[++i];
        else if (a == "--lib"       && i + 1 < argc) lib_dir   = argv[++i];
        else if (a == "--vectores"  && i + 1 < argc) vectores  = argv[++i];
        else if (a == "--radii-km") lista(radios,   [](const char* s){ return std::atof(s); });
        else if (a == "--rules")    lista(reglas,   [](const char* s){ return std::string(s); });
        else if (a == "--seeds")    lista(semillas, [](const char* s){ return std::atoi(s); });
        else { std::fprintf(stderr, "argumento desconocido: %s\n", a.c_str()); return 2; }
    }
    if (label.empty() || radios.empty()) {
        std::fprintf(stderr, "uso: bench_w2 --label L --radii-km 10 25 50 "
                             "[--rules random mindeg] [--seeds 0] [--processed DIR] "
                             "[--out CSV] [--lib DIR]\n");
        return 2;
    }

    int n_gpu = 0;
    cudaGetDeviceCount(&n_gpu);
    if (n_gpu == 0) { std::fprintf(stderr, "ERROR: no hay GPU.\n"); return 1; }

    // La primera puerta: si el hash no coincide bit a bit, el orden de
    // prioridades es otro y el conjunto resultante no puede compararse.
    if (!verificar_splitmix(vectores)) {
        std::fprintf(stderr, "ERROR: splitmix64 no reproduce los vectores. Abortando.\n");
        return 1;
    }

    const Entorno env = Entorno::detectar(lib_dir);
    std::printf("GPU %s · CUDA %s · lib %s\n", env.gpu.c_str(), env.cuda.c_str(),
                env.lib_commit.empty() ? "(desconocido)" : env.lib_commit.c_str());

    Nube c = cargar(processed, label);
    c.subir();
    long long con_otu = 0;
    for (auto o : c.otu) con_otu += (o >= 0);
    std::printf("[%s] N = %u  (con especie: %lld)\n", label.c_str(), c.n, con_otu);

    std::FILE* csv = std::fopen(out.c_str(), "w");
    if (!csv) { std::fprintf(stderr, "no se pudo abrir %s\n", out.c_str()); return 1; }
    csv_header(csv);

    for (double km : radios) {
        const float radio = strict_radius_f32(km * 1000.0);
        char tag[64]; std::snprintf(tag, sizeof(tag), "%g", km);
        std::printf("\n[%s] d = %g km\n", label.c_str(), km);

        const GridSize g = grid_cells(c.mn, c.mx, radio, sizeof(GridCell), free_bytes());

        for (const auto& regla : reglas) {
            const bool usar_grado = (regla == "mindeg");
            for (int semilla : semillas) {
                const std::string base_ref =
                    processed + "/" + label + "_thin_" + regla + "_" + tag +
                    "km_s" + std::to_string(semilla) + "_f32c";
                const std::string ref        = base_ref + ".npy";
                const std::string ref_rondas = base_ref + "_rondas.json";
                const int rondas_ref = leer_rondas_ref(ref_rondas);

                Row base;
                base.label = label; base.N = c.n; base.d_km = km;
                base.carga = "W2"; base.regla = regla; base.semilla = semilla;
                base.variante_w2 = "csr";
                base.lib_commit = env.lib_commit; base.gpu = env.gpu; base.cuda = env.cuda;

                auto correr = [&](const char* nombre, auto construir, bool ejecutable) {
                    if (!ejecutable) {
                        Row r = base;
                        r.estructura = nombre; r.valido = "no_ejecutable";
                        r.vecinos_total = static_cast<long long>(g.n_cells);
                        r.mem_pico_mb = g.bytes / (1024.0 * 1024.0);
                        csv_row(csv, r);
                        return;
                    }
                    long long dif = -2;
                    bool rondas_ok = true;
                    for (int rep = -1; rep < REPS_W2; rep++) {
                        Resultado R = construir();
                        if (rep == -1) {
                            dif = comparar_mascara(ref, R.mascara);
                            rondas_ok = (rondas_ref < 0) || (R.rondas == rondas_ref);
                            std::printf("  %-12s %-7s s%d  rondas %2d (ref %s)  "
                                        "retenidos %8lld  aristas %10lld  dif %lld  "
                                        "overflow %u%s\n",
                                        nombre, regla.c_str(), semilla, R.rondas,
                                        rondas_ref < 0 ? "n/d" : std::to_string(rondas_ref).c_str(),
                                        R.retenidos, R.aristas, dif, R.overflow,
                                        rondas_ok ? "" : "   <- RONDAS DISTINTAS");
                            continue;
                        }
                        Row r = base;
                        r.estructura = nombre; r.rep = rep;
                        r.build_ms = R.build_ms; r.query_ms = R.query_ms;
                        r.rondas = R.rondas; r.retenidos = R.retenidos;
                        r.vecinos_total = R.aristas;
                        r.mem_pico_mb = R.mem_pico_mb;
                        r.discrepancias = dif;
                        // Las rondas son una comprobacion independiente del
                        // conjunto: dos implementaciones pueden dar la misma
                        // mascara por caminos distintos, y eso significaria que
                        // una de las dos no esta haciendo el voraz por rondas.
                        r.valido = (R.overflow > 0) ? "FALLA"
                                 : (dif > 0)        ? "FALLA"
                                 : (!rondas_ok)     ? "FALLA_rondas"
                                 : (dif == 0)       ? "ok"
                                 : "sin_referencia";
                        csv_row(csv, r);
                    }
                };

                correr("KDTree", [&] {
                    return filtrar<KDTree<Metrica, 3>>(c, radio, usar_grado, semilla,
                        [&](KDTree<Metrica, 3>& ds) { ds.build_device(c.dev, c.n, Metrica{}); });
                }, true);

                correr("LinearBVH", [&] {
                    return filtrar<LinearBVH<Metrica, 3>>(c, radio, usar_grado, semilla,
                        [&](LinearBVH<Metrica, 3>& ds) { ds.build_device(c.dev, c.n, Metrica{}); });
                }, true);

                const float cell[3] = {radio, radio, radio};
                correr("UniformGrid", [&] {
                    return filtrar<UniformGrid<Metrica, 3>>(c, radio, usar_grado, semilla,
                        [&](UniformGrid<Metrica, 3>& ds) { ds.build_device(c.dev, c.n, cell, Metrica{}); });
                }, g.ejecutable());
            }
        }
    }

    std::fclose(csv);
    c.liberar();
    std::printf("\n-> %s\n", out.c_str());
    return 0;
}
