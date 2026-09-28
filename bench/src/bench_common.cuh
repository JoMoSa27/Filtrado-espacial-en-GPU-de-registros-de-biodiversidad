// bench_common.cuh — Piezas compartidas por W1 y W2.
//
// Reune lo que bench/README.md exige medir o garantizar de forma identica en
// todas las filas del CSV: el radio estricto, el temporizador, la memoria pico
// y el esquema de salida.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do {                                                 \
    cudaError_t _e = (expr);                                                  \
    if (_e != cudaSuccess) {                                                  \
        std::fprintf(stderr, "[bench] CUDA %s:%d — %s\n",                     \
                     __FILE__, __LINE__, cudaGetErrorString(_e));             \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

// Geometria

/// Radio medio WGS84, el mismo que geo.R_EARTH.
static constexpr double R_EARTH = 6371008.8;

/** Radio de cuerda: la distancia euclidiana 3D que equivale al arco d_m. */
inline double chord_radius(double d_m, double radius = R_EARTH) {
    return 2.0 * radius * std::sin(d_m / (2.0 * radius));
}

/**
 * Radio en metros que debe recibir la biblioteca para que la condicion de
 * vecindad sea estrictamente menor que d, como en GeoThinneR y en thin.py.
 *
 * Las consultas de la biblioteca comparan con <=. En float64 la referencia
 * obtiene el estricto multiplicando por (1 - 1e-12), pero en float32 ese factor
 * no altera ningun bit, de modo que el par situado exactamente sobre el radio
 * seguiria contandose. nextafterf devuelve el float inmediatamente inferior,
 * que es la minima correccion que convierte <= en <.
 *
 * Los pares a distancia exactamente d son infrecuentes con coordenadas
 * redondeadas a cinco decimales. Los pares a distancia 0, procedentes de
 * presencias repetidas, son vecinos validos y no se ven afectados.
 *
 * @param d_m  distancia geodesica en metros
 * @return     radio de cuerda en metros, representable en float32
 */
inline float strict_radius_f32(double d_m) {
    return std::nextafterf(static_cast<float>(chord_radius(d_m)), 0.0f);
}

// Temporizador

struct GpuTimer {
    cudaEvent_t a{}, b{};
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); }
    ~GpuTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void  start() { CUDA_CHECK(cudaEventRecord(a, 0)); }
    float stop()  {
        CUDA_CHECK(cudaEventRecord(b, 0));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        return ms;
    }
};

// Memoria

/**
 * Memoria pico, calculada por diferencia contra la memoria libre inicial.
 *
 * cudaMemGetInfo informa del dispositivo completo y no del proceso, de modo que
 * en un nodo compartido la cifra podria incluir otros trabajos. En nukwa-l40s
 * el trabajo dispone de la GPU entera. La limitacion se documenta porque el
 * valor se publica en el paper.
 */
struct MemProbe {
    size_t libre_inicial = 0, total = 0, pico_usado = 0;

    void begin() {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemGetInfo(&libre_inicial, &total));
        pico_usado = 0;
    }
    void sample() {
        size_t libre = 0, tot = 0;
        CUDA_CHECK(cudaMemGetInfo(&libre, &tot));
        if (libre_inicial > libre)
            pico_usado = std::max(pico_usado, libre_inicial - libre);
    }
    double pico_mb() const { return pico_usado / (1024.0 * 1024.0); }
};

/** Memoria libre del dispositivo en bytes, usada para decidir si una fila es
 *  ejecutable antes de intentar construir la grilla. */
inline size_t free_bytes() {
    size_t libre = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&libre, &total));
    return libre;
}

// Grilla: cuantas celdas pide una celda = radio

/** Resultado de grid_cells: dimensionamiento y si la fila puede ejecutarse. */
struct GridSize {
    unsigned long long n_cells = 0;
    unsigned long long bytes   = 0;
    bool cabe_en_u32 = false;
    bool cabe_en_mem = false;
    bool ejecutable() const { return cabe_en_u32 && cabe_en_mem; }
};

/**
 * Numero de celdas de una grilla densa de arista `cell` sobre el bbox dado,
 * calculado en 64 bits antes de construir la estructura.
 *
 * El tamanno de celda queda fijado al radio de consulta en todas las filas y la
 * grilla es densa sobre el bounding box. Con una nube global como el atun el
 * bbox es el cubo terrestre, 12 742 km por eje, lo que a 10 km supone unas
 * 2,07e9 celdas. El contador interno de la biblioteca es uint32_t y a radios
 * menores desbordaria sin aviso, de modo que el producto se evalua aqui en
 * unsigned long long y la fila se declara no ejecutable si no cabe.
 *
 * @param mn, mx           extremos del bbox en metros, tres componentes
 * @param cell             arista de la celda en metros
 * @param bytes_por_celda  sizeof(GridCell)
 * @param libre            memoria libre del dispositivo en bytes
 */
inline GridSize grid_cells(const float mn[3], const float mx[3], float cell,
                           size_t bytes_por_celda, size_t libre) {
    GridSize g;
    g.n_cells = 1ull;
    for (int d = 0; d < 3; d++) {
        const unsigned long long k =
            static_cast<unsigned long long>((mx[d] - mn[d]) / cell) + 1ull;
        g.n_cells *= k;
    }
    g.bytes = g.n_cells * bytes_por_celda;
    g.cabe_en_u32 = g.n_cells <= 0xFFFFFFFFull;
    // Margen del 10 %: ademas de las celdas hacen falta prim_ids, el scratch de
    // CUB y las coordenadas ya residentes en el dispositivo.
    g.cabe_en_mem = g.bytes < static_cast<unsigned long long>(libre * 0.90);
    return g;
}

// CSV

/// Una fila del CSV, con las columnas exactas de bench/README.md.
struct Row {
    std::string label;
    long long   N = 0;
    double      d_km = 0;
    std::string estructura;
    std::string carga;          ///< "W1" | "W2"
    std::string regla   = "";   ///< W2: random | mindeg
    int         semilla = -1;
    int         rep     = -1;   ///< -1 = calentamiento
    double      build_ms = 0, query_ms = 0;
    int         rondas = -1;
    long long   vecinos_total = -1;
    long long   retenidos = -1;
    double      mem_pico_mb = 0;
    std::string valido = "";        ///< ok | FALLA | no_ejecutable
    long long   discrepancias = -1; ///< contra la referencia _f32c
    std::string precision = "f32c"; ///< coordenadas que consumio la estructura
    std::string variante_w2 = "";   ///< "csr" | ""
    std::string lib_commit = "";
    std::string gpu = "";
    std::string cuda = "";
};

inline void csv_header(std::FILE* f) {
    std::fprintf(f,
        "label,N,d_km,estructura,carga,regla,semilla,rep,build_ms,query_ms,"
        "rondas,vecinos_total,retenidos,mem_pico_mb,valido,discrepancias,"
        "precision,variante_w2,lib_commit,gpu,cuda\n");
}

inline void csv_row(std::FILE* f, const Row& r) {
    std::fprintf(f,
        "%s,%lld,%g,%s,%s,%s,%d,%d,%.4f,%.4f,%d,%lld,%lld,%.2f,%s,%lld,%s,%s,%s,%s,%s\n",
        r.label.c_str(), r.N, r.d_km, r.estructura.c_str(), r.carga.c_str(),
        r.regla.c_str(), r.semilla, r.rep, r.build_ms, r.query_ms, r.rondas,
        r.vecinos_total, r.retenidos, r.mem_pico_mb, r.valido.c_str(),
        r.discrepancias, r.precision.c_str(), r.variante_w2.c_str(),
        r.lib_commit.c_str(), r.gpu.c_str(), r.cuda.c_str());
    std::fflush(f);
}

// Entorno, registrado en el CSV para reproducibilidad

/** Salida de un comando sin el salto de linea final; cadena vacia si falla. */
inline std::string shell(const std::string& cmd) {
    std::string out;
    if (FILE* p = popen(cmd.c_str(), "r")) {
        char buf[256];
        while (std::fgets(buf, sizeof(buf), p)) out += buf;
        pclose(p);
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out;
}

struct Entorno {
    std::string gpu, cuda, lib_commit;

    static Entorno detectar(const std::string& lib_dir) {
        Entorno e;
        cudaDeviceProp p{};
        CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
        e.gpu = p.name;
        int rt = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&rt));
        e.cuda = std::to_string(rt / 1000) + "." + std::to_string((rt % 1000) / 10);
        if (!lib_dir.empty())
            e.lib_commit = shell("git -C '" + lib_dir + "' rev-parse HEAD 2>/dev/null");
        return e;
    }
};
