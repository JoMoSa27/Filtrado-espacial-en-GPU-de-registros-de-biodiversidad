/**
 * @file w1.cu
 * @brief W1 — densidad de registros del grupo objetivo: para cada registro,
 *        cuantos OTROS estan a menos de d (geodesica, via radio de cuerda).
 *
 * Corresponde a la correccion de esfuerzo de muestreo por densidad del grupo
 * objetivo (Phillips et al. 2009) y constituye la parte dominante del coste del
 * filtrado. Se mide sobre las tres estructuras de la biblioteca del trabajo
 * previo, que se consume sin modificarla.
 *
 * Referencias de validacion
 *   La biblioteca opera en float32, dado que CoordPtrs<Dim> es const float*, de
 *   modo que consume <label>_xyz_f32_centered.npy y se valida contra la
 *   referencia calculada sobre esas mismas coordenadas:
 *
 *     <label>_count_<d>km_f32c.npy   igualdad exacta, columna 'discrepancias'
 *     <label>_count_<d>km.npy        diferencia informada en el log
 *
 *   Exigir igualdad contra la referencia float64 equivaldria a pedir a la
 *   estructura que compense una cuantizacion anterior a su entrada. Segun
 *   bench/README.md, el atun difiere en 36, 40 y 59 conteos de 80 163 a 10, 25
 *   y 50 km respectivamente.
 *
 * Condicion de vecindad
 *   El radio que recibe la biblioteca es nextafterf(cuerda, 0), porque sus
 *   consultas comparan con <= (vease bench_common.cuh). count_in_radius incluye
 *   el propio punto, situado a distancia 0, de modo que se resta una unidad.
 *   Las presencias repetidas, tambien a distancia 0, son vecinos validos y se
 *   conservan; en las nubes _dup constituyen la mayoria.
 *
 * Grilla
 *   La arista de celda es igual al radio de consulta en todas las filas. El
 *   numero de celdas se evalua en 64 bits antes de construir; si excede
 *   uint32_t o la memoria disponible, la fila se registra como 'no_ejecutable'
 *   junto con ese numero, en lugar de omitirse.
 *
 * USO
 *   bench_w1 --label tortuga_gt --radii-km 10 25 50 \
 *            --processed ../data/processed --out ../bench/results/x.csv
 */

#include <algorithm>
#include <string>
#include <vector>

#include "bench_common.cuh"
#include "npy.hpp"

#include "SpatialTags.cuh"
#include "Metric.cuh"
#include "KDTree2D.cuh"
#include "UniformGrid2D.cuh"
#include "LinearBVH.cuh"

using Metrica = Cartesian<3>;

// Kernel de conteo
/**
 * Un hilo por registro. Cada registro actua a la vez como punto y como
 * consulta, de modo que se descuenta a si mismo.
 *
 * `overflow` se propaga porque, si una estructura abandona parte del recorrido
 * (pila llena en el kd-tree o el BVH, caja de celdas no representable en la
 * grilla), el conteo resultante es una cota inferior y la fila no puede
 * declararse valida.
 *
 * @param ds          estructura ya construida, pasada por valor
 * @param pts         coordenadas en device, SoA de tres componentes
 * @param radio       radio de cuerda en metros, ya estricto
 * @param out         [out] conteo por registro, N elementos int32
 * @param n_overflow  [out] numero de consultas con recorrido incompleto
 * @param n           numero de registros
 */
template <typename DS>
__global__ void w1_k(DS ds, CoordPtrs<3> pts, float radio,
                     int32_t* out, uint32_t* n_overflow, uint32_t n)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    float q[3] = {pts.c[0][i], pts.c[1][i], pts.c[2][i]};
    bool ov = false;
    const uint32_t c = ds.count_in_radius(q, radio, &ov);
    out[i] = static_cast<int32_t>(c) - 1;          // fuera el propio punto
    if (ov) atomicAdd(n_overflow, 1u);
}

// Utilidades

struct Nube {
    uint32_t           n = 0;
    std::vector<float> x, y, z;                 // SoA en host
    float              mn[3]{}, mx[3]{};
    CoordPtrs<3>       dev{};                   // punteros de device
    float*             dev_raw[3]{};

    void subir() {
        for (int d = 0; d < 3; d++)
            CUDA_CHECK(cudaMalloc(&dev_raw[d], n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dev_raw[0], x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_raw[1], y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_raw[2], z.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        for (int d = 0; d < 3; d++) dev.c[d] = dev_raw[d];
    }
    void liberar() { for (int d = 0; d < 3; d++) if (dev_raw[d]) cudaFree(dev_raw[d]); }
};

/**
 * Carga <label>_xyz_f32_centered.npy, de forma (N,3) float32, y lo reorganiza
 * en SoA, que es el formato que consumen las estructuras: CoordPtrs<Dim>
 * contiene un puntero por coordenada. Calcula ademas el bbox, necesario para
 * dimensionar la grilla.
 *
 * @throws std::runtime_error  si el arreglo no tiene tres columnas
 */
static Nube cargar(const std::string& processed, const std::string& label) {
    const std::string p = processed + "/" + label + "_xyz_f32_centered.npy";
    size_t filas = 0, cols = 0;
    std::vector<float> v = npy::load<float>(p, &filas, &cols);
    if (cols != 3)
        throw std::runtime_error("w1: se esperaba (N,3) en " + p);

    Nube c;
    c.n = static_cast<uint32_t>(filas);
    c.x.resize(filas); c.y.resize(filas); c.z.resize(filas);
    for (int d = 0; d < 3; d++) { c.mn[d] = 1e30f; c.mx[d] = -1e30f; }
    for (size_t i = 0; i < filas; i++) {
        const float a = v[3 * i], b = v[3 * i + 1], g = v[3 * i + 2];
        c.x[i] = a; c.y[i] = b; c.z[i] = g;
        const float t[3] = {a, b, g};
        for (int d = 0; d < 3; d++) {
            c.mn[d] = std::min(c.mn[d], t[d]);
            c.mx[d] = std::max(c.mx[d], t[d]);
        }
    }
    return c;
}

/**
 * Discrepancias contra una referencia .npy, o -1 si la referencia no esta.
 *
 * Si `volcado` no esta vacio y existen discrepancias, escribe un CSV con una
 * fila por registro que difiere: indice, conteo de la GPU y conteo de la
 * referencia.
 *
 * Los indices se vuelcan porque el ruido residual de float32 no es nulo: la GPU
 * acumula distance_sq en float32 y la referencia en float64 sobre las mismas
 * coordenadas, de modo que un par situado a menos de un centimetro del radio
 * puede quedar a un lado o a otro. Un conteo agregado no permite distinguir ese
 * caso de un error de implementacion. Con los indices, scripts/08_check_w1.py
 * recalcula en float64 la distancia de cada par dudoso y resuelve la
 * ambiguedad.
 *
 * @return  numero de registros con conteo distinto, o -1 si falta la referencia
 */
static long long comparar(const std::string& ruta, const std::vector<int32_t>& got,
                          const std::string& volcado = "") {
    if (!npy::exists(ruta)) return -1;
    size_t filas = 0, cols = 0;
    std::vector<int32_t> ref = npy::load<int32_t>(ruta, &filas, &cols);
    if (ref.size() != got.size()) {
        std::fprintf(stderr, "[w1] tamanno distinto en %s: %zu vs %zu\n",
                     ruta.c_str(), ref.size(), got.size());
        return -1;
    }
    long long dif = 0;
    for (size_t i = 0; i < ref.size(); i++) if (ref[i] != got[i]) ++dif;

    if (dif > 0 && !volcado.empty()) {
        if (std::FILE* f = std::fopen(volcado.c_str(), "w")) {
            std::fprintf(f, "idx,gpu,ref\n");
            for (size_t i = 0; i < ref.size(); i++)
                if (ref[i] != got[i])
                    std::fprintf(f, "%zu,%d,%d\n", i, got[i], ref[i]);
            std::fclose(f);
            std::fprintf(stderr, "[w1] %lld discrepancias -> %s\n", dif, volcado.c_str());
        }
    }
    return dif;
}

// Medicion de una estructura a un radio

static constexpr int REPS = 10;   ///< 1 calentamiento + 10 medidas (README)

/**
 * Construye la estructura, ejecuta la consulta REPS+1 veces y escribe una fila
 * del CSV por repeticion medida. La primera pasada, con rep = -1, es de
 * calentamiento y se emplea para validar contra las referencias.
 *
 * @param construir  invocable que recibe la estructura por referencia y la
 *                   deja construida
 */
template <typename DS, typename Construir>
static void medir(const char* nombre, const Nube& c, float radio, double d_km,
                  const Row& plantilla, std::FILE* csv,
                  const std::string& ref_f32c, const std::string& ref_f64,
                  const std::string& volcado_base, Construir construir)
{
    MemProbe mem;
    mem.begin();

    int32_t*  d_out = nullptr;
    uint32_t* d_ov  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, c.n * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_ov,  sizeof(uint32_t)));

    DS ds;
    GpuTimer t;
    t.start();
    construir(ds);
    const double build_ms = t.stop();
    CUDA_CHECK(cudaGetLastError());
    mem.sample();

    const uint32_t TPB = 256;
    const uint32_t blk = (c.n + TPB - 1) / TPB;

    std::vector<int32_t> h_out(c.n);
    long long dif_f32c = -1, dif_f64 = -1, vecinos = -1;
    uint32_t  overflow = 0;

    for (int rep = -1; rep < REPS; rep++) {      // rep = -1 -> calentamiento
        CUDA_CHECK(cudaMemset(d_ov, 0, sizeof(uint32_t)));
        t.start();
        w1_k<DS><<<blk, TPB>>>(ds, c.dev, radio, d_out, d_ov, c.n);
        const double query_ms = t.stop();
        CUDA_CHECK(cudaGetLastError());
        mem.sample();

        if (rep == -1) {                          // validar una sola vez
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, c.n * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&overflow, d_ov, sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
            vecinos = 0;
            for (uint32_t i = 0; i < c.n; i++) vecinos += h_out[i];
            dif_f32c = comparar(ref_f32c, h_out,
                                volcado_base.empty() ? std::string()
                                                     : volcado_base + "_" + nombre + ".csv");
            dif_f64  = comparar(ref_f64,  h_out);
            std::printf("  %-12s build %8.2f ms  vecinos %12lld  "
                        "dif_f32c %6lld  dif_f64 %6lld  overflow %u\n",
                        nombre, build_ms, vecinos, dif_f32c, dif_f64, overflow);
            continue;
        }

        Row r = plantilla;
        r.estructura    = nombre;
        r.rep           = rep;
        r.build_ms      = build_ms;
        r.query_ms      = query_ms;
        r.vecinos_total = vecinos;
        r.mem_pico_mb   = mem.pico_mb();
        r.discrepancias = dif_f32c;
        r.valido = (overflow > 0)       ? "FALLA"
                 : (dif_f32c == 0)      ? "ok"
                 : (dif_f32c  > 0)      ? "FALLA"
                 : "sin_referencia";
        csv_row(csv, r);
    }

    ds.clear();
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_ov));
}

// main

int main(int argc, char** argv) {
    std::string label, processed = "../data/processed", out = "w1.csv", lib_dir;
    std::vector<double> radios;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--label"     && i + 1 < argc) label     = argv[++i];
        else if (a == "--processed" && i + 1 < argc) processed = argv[++i];
        else if (a == "--out"  && i + 1 < argc) out       = argv[++i];
        else if (a == "--lib"  && i + 1 < argc) lib_dir   = argv[++i];
        else if (a == "--radii-km") { while (i + 1 < argc && argv[i + 1][0] != '-') radios.push_back(std::atof(argv[++i])); }
        else { std::fprintf(stderr, "argumento desconocido: %s\n", a.c_str()); return 2; }
    }
    if (label.empty() || radios.empty()) {
        std::fprintf(stderr, "uso: bench_w1 --label L --radii-km 10 25 50 "
                             "[--processed DIR] [--out CSV] [--lib DIR]\n");
        return 2;
    }

    int n_gpu = 0;
    cudaGetDeviceCount(&n_gpu);
    if (n_gpu == 0) { std::fprintf(stderr, "ERROR: no hay GPU.\n"); return 1; }

    const Entorno env = Entorno::detectar(lib_dir);
    std::printf("GPU %s · CUDA %s · lib %s\n", env.gpu.c_str(), env.cuda.c_str(),
                env.lib_commit.empty() ? "(desconocido)" : env.lib_commit.c_str());

    Nube c = cargar(processed, label);
    c.subir();
    std::printf("[%s] N = %u\n", label.c_str(), c.n);

    std::FILE* csv = std::fopen(out.c_str(), "w");
    if (!csv) { std::fprintf(stderr, "no se pudo abrir %s\n", out.c_str()); return 1; }
    csv_header(csv);

    for (double km : radios) {
        const float radio = strict_radius_f32(km * 1000.0);
        std::printf("\n[%s] d = %g km  (cuerda estricta %.6f m)\n",
                    label.c_str(), km, static_cast<double>(radio));

        char tag[64];
        std::snprintf(tag, sizeof(tag), "%g", km);
        const std::string ref_f32c = processed + "/" + label + "_count_" + tag + "km_f32c.npy";
        const std::string ref_f64  = processed + "/" + label + "_count_" + tag + "km.npy";
        // Base del nombre del volcado; medir() anade el nombre de la estructura.
        const std::string volcado = out.substr(0, out.find_last_of('.')) +
                                    "_dif_" + label + "_" + tag + "km";

        Row plantilla;
        plantilla.label = label;
        plantilla.N     = c.n;
        plantilla.d_km  = km;
        plantilla.carga = "W1";
        plantilla.lib_commit = env.lib_commit;
        plantilla.gpu = env.gpu;
        plantilla.cuda = env.cuda;

        // k-d tree
        medir<KDTree<Metrica, 3>>("KDTree", c, radio, km, plantilla, csv,
                                  ref_f32c, ref_f64, volcado,
                                  [&](KDTree<Metrica, 3>& ds) {
                                      ds.build_device(c.dev, c.n, Metrica{});
                                  });

        // LBVH
        medir<LinearBVH<Metrica, 3>>("LinearBVH", c, radio, km, plantilla, csv,
                                     ref_f32c, ref_f64, volcado,
                                     [&](LinearBVH<Metrica, 3>& ds) {
                                         ds.build_device(c.dev, c.n, Metrica{});
                                     });

        // Grilla, celda = radio
        // La comprobacion precede a la construccion: la grilla es densa sobre
        // el bbox y con una nube global el numero de celdas puede desbordar
        // uint32_t o exceder la memoria. La fila no ejecutable se registra con
        // ese numero de celdas, que es en si mismo un resultado: una grilla
        // densa sobre datos globales reserva memoria para el interior de la
        // Tierra.
        const GridSize g = grid_cells(c.mn, c.mx, radio, sizeof(GridCell), free_bytes());
        std::printf("  %-12s celdas %llu (%.2f GB)%s\n", "UniformGrid",
                    g.n_cells, g.bytes / (1024.0 * 1024.0 * 1024.0),
                    g.ejecutable() ? "" : "  -> NO EJECUTABLE");

        if (!g.ejecutable()) {
            Row r = plantilla;
            r.estructura    = "UniformGrid";
            r.valido        = "no_ejecutable";
            r.vecinos_total = static_cast<long long>(g.n_cells);
            r.mem_pico_mb   = g.bytes / (1024.0 * 1024.0);
            csv_row(csv, r);
        } else {
            const float cell[3] = {radio, radio, radio};
            medir<UniformGrid<Metrica, 3>>("UniformGrid", c, radio, km, plantilla, csv,
                                           ref_f32c, ref_f64, volcado,
                                           [&](UniformGrid<Metrica, 3>& ds) {
                                               ds.build_device(c.dev, c.n, cell, Metrica{});
                                           });
        }
    }

    std::fclose(csv);
    c.liberar();
    std::printf("\n-> %s\n", out.c_str());
    return 0;
}
