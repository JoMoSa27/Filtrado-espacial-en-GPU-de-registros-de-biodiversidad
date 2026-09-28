// npy.hpp — Lector minimo de arreglos .npy de NumPy (solo C-order).
//
// POR QUE UN LECTOR PROPIO
//   El pipeline entrega los datos como .npy y la validacion exige comparar
//   contra las referencias de 05_references.py archivo por archivo. Meter una
//   dependencia (cnpy, xtensor) por leer una cabecera de 128 bytes no se
//   justifica, y en Kabre cada dependencia es un modulo mas que cargar.
//
// QUE SOPORTA
//   dtypes: <f8 (float64), <f4 (float32), <i4 (int32), |b1 (bool)
//   forma:  1-D (N,) y 2-D (N,3); siempre C-order.
//   Si el archivo viene en fortran_order se aborta: transponer en silencio
//   produciria coordenadas mezcladas que pasan todas las comprobaciones de
//   tamanno y fallan solo en los conteos, que es el peor sitio donde enterarse.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace npy {

struct Header {
    std::string dtype;     ///< "<f8", "<f4", "<i4", "|b1"
    size_t      rows = 0;  ///< primera dimension
    size_t      cols = 1;  ///< segunda dimension (1 si el arreglo es 1-D)
    size_t      data_offset = 0;
};

inline size_t itemsize(const std::string& d) {
    if (d == "<f8") return 8;
    if (d == "<f4" || d == "<i4") return 4;
    if (d == "|b1") return 1;
    throw std::runtime_error("npy: dtype no soportado: " + d);
}

/** Lee la cabecera y deja el stream posicionado al inicio de los datos. */
inline Header read_header(std::ifstream& f, const std::string& path) {
    char magic[6] = {};
    f.read(magic, 6);
    if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("npy: no es un .npy: " + path);

    uint8_t major = 0, minor = 0;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    size_t hlen = 0;
    if (major == 1) {
        uint16_t h16 = 0;
        f.read(reinterpret_cast<char*>(&h16), 2);
        hlen = h16;
    } else {                       // v2/v3: longitud de cabecera en 4 bytes
        uint32_t h32 = 0;
        f.read(reinterpret_cast<char*>(&h32), 4);
        hlen = h32;
    }

    std::string dict(hlen, '\0');
    f.read(&dict[0], static_cast<std::streamsize>(hlen));

    Header h;
    h.data_offset = static_cast<size_t>(f.tellg());

    // El diccionario es un literal de Python; se extraen tres campos por
    // busqueda directa en vez de parsearlo, que para este formato fijo alcanza.
    auto field = [&](const char* key) -> size_t {
        size_t p = dict.find(key);
        if (p == std::string::npos)
            throw std::runtime_error("npy: falta '" + std::string(key) + "' en " + path);
        return p;
    };

    size_t p = field("'descr':");
    p = dict.find('\'', p + 8);
    size_t q = dict.find('\'', p + 1);
    h.dtype = dict.substr(p + 1, q - p - 1);

    p = field("'fortran_order':");
    if (dict.compare(p + 16, 5, " True") == 0)
        throw std::runtime_error("npy: fortran_order=True no soportado: " + path);

    p = field("'shape':");
    p = dict.find('(', p);
    q = dict.find(')', p);
    const std::string shape = dict.substr(p + 1, q - p - 1);

    // "(N,)" -> 1-D ; "(N, 3)" -> 2-D
    size_t coma = shape.find(',');
    h.rows = std::strtoull(shape.c_str(), nullptr, 10);
    h.cols = 1;
    if (coma != std::string::npos) {
        const std::string rest = shape.substr(coma + 1);
        size_t nz = rest.find_first_not_of(" \t");
        if (nz != std::string::npos && std::isdigit(static_cast<unsigned char>(rest[nz])))
            h.cols = std::strtoull(rest.c_str() + nz, nullptr, 10);
    }
    return h;
}

/**
 * Carga un .npy a un vector<T>, convirtiendo desde el dtype del archivo.
 *
 * La conversion es explicita y no silenciosa: se pide T y se acepta cualquier
 * dtype de la lista, porque las coordenadas llegan en float64 y la biblioteca
 * las consume en float32. Ese estrechamiento es una decision del diseno
 * (bench/README.md), no un accidente, asi que se hace aqui y en un solo sitio.
 */
template <typename T>
std::vector<T> load(const std::string& path, size_t* rows = nullptr,
                    size_t* cols = nullptr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy: no se pudo abrir " + path);

    const Header h = read_header(f, path);
    const size_t n = h.rows * h.cols;
    if (rows) *rows = h.rows;
    if (cols) *cols = h.cols;

    std::vector<char> raw(n * itemsize(h.dtype));
    f.read(raw.data(), static_cast<std::streamsize>(raw.size()));
    if (static_cast<size_t>(f.gcount()) != raw.size())
        throw std::runtime_error("npy: archivo truncado: " + path);

    std::vector<T> out(n);
    if (h.dtype == "<f8") {
        const auto* p = reinterpret_cast<const double*>(raw.data());
        for (size_t i = 0; i < n; i++) out[i] = static_cast<T>(p[i]);
    } else if (h.dtype == "<f4") {
        const auto* p = reinterpret_cast<const float*>(raw.data());
        for (size_t i = 0; i < n; i++) out[i] = static_cast<T>(p[i]);
    } else if (h.dtype == "<i4") {
        const auto* p = reinterpret_cast<const int32_t*>(raw.data());
        for (size_t i = 0; i < n; i++) out[i] = static_cast<T>(p[i]);
    } else if (h.dtype == "|b1") {
        const auto* p = reinterpret_cast<const uint8_t*>(raw.data());
        for (size_t i = 0; i < n; i++) out[i] = static_cast<T>(p[i] != 0);
    }
    return out;
}

/** true si el archivo existe y se puede abrir. */
inline bool exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return static_cast<bool>(f);
}

}  // namespace npy
