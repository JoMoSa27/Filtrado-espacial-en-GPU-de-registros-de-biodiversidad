# bench/ — Benchmark en GPU del poster BIP 2026
#
#   cmake -S bench -B build -DDATA_STRUCTURES_DIR=/ruta/a/Data_Structures
#   cmake --build build -j
#
# DATA_STRUCTURES_DIR apunta al repo de las estructuras (el directorio que
# contiene structures/). Se puede dar tambien por variable de entorno.
#
# POR QUE NO SE USA add_subdirectory
#   El prompt pedia enlazar con add_subdirectory. No se puede: la biblioteca no
#   expone ningun target de biblioteca. Su CMakeLists solo declara ejecutables
#   (kdtree, ugrid, bvh, heap_probe, bench_compare, bench_verify), asi que un
#   add_subdirectory arrastraria esos binarios a nuestra compilacion sin darnos
#   nada enlazable a cambio.
#
#   En su lugar se consumen sus fuentes directamente, que es exactamente lo que
#   hacen sus propios targets: las estructuras son plantillas en cabeceras mas
#   tres .cu de construccion con instanciacion explicita para Dim 2 y 3. No se
#   modifica ni un archivo de la biblioteca; solo se leen.
#
#   Si algun dia la biblioteca expone `add_library(spatial_structures ...)`,
#   esto se reemplaza por add_subdirectory + target_link_libraries y se borra la
#   lista SPATIAL_SOURCES de abajo.

cmake_minimum_required(VERSION 3.18)
project(bip2026_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED DATA_STRUCTURES_DIR)
  if(DEFINED ENV{DATA_STRUCTURES_DIR})
    set(DATA_STRUCTURES_DIR $ENV{DATA_STRUCTURES_DIR})
  else()
    message(FATAL_ERROR
      "Falta DATA_STRUCTURES_DIR.\n"
      "  cmake -S bench -B build -DDATA_STRUCTURES_DIR=/ruta/a/Data_Structures")
  endif()
endif()

set(SPATIAL "${DATA_STRUCTURES_DIR}/structures")
if(NOT EXISTS "${SPATIAL}/common/include/SpatialTags.cuh")
  message(FATAL_ERROR
    "No parece la biblioteca de estructuras: no existe\n"
    "  ${SPATIAL}/common/include/SpatialTags.cuh")
endif()
message(STATUS "Estructuras (solo lectura): ${SPATIAL}")

find_package(CUDAToolkit REQUIRED)

# Fuentes de la biblioteca. Solo los tres constructores: todo lo demas son
# plantillas en cabeceras. Cada uno instancia explicitamente Dim = 2 y 3.
set(SPATIAL_SOURCES
  "${SPATIAL}/KDTree/src/KDBuilder.cu"
  "${SPATIAL}/UniformGrid/src/GridBuilder.cu"
  "${SPATIAL}/LinearBVH/src/BVHBuilder.cu"
)

set(SPATIAL_INCLUDES
  "${SPATIAL}/common/include"
  "${SPATIAL}/KDTree/include"
  "${SPATIAL}/UniformGrid/include"
  "${SPATIAL}/LinearBVH/include"
)

# ── W1 — densidad del grupo objetivo ─────────────────────────────────────────
add_executable(bench_w1 src/w1.cu ${SPATIAL_SOURCES})
target_include_directories(bench_w1 PRIVATE src ${SPATIAL_INCLUDES})
set_target_properties(bench_w1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(bench_w1 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O2 --expt-extended-lambda>)

# La arquitectura se fija desde fuera para no compilar PTX de mas:
#   -DCMAKE_CUDA_ARCHITECTURES=89   (L40S)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(WARNING "CMAKE_CUDA_ARCHITECTURES sin definir; en nukwa-l40s usar 89")
endif()

# ── W2 — filtrado por especie, voraz paralelo por prioridad ──────────────────
# Usa cub::DeviceScan para los desplazamientos del CSR; CUB viene con el toolkit.
add_executable(bench_w2 src/w2.cu ${SPATIAL_SOURCES})
target_include_directories(bench_w2 PRIVATE src ${SPATIAL_INCLUDES})
set_target_properties(bench_w2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(bench_w2 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O2 --expt-extended-lambda>)
