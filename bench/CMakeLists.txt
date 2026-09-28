# bench/ — Benchmark en GPU del poster BIP 2026
#
#   cmake -S bench -B build -DDATA_STRUCTURES_DIR=/ruta/a/Data_Structures
#   cmake --build build -j
#
# DATA_STRUCTURES_DIR apunta al repo de las estructuras (el directorio que
# contiene structures/). Se puede dar tambien por variable de entorno.
#
# La biblioteca de estructuras se consume por sus fuentes, no con
# add_subdirectory, porque su CMakeLists declara unicamente ejecutables
# (kdtree, ugrid, bvh, heap_probe, bench_compare, bench_verify) y ningun target
# de biblioteca: add_subdirectory compilaria esos binarios sin aportar nada
# enlazable. Las estructuras son plantillas en cabeceras mas tres .cu de
# construccion con instanciacion explicita para Dim 2 y 3, de modo que basta
# con compilarlos junto al benchmark. La biblioteca no se modifica.
#
# Si en el futuro expone add_library(spatial_structures ...), sustituir
# SPATIAL_SOURCES por add_subdirectory y target_link_libraries.

cmake_minimum_required(VERSION 3.18)
project(bip2026_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# DATA_STRUCTURES_DIR es la raiz del repo de estructuras: la carpeta que
# contiene docs/ y structures/. Las cabeceras y los .cu viven bajo structures/.
if(NOT DEFINED DATA_STRUCTURES_DIR)
  if(DEFINED ENV{DATA_STRUCTURES_DIR})
    set(DATA_STRUCTURES_DIR $ENV{DATA_STRUCTURES_DIR})
  else()
    set(DATA_STRUCTURES_DIR "$ENV{HOME}/asistencia_cenat_josue_morera/projects/Data_Structures")
  endif()
endif()

set(SPATIAL "${DATA_STRUCTURES_DIR}/structures")
if(NOT EXISTS "${SPATIAL}/common/include/SpatialTags.cuh")
  message(FATAL_ERROR
    "DATA_STRUCTURES_DIR no apunta al repo de las estructuras.\n"
    "  Debe ser la carpeta que contiene docs/ y structures/, no structures/ misma.\n"
    "  Se buscaba: ${SPATIAL}/common/include/SpatialTags.cuh\n"
    "  Recibido:   ${DATA_STRUCTURES_DIR}")
endif()
message(STATUS "Estructuras (solo lectura): ${SPATIAL}")

find_package(CUDAToolkit REQUIRED)

# La L40S de la particion nukwa-l40s es sm_89. El valor se fija antes de crear
# los targets porque CMake congela CUDA_ARCHITECTURES en cada uno al declararlo.
# Se puede sobrescribir con -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES sin definir; usando 89 (L40S)")
endif()

# Los tres constructores de la biblioteca. El resto son plantillas en
# cabeceras. Cada uno instancia explicitamente Dim = 2 y 3.
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

# W1 — densidad del grupo objetivo
add_executable(bench_w1 src/w1.cu ${SPATIAL_SOURCES})
target_include_directories(bench_w1 PRIVATE src ${SPATIAL_INCLUDES})
set_target_properties(bench_w1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(bench_w1 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O2 --expt-extended-lambda>)


# W2 — filtrado por especie, voraz paralelo por prioridad
# Usa cub::DeviceScan para los desplazamientos del CSR. CUB viene con el toolkit.
add_executable(bench_w2 src/w2.cu ${SPATIAL_SOURCES})
target_include_directories(bench_w2 PRIVATE src ${SPATIAL_INCLUDES})
set_target_properties(bench_w2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(bench_w2 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O2 --expt-extended-lambda>)
