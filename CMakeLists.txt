cmake_minimum_required(VERSION 3.16)
project(bsca_lite VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(
    -O3
    -march=native
    -funroll-loops
    -ffast-math
    -Wall
    -Wextra
    -Wno-unused-parameter
)

find_package(OpenMP REQUIRED)

set(CONDA_PREFIX $ENV{CONDA_PREFIX})
if(NOT CONDA_PREFIX)
    message(FATAL_ERROR "CONDA_PREFIX not set. Activate your conda environment.")
endif()

include_directories(
    ${CONDA_PREFIX}/include/rdkit
    ${CONDA_PREFIX}/include
)
link_directories(
    ${CONDA_PREFIX}/lib
)

find_path(CUDA_INCLUDE_DIR cublas_v2.h
    PATHS /usr/local/cuda/include /usr/local/cuda-12.5/include
    NO_DEFAULT_PATH)
if(CUDA_INCLUDE_DIR)
    include_directories(${CUDA_INCLUDE_DIR})
    message(STATUS "CUDA include: ${CUDA_INCLUDE_DIR}")
else()
    message(WARNING "CUDA headers not found; GPU index build may fail")
endif()

set(RDKIT_LIBS
    RDKitGraphMol
    RDKitSmilesParse
    RDKitRDGeneral
    RDKitFingerprints
    RDKitDataStructs
    RDKitMolStandardize
)

set(COMMON_LIBS
    faiss
    ${RDKIT_LIBS}
    OpenMP::OpenMP_CXX
)

# ── Prepare tools ─────────────────────────────────────────────────────────────
add_executable(prepare_npass     src/prepare_npass.cpp)
add_executable(prepare_drugbank  src/prepare_drugbank.cpp)
add_executable(prepare_enamine   src/prepare_enamine.cpp)

target_link_libraries(prepare_npass    ${COMMON_LIBS})
target_link_libraries(prepare_drugbank ${COMMON_LIBS})
target_link_libraries(prepare_enamine  ${COMMON_LIBS})

target_include_directories(prepare_npass    PRIVATE src)
target_include_directories(prepare_drugbank PRIVATE src)
target_include_directories(prepare_enamine  PRIVATE src)

# ── Enamine index build tools (for future use with Enamine) ──────────────────
add_executable(build_enamine_index   src/build_enamine_index.cpp)
add_executable(build_sharded_database src/build_sharded_database.cpp)

target_link_libraries(build_enamine_index    ${COMMON_LIBS})
target_link_libraries(build_sharded_database ${COMMON_LIBS})

target_include_directories(build_enamine_index    PRIVATE src)
target_include_directories(build_sharded_database PRIVATE src)

# ── search_lite: main search tool (NPASS + DrugBank, optionally Enamine) ─────
add_executable(search_lite src/search_lite.cpp)
target_link_libraries(search_lite ${COMMON_LIBS})
target_include_directories(search_lite PRIVATE src)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
