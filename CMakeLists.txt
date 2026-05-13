cmake_minimum_required(VERSION 3.18)
project(spmm_all LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CXX_COMPILER AND DEFINED ENV{KOKKOS_PREFIX}
   AND EXISTS "$ENV{KOKKOS_PREFIX}/bin/nvcc_wrapper")
  set(CMAKE_CXX_COMPILER "$ENV{KOKKOS_PREFIX}/bin/nvcc_wrapper" CACHE FILEPATH "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

find_package(Ginkgo      REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP      REQUIRED)
find_library(NVTOOLSEXT nvToolsExt
  PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu
  REQUIRED)

set(NCU_WARM_UP    0   CACHE STRING "Warm-up iterations for NCU profiling")
set(NCU_ITERATIONS 0   CACHE STRING "Measurement iterations for NCU profiling")
set(TIME_WARM_UP   100   CACHE STRING "Warm-up iterations for timing")
set(TIME_ITERATIONS 10 CACHE STRING "Measurement iterations for timing")

# /////////////////////////////////////////////////

add_executable(Time_spmm_cuSPARSE
  src/Time_cuSPARSE_main.cu
  src/external/BSA-SpMM/src/matrices.cpp
)

add_executable(Time_spmm_Ginkgo
  src/Time_Ginkgo_main.cu
  src/external/BSA-SpMM/src/matrices.cpp
)

add_executable(NCU_spmm_cuSPARSE
  src/NCU_cuSPARSE_main.cu
  src/external/BSA-SpMM/src/matrices.cpp
)

add_executable(NCU_spmm_Ginkgo
  src/NCU_Ginkgo_main.cu
  src/external/BSA-SpMM/src/matrices.cpp
)

# ///////////////////////////////////////////////

foreach(TARGET NCU_spmm_cuSPARSE NCU_spmm_Ginkgo Time_spmm_cuSPARSE Time_spmm_Ginkgo)
  target_link_libraries(${TARGET} PRIVATE
    Ginkgo::ginkgo
    CUDA::cusparse
    CUDA::cudart
    OpenMP::OpenMP_CXX
    ${NVTOOLSEXT}
  )
  target_include_directories(${TARGET} PRIVATE
    "$ENV{GKO_PREFIX}/include"
    "include"
    "src/external/BSA-SpMM/include"
  )
  set_target_properties(${TARGET} PROPERTIES
    BUILD_RPATH   "$ENV{GKO_PREFIX}/lib"
    INSTALL_RPATH "$ENV{GKO_PREFIX}/lib"
  )
  target_compile_options(${TARGET} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG -march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -DNDEBUG -Xcompiler=-fopenmp --expt-relaxed-constexpr --expt-extended-lambda>
  )
endforeach()

# NCU 타겟용
foreach(TARGET NCU_spmm_cuSPARSE NCU_spmm_Ginkgo)
  target_compile_definitions(${TARGET} PRIVATE
    WARM_UP=${NCU_WARM_UP}
    ITERATIONS=${NCU_ITERATIONS}
  )
endforeach()

# Time 타겟용
foreach(TARGET Time_spmm_cuSPARSE Time_spmm_Ginkgo)
  target_compile_definitions(${TARGET} PRIVATE
    WARM_UP=${TIME_WARM_UP}
    ITERATIONS=${TIME_ITERATIONS}
  )
endforeach()
