cmake_minimum_required(VERSION 3.24)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
project(gemm_benchmark LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(CUDA_COMMON_FLAGS
    -O3 -DNDEBUG -w
    --expt-relaxed-constexpr --expt-extended-lambda
    --use_fast_math -lineinfo
    -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
)

set(TARGET_GPU "" CACHE STRING "Target GPU: A100 or H100")
set_property(CACHE TARGET_GPU PROPERTY STRINGS "" A100 H100)

function(add_gemm_library target source arch)
    add_library(${target} STATIC ${source})
    set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES "${arch}")
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMMON_FLAGS}>)
endfunction()

set(GEMM_LIBS "")
set(GEMM_ARCHS "")

if(TARGET_GPU STREQUAL "H100")
    add_gemm_library(gemm_h100 csrc/gemm_h100_backends.cu "90a")
    list(APPEND GEMM_LIBS $<LINK_LIBRARY:WHOLE_ARCHIVE,gemm_h100>)
    list(APPEND GEMM_ARCHS "90a")
    add_compile_definitions(HAS_H100)
elseif(TARGET_GPU STREQUAL "A100")
    add_gemm_library(gemm_a100 csrc/gemm_a100_backends.cu "80")
    list(APPEND GEMM_LIBS $<LINK_LIBRARY:WHOLE_ARCHIVE,gemm_a100>)
    list(APPEND GEMM_ARCHS "80")
    add_compile_definitions(HAS_A100)
else()
    message(WARNING "TARGET_GPU not set. Use -DTARGET_GPU=A100 or -DTARGET_GPU=H100. Defaulting to native.")
    set(GEMM_ARCHS "native")
endif()

add_gemm_library(cublas_backend csrc/cublas_backends.cu "${GEMM_ARCHS}")
list(APPEND GEMM_LIBS $<LINK_LIBRARY:WHOLE_ARCHIVE,cublas_backend>)

add_executable(VibeGEMM main.cu)
set_target_properties(VibeGEMM PROPERTIES
    CUDA_ARCHITECTURES "${GEMM_ARCHS}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/out"
)
target_include_directories(VibeGEMM PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include)
target_compile_options(VibeGEMM PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMMON_FLAGS}>)
target_link_libraries(VibeGEMM PRIVATE
    ${GEMM_LIBS}
    CUDA::cublas CUDA::cuda_driver CUDA::cudart
)

install(TARGETS VibeGEMM DESTINATION bin)