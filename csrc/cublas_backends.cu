#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <string>

#include "backend.cuh"
#include "registry.cuh"
#include "utils.cuh"

class CublasBackend : public GemmBackend {
  public:
    CublasBackend() {
        CUBLAS_CHECK(cublasCreate(&handle_));
        // CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));
    }

    ~CublasBackend() override { cublasDestroy(handle_); }

    void run(half *A, half *B, half *C, int M, int N, int K) override {
        const float alpha = 1.f, beta = 0.f;

#if defined(HAS_H100)
        CUBLAS_CHECK(cublasGemmEx(handle_, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A, CUDA_R_16F,
                                  K, B, CUDA_R_16F, K, &beta, C, CUDA_R_16F, M, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
#elif defined(HAS_A100)
        CUBLAS_CHECK(cublasGemmEx(handle_, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F,
                                  K, A, CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
#else
#error "Neither HAS_H100 nor HAS_A100 is defined"
#endif
    }

    std::string name() const override { return "cuBLAS"; }
    void teardown() override {}

  private:
    cublasHandle_t handle_{};
};

REGISTER_CUBLAS("cuBLAS", CublasBackend);