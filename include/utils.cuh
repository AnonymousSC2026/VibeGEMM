#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <random>
#include <string>
#include <vector>

// ── Non-fatal CUDA helpers (benchmark: log and continue) ─────────────────────
inline bool cuda_report_error(cudaError_t err, const char *file, int line) {
    if (err == cudaSuccess)
        return true;
    fprintf(stderr, "CUDA error %s:%d  %s\n", file, line, cudaGetErrorString(err));
    return false;
}

#define CUDA_TRY(call) cuda_report_error((call), __FILE__, __LINE__)

// Synchronize after kernel/async work; clears sticky error state on failure.
inline bool cuda_check(const char *context) {
    cudaError_t e = cudaDeviceSynchronize();
    if (e == cudaSuccess)
        return true;
    fprintf(stderr, "[%s] %s\n", context, cudaGetErrorString(e));
    (void)cudaGetLastError();
    return false;
}

// ── Error-checking macro ──────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t _e = (call);                                                                   \
        if (_e != cudaSuccess) {                                                                   \
            fprintf(stderr, "CUDA error %s:%d  %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)

#define CUBLAS_CHECK(call)                                                                         \
    do {                                                                                           \
        cublasStatus_t _s = (call);                                                                \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                                         \
            fprintf(stderr, "cuBLAS error %s:%d  status=%d\n", __FILE__, __LINE__, (int)_s);       \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)

// ── Device memory helpers ─────────────────────────────────────────────────────
inline half *alloc_fp16(size_t elems) {
    half *ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, elems * sizeof(half)));
    return ptr;
}

// ── Host-side FP16 matrix initialisation (uniform random in [-1, 1]) ─────────
inline bool fill_random_fp16(half *d_ptr, size_t elems, unsigned seed = 42) {
    std::vector<half> host(elems);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : host)
        v = __float2half(dist(rng));
    return CUDA_TRY(cudaMemcpy(d_ptr, host.data(), elems * sizeof(half), cudaMemcpyHostToDevice));
}

// ── Relative-error correctness check ─────────────────────────────────────────

inline bool check_error(half *ref, half *cmp, size_t elems, float atol = 1e-3f, float rtol = 1e-2f,
                        size_t print_first_n = 5) {
    std::vector<half> h_ref(elems), h_cmp(elems);
    if (!CUDA_TRY(cudaMemcpy(h_ref.data(), ref, elems * sizeof(half), cudaMemcpyDeviceToHost)))
        return false;
    if (!CUDA_TRY(cudaMemcpy(h_cmp.data(), cmp, elems * sizeof(half), cudaMemcpyDeviceToHost)))
        return false;

    float max_abs_err = 0.f;
    float max_rel_err = 0.f;
    size_t max_abs_idx = 0;
    size_t max_rel_idx = 0;

    size_t n_print = 0;
    bool passed = true;

    for (size_t i = 0; i < elems; ++i) {
        float r = __half2float(h_ref[i]);
        float c = __half2float(h_cmp[i]);

        float abs_err = std::abs(r - c);
        float rel_err = abs_err / std::max(std::abs(r), 1e-6f);

        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            max_abs_idx = i;
        }
        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
            max_rel_idx = i;
        }

        bool ok = abs_err <= (atol + rtol * std::abs(r));
        if (!ok) {
            if (n_print < print_first_n) {
                fprintf(stdout, "  mismatch[%zu]: ref=%.6f  cmp=%.6f  abs=%.6e  rel=%.6e\n", i, r,
                        c, abs_err, rel_err);
                ++n_print;
            }
            passed = false;
        }
    }

    fprintf(stdout, "  max absolute error = %e at [%zu]\n", max_abs_err, max_abs_idx);
    fprintf(stdout, "  max relative error = %e at [%zu]  (%s)\n", max_rel_err, max_rel_idx,
            passed ? "PASS" : "FAIL");

    return passed;
}

// ── Timing helpers ────────────────────────────────────────────────────────────
struct CudaTimer {
    cudaEvent_t start_{}, stop_{};
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() { (void)CUDA_TRY(cudaEventRecord(start_)); }
    // Returns elapsed ms, or a negative value if timing failed.
    float stop() {
        float ms = -1.f;
        if (!CUDA_TRY(cudaEventRecord(stop_)))
            return ms;
        if (!CUDA_TRY(cudaEventSynchronize(stop_)))
            return ms;
        if (!CUDA_TRY(cudaEventElapsedTime(&ms, start_, stop_)))
            return -1.f;
        return ms;
    }
};

// ── TFLOPS from milliseconds ──────────────────────────────────────────────────
inline double tflops(int m, int n, int k, double ms) {
    // 2*M*N*K FLOPs for GEMM
    return 2.0 * m * n * k / (ms * 1e9);
}
