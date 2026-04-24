#pragma once

#include "backend.cuh"
#include "utils.cuh"
#include <cstddef>
#include <cstdio>
#include <cstdarg>
#include <memory>
#include <vector>

class BenchmarkEngine {
  public:
    static constexpr int kWarmup = 10;
    static constexpr int kBenchmark = 30;

    void set_log_file(FILE* fp) { fp_ = fp; }

    void add_backend(std::shared_ptr<GemmBackend> b) { backends_.push_back(std::move(b)); }

    void execute(int M, int N, int K) {
        log_print("\n=== GEMM  M=%d  N=%d  K=%d ===\n", M, N, K);

        if (!alloc_buffers(M, N, K)) {
            log_print("  [ERROR] device buffer allocation failed; skipping shape.\n");
            return;
        }

        if (!fill_random_fp16(d_A_, (size_t)M * K, /*seed=*/42) ||
            !fill_random_fp16(d_B_, (size_t)K * N, /*seed=*/77)) {
            log_print("  [ERROR] failed to upload random A/B; skipping shape.\n");
            return;
        }

        if (backends_.empty())
            return;
        auto &ref_backend = backends_[0];
        ref_backend->run(d_A_, d_B_, d_C_ref_, M, N, K);
        if (!cuda_check("reference")) {
            log_print("  [ERROR] reference backend failed; skipping shape.\n");
            ref_backend->teardown();
            return;
        }

        for (auto &b : backends_) {
            log_print("\n[%s]\n", b->name().c_str());

            if (!b->is_available()) {
                log_print("  [SKIP] backend not available\n");
                b->teardown();
                continue;
            }

            for (int i = 0; i < kWarmup; ++i)
                b->run(d_A_, d_B_, d_C_test_, M, N, K);

            CudaTimer timer;
            timer.start();
            for (int i = 0; i < kBenchmark; ++i)
                b->run(d_A_, d_B_, d_C_test_, M, N, K);
            float total_ms = timer.stop();
            if (total_ms < 0.f) {
                log_print("  [ERROR] CUDA timer failed; skipping backend.\n");
                b->teardown();
                continue;
            }

            float avg_ms = total_ms / kBenchmark;
            double tf = tflops(M, N, K, avg_ms);
            log_print("  avg latency = %.3f ms   %.2f TFLOPS\n", avg_ms, tf);

            if (!CUDA_TRY(cudaMemset(d_C_test_, 0, (size_t)M * N * sizeof(half)))) {
                log_print("  [WARN] cudaMemset failed; skipping correctness check.\n");
                b->teardown();
                continue;
            }
            b->run(d_A_, d_B_, d_C_test_, M, N, K);
            if (!cuda_check(b->name().c_str())) {
                log_print("  [WARN] CUDA failure on correctness run.\n");
                b->teardown();
                continue;
            }
            check_error(d_C_ref_, d_C_test_, (size_t)M * N);

            b->teardown();
        }
    }

  private:
    FILE* fp_ = nullptr;
    std::vector<std::shared_ptr<GemmBackend>> backends_;

    half *d_A_ = nullptr;
    half *d_B_ = nullptr;
    half *d_C_ref_ = nullptr;
    half *d_C_test_ = nullptr;
    size_t buf_A_ = 0;
    size_t buf_B_ = 0;
    size_t buf_C_ref_ = 0;
    size_t buf_C_tst_ = 0;

    // ── 同时输出到 stdout 和文件 ──────────────────────────────────────────────
    void log_print(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        vprintf(fmt, args);
        va_end(args);

        if (fp_) {
            va_start(args, fmt);
            vfprintf(fp_, fmt, args);
            va_end(args);
        }
    }

    bool alloc_buffers(int M, int N, int K) {
        auto need_A = (size_t)M * K * sizeof(half);
        auto need_B = (size_t)K * N * sizeof(half);
        auto need_C = (size_t)M * N * sizeof(half);

        auto realloc = [](half *&ptr, size_t &cap, size_t need) -> bool {
            if (need > cap) {
                if (ptr) {
                    if (!CUDA_TRY(cudaFree(ptr)))
                        return false;
                    ptr = nullptr;
                }
                void *raw = nullptr;
                if (!CUDA_TRY(cudaMalloc(&raw, need)))
                    return false;
                ptr = static_cast<half *>(raw);
                cap = need;
            }
            return true;
        };

        if (!realloc(d_A_, buf_A_, need_A) || !realloc(d_B_, buf_B_, need_B) ||
            !realloc(d_C_ref_, buf_C_ref_, need_C) || !realloc(d_C_test_, buf_C_tst_, need_C)) {
            fprintf(stderr, "BenchmarkEngine: buffer allocation failed; skipping shape.\n");
            return false;
        }
        return true;
    }
};