#include <cuda.h>
#include <cuda/barrier>
#include <cuda/std/atomic>
#include <string>

#include "backend.cuh"
#include "registry.cuh"

#include "a100/gemm_8192x8192_v0.cuh"
#include "a100/gemm_8192x8192_v1.cuh"
#include "a100/gemm_8192x8192_v10.cuh"
#include "a100/gemm_8192x8192_v11.cuh"
#include "a100/gemm_8192x8192_v2.cuh"
#include "a100/gemm_8192x8192_v3.cuh"
#include "a100/gemm_8192x8192_v4.cuh"
#include "a100/gemm_8192x8192_v5.cuh"
#include "a100/gemm_8192x8192_v6.cuh"
#include "a100/gemm_8192x8192_v7.cuh"
#include "a100/gemm_8192x8192_v8.cuh"
#include "a100/gemm_8192x8192_v9.cuh"

// KernelFn: the uniform function signature all A100 kernels expose to this hub.
// Each .cuh provides a run_fp16() wrapper that matches this type.
using KernelFn = void (*)(half *, half *, half *, int, int, int);

template <KernelFn Fn> class GeneratedA100KernelBackend : public GemmBackend {
  public:
    explicit GeneratedA100KernelBackend(std::string backend_name)
        : backend_name_(std::move(backend_name)) {}

    void run(half *A, half *B, half *C, int M, int N, int K) override { Fn(A, B, C, M, N, K); }

    std::string name() const override { return backend_name_; }

    void teardown() override {}

  private:
    std::string backend_name_;
};

// ── Register A100 kernels ─────────────────────────────────────────────────────
// To add a new kernel:
//   1. #include "your_kernel.cuh"
//   2. REGISTER_A100_KERNEL("display name", GeneratedA100KernelBackend<your_ns::run_fp16>);

REGISTER_A100_KERNEL("VibeGEMM A100 8192^2 v0",
                     GeneratedA100KernelBackend<gemm_8192x8192_v0::launchKernel>);

REGISTER_A100_KERNEL("VibeGEMM A100 8192^2 v1",
                     GeneratedA100KernelBackend<gemm_8192x8192_v1::launchKernel>);

REGISTER_A100_KERNEL("VibeGEMM A100 8192^2 v2",
                     GeneratedA100KernelBackend<gemm_8192x8192_v2::launchKernel>);

REGISTER_A100_KERNEL("VibeGEMM A100 8192^2 v3",
                     GeneratedA100KernelBackend<gemm_8192x8192_v3::launchKernel>);

REGISTER_A100_KERNEL("VibeGEMM A100 8192^2 v4",
                     GeneratedA100KernelBackend<gemm_8192x8192_v4::launchKernel>);

REGISTER_A100_KERNEL("VibeGEMM A100 8192^2 v5",
                     GeneratedA100KernelBackend<gemm_8192x8192_v5::launchKernel>);

REGISTER_A100_KERNEL("VibeGEMM A100 8192^2 v6",
                     GeneratedA100KernelBackend<gemm_8192x8192_v6::launchKernel>);

REGISTER_A100_KERNEL("VibeGEMM A100 8192^2 v7",
                     GeneratedA100KernelBackend<gemm_8192x8192_v7::launchKernel>);

REGISTER_A100_KERNEL("VibeGEMM A100 8192^2 v8",
                     GeneratedA100KernelBackend<gemm_8192x8192_v8::launchKernel>);

REGISTER_A100_KERNEL("VibeGEMM A100 8192^2 v9",
                     GeneratedA100KernelBackend<gemm_8192x8192_v9::launchKernel>);

REGISTER_A100_KERNEL("VibeGEMM A100 8192^2 v10",
                     GeneratedA100KernelBackend<gemm_8192x8192_v10::launchKernel>);

REGISTER_A100_KERNEL("VibeGEMM A100 8192^2 v11",
                     GeneratedA100KernelBackend<gemm_8192x8192_v11::launchKernel>);
