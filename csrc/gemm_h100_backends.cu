#include <cuda.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cuda/std/atomic>
#include <string>

#include "backend.cuh"
#include "registry.cuh"

#include "h100/gemm_8192x8192_v0.cuh"
#include "h100/gemm_8192x8192_v1.cuh"
#include "h100/gemm_8192x8192_v10.cuh"
#include "h100/gemm_8192x8192_v11.cuh"
#include "h100/gemm_8192x8192_v12.cuh"
#include "h100/gemm_8192x8192_v13.cuh"
#include "h100/gemm_8192x8192_v14.cuh"
#include "h100/gemm_8192x8192_v15.cuh"
#include "h100/gemm_8192x8192_v16.cuh"
#include "h100/gemm_8192x8192_v17.cuh"
#include "h100/gemm_8192x8192_v18.cuh"
#include "h100/gemm_8192x8192_v19.cuh"
#include "h100/gemm_8192x8192_v2.cuh"
#include "h100/gemm_8192x8192_v20.cuh"
#include "h100/gemm_8192x8192_v3.cuh"
#include "h100/gemm_8192x8192_v4.cuh"
#include "h100/gemm_8192x8192_v5.cuh"
#include "h100/gemm_8192x8192_v6.cuh"
#include "h100/gemm_8192x8192_v7.cuh"
#include "h100/gemm_8192x8192_v8.cuh"
#include "h100/gemm_8192x8192_v9.cuh"

// KernelFn: the uniform function signature all H100 kernels expose to this hub.
// Each .cuh provides a run_fp16() wrapper that matches this type.
using KernelFn = void (*)(half *, half *, half *, int, int, int);

template <KernelFn Fn> class GeneratedH100KernelBackend : public GemmBackend {
  public:
    explicit GeneratedH100KernelBackend(std::string backend_name)
        : backend_name_(std::move(backend_name)) {}

    void run(half *A, half *B, half *C, int M, int N, int K) override { Fn(A, B, C, M, N, K); }

    std::string name() const override { return backend_name_; }

    void teardown() override {}

  private:
    std::string backend_name_;
};

// ── Register H100 kernels ─────────────────────────────────────────────────────
// To add a new kernel:
//   1. #include "your_kernel.cuh"
//   2. REGISTER_H100_KERNEL("display name", GeneratedH100KernelBackend<your_ns::run_fp16>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v0",
                     GeneratedH100KernelBackend<gemm_8192x8192_v0::launchKernel>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v1",
                     GeneratedH100KernelBackend<gemm_8192x8192_v1::launchKernel>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v2",
                     GeneratedH100KernelBackend<gemm_8192x8192_v2::launchKernel>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v3",
                     GeneratedH100KernelBackend<gemm_8192x8192_v3::launchKernel>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v4",
                     GeneratedH100KernelBackend<gemm_8192x8192_v4::launchKernel>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v5",
                     GeneratedH100KernelBackend<gemm_8192x8192_v5::launchKernel>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v6",
                     GeneratedH100KernelBackend<gemm_8192x8192_v6::launchKernel>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v7",
                     GeneratedH100KernelBackend<gemm_8192x8192_v7::launchKernel>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v8",
                     GeneratedH100KernelBackend<gemm_8192x8192_v8::launchKernel>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v9",
                     GeneratedH100KernelBackend<gemm_8192x8192_v9::launchKernel>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v10",
                     GeneratedH100KernelBackend<gemm_8192x8192_v10::launchKernel>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v11",
                     GeneratedH100KernelBackend<gemm_8192x8192_v11::launchKernel>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v12",
                     GeneratedH100KernelBackend<gemm_8192x8192_v12::launchKernel>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v13",
                     GeneratedH100KernelBackend<gemm_8192x8192_v13::launchKernel>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v14",
                     GeneratedH100KernelBackend<gemm_8192x8192_v14::launchKernel>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v15",
                     GeneratedH100KernelBackend<gemm_8192x8192_v15::launchKernel>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v16",
                     GeneratedH100KernelBackend<gemm_8192x8192_v16::launchKernel>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v17",
                     GeneratedH100KernelBackend<gemm_8192x8192_v17::launchKernel>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v18",
                     GeneratedH100KernelBackend<gemm_8192x8192_v18::launchKernel>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v19",
                     GeneratedH100KernelBackend<gemm_8192x8192_v19::launchKernel>);

REGISTER_H100_KERNEL("VibeGEMM H100 8192^2 v20",
                     GeneratedH100KernelBackend<gemm_8192x8192_v20::launchKernel>);