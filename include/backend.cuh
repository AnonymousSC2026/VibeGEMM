#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <string>

class GemmBackend {
  public:
    virtual ~GemmBackend() = default;

    // Launch the GEMM kernel.  Caller owns A/B/C device pointers.
    // C = alpha * A * B + beta * C   (alpha=1, beta=0 by convention)
    virtual void run(half *A, half *B, half *C, int M, int N, int K) = 0;

    // Short identifier printed in the benchmark report.
    virtual std::string name() const = 0;

    // Optional: release heavy resources between problem sizes.
    // Default is no-op; override when backend holds large workspace.
    virtual void teardown() {}

    // Returns false if the backend failed to load.
    // Engine skips correctness & benchmark for unavailable backends.
    virtual bool is_available() const { return true; }
};
