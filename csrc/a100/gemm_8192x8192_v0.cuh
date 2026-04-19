namespace gemm_8192x8192_v0 {

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_e));                                   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

__host__ __device__ __forceinline__ size_t div_ceil(size_t a, size_t b) {
    return (a + b - 1) / b;
}

// ------------------------------
// Kernel config
// ------------------------------
#define TILE 16
#define THREADS_PER_BLOCK (TILE * TILE)

__global__ void mmaScalarTiledKernel(const half *__restrict__ A,
                                     const half *__restrict__ B,
                                     half *__restrict__ C,
                                     size_t M, size_t N, size_t K) {
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int row = (int)blockIdx.y * TILE + ty;
    const int col = (int)blockIdx.x * TILE + tx;

    __shared__ half sA[TILE][TILE];
    __shared__ half sB[TILE][TILE];

    float acc = 0.0f;

    const int num_k_tiles = (int)div_ceil(K, (size_t)TILE);

    for (int tk = 0; tk < num_k_tiles; ++tk) {
        const int kA = tk * TILE + tx;
        const int kB = tk * TILE + ty;

        // ----------- Load A tile -----------
        if (row < (int)M && kA < (int)K) {
            sA[ty][tx] = A[(size_t)row * K + (size_t)kA];
        } else {
            sA[ty][tx] = __float2half(0.0f);
        }

        // ----------- Load B tile -----------
        // This keeps your original layout assumption:
        // B is addressed as B[n][k], i.e. leading dimension is K.
        if (col < (int)N && kB < (int)K) {
            sB[ty][tx] = B[(size_t)col * K + (size_t)kB];
        } else {
            sB[ty][tx] = __float2half(0.0f);
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float a = __half2float(sA[ty][k]);
            float b = __half2float(sB[k][tx]);
            acc += a * b;
        }

        __syncthreads();
    }

    if (row < (int)M && col < (int)N) {
        C[(size_t)row * N + (size_t)col] = __float2half(acc);
    }
}

static void launchKernel( half *dA, half *dB, half *dC,
                                 int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((unsigned)div_ceil(N, (size_t)TILE),
              (unsigned)div_ceil(M, (size_t)TILE));
    mmaScalarTiledKernel<<<grid, block>>>(dA, dB, dC, M, N, K);
}
    } // namespace gemm_8192x8192_v0
    
    using gemm_8192x8192_v0::launchKernel;