namespace gemm_8192x8192_v4 {

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
// PTX helpers (ldmatrix + mma)
// ------------------------------
#define LDMATRIX_X4(R0, R1, R2, R3, ADDR)                                         \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                         \
                 : "r"(ADDR))

#define LDMATRIX_X2(R0, R1, ADDR)                                               \
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"     \
                 : "=r"(R0), "=r"(R1)                                           \
                 : "r"(ADDR))

// FP32 accumulation MMA: f32.f16.f16.f32
#define HMMA16816(RC0, RC1, RC2, RC3, RA0, RA1, RA2, RA3, RB0, RB1)            \
    asm volatile(                                                              \
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "                   \
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"   \
        : "+f"(RC0), "+f"(RC1), "+f"(RC2), "+f"(RC3)                           \
        : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3),                              \
          "r"(RB0), "r"(RB1))

// ------------------------------
// cp.async (sm_80): bypass L1 / register, GMEM -> SMEM directly
// ------------------------------
#define CP_ASYNC_CG(dst_smem_addr, src_gmem_ptr, bytes)                        \
    asm volatile(                                                              \
        "cp.async.cg.shared.global [%0], [%1], %2;\n"                          \
        :                                                                      \
        : "r"(dst_smem_addr), "l"(src_gmem_ptr), "n"(bytes))

#define CP_ASYNC_CA(dst_smem_addr, src_gmem_ptr, bytes)                        \
    asm volatile(                                                              \
        "cp.async.ca.shared.global [%0], [%1], %2;\n"                          \
        :                                                                      \
        : "r"(dst_smem_addr), "l"(src_gmem_ptr), "n"(bytes))

#define CP_ASYNC_COMMIT_GROUP()                                                \
    asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N)                                                 \
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

#define CP_ASYNC_WAIT_ALL()                                                    \
    asm volatile("cp.async.wait_all;\n" ::)

// ------------------------------
// Kernel config
// ------------------------------
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define BLOCK_ROWS 256
#define BLOCK_COLS 128

#define WARP_ROWS 64
#define WARP_COLS 64

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define BLOCK_ROW_TILES 16
#define BLOCK_COL_TILES 16

#define WARP_ROW_TILES 8
#define WARP_COL_TILES 4

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK 256

#define CHUNK_K 2

#define CHUNK_LINE_BYTES 64
#define CHUNK_COPY_LINES_PER_WARP 8
#define CHUNK_COPY_LINE_LANES 4

#define AB_SMEM_STRIDE 32

#define C_SMEM_STRIDE 128
#define C_SMEM_OFFSET 64

#define BLOCK_STRIDE 16

__global__ void mmaBaseKernel(const half *__restrict__ A,
                              const half *__restrict__ B,
                              half *__restrict__ C,
                              int M, int N, int K) {
    const size_t M_tiles = div_ceil(M, (size_t)MMA_M);
    const size_t N_tiles = div_ceil(N, (size_t)MMA_N);
    const size_t K_tiles = div_ceil(K, (size_t)MMA_K);

    const size_t block_tile_i =
        (blockIdx.z % 2) ? ((gridDim.y - blockIdx.y - 1) * (size_t)BLOCK_COL_TILES)
                         : ((size_t)blockIdx.y * (size_t)BLOCK_COL_TILES);
    const size_t block_tile_j = ((size_t)blockIdx.z * (size_t)gridDim.x + (size_t)blockIdx.x) * (size_t)BLOCK_ROW_TILES;

    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles) return;

    extern __shared__ half smem[][AB_SMEM_STRIDE];

    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    constexpr size_t B_smem_idx_off = BLOCK_ROWS;

    half *smem_warp_tile_row_ptr =
        &smem[0][0] + (warp_id / BLOCK_ROW_WARPS) * (size_t)C_SMEM_STRIDE * (size_t)WARP_ROWS;

    const half *smem_warp_stream_ptr =
        &smem[0][0] + warp_id * (size_t)MMA_M * 2 * (size_t)C_SMEM_STRIDE;

    const size_t gmem_idx = (block_tile_i + warp_id * 2) * (size_t)MMA_M * N + block_tile_j * (size_t)MMA_N;
    half *dst_gmem_warp_stream_ptr = &C[gmem_idx];

    // FP32 accumulator: 4 floats per MMA tile (was 2 x uint32_t for FP16)
    float RC[WARP_COL_TILES][WARP_ROW_TILES][4];

#pragma unroll
    for (int i = 0; i < (int)WARP_COL_TILES; ++i) {
#pragma unroll
        for (int j = 0; j < (int)WARP_ROW_TILES; ++j) {
            RC[i][j][0] = 0.0f;
            RC[i][j][1] = 0.0f;
            RC[i][j][2] = 0.0f;
            RC[i][j][3] = 0.0f;
        }
    }

    const half *A_warp_ptr = &A[block_tile_i * (size_t)MMA_M * K] +
                             ((size_t)BLOCK_ROWS / (size_t)WARPS_PER_BLOCK) * (size_t)K * warp_id;
    const half *B_warp_ptr = &B[block_tile_j * (size_t)MMA_N * K] +
                             ((size_t)BLOCK_COLS / (size_t)WARPS_PER_BLOCK) * (size_t)K * warp_id;

    constexpr size_t A_smem_iters = BLOCK_ROWS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);
    constexpr size_t B_smem_iters = BLOCK_COLS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);

#pragma unroll
    for (size_t tile_k = 0; tile_k < K_tiles; tile_k += CHUNK_K) {
        // ---- Load A to smem ----
        size_t A_smem_idx = ((size_t)BLOCK_ROWS / (size_t)WARPS_PER_BLOCK) * warp_id;
        int4 *A_lane_ptr =
            (int4 *)(A_warp_ptr + tile_k * (size_t)MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * (size_t)K) +
            (lane_id % CHUNK_COPY_LINE_LANES);
        A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < A_smem_iters; ++i) {
            uint32_t A_smem_addr = __cvta_generic_to_shared(
                (int4 *)&smem[A_smem_idx][0] + (lane_id % CHUNK_COPY_LINE_LANES));
            CP_ASYNC_CG(A_smem_addr, A_lane_ptr, 16);

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + (size_t)CHUNK_COPY_LINES_PER_WARP * (size_t)K);
            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        // ---- Load B to smem ----
        size_t B_smem_idx = B_smem_idx_off + ((size_t)BLOCK_COLS / (size_t)WARPS_PER_BLOCK) * warp_id;
        int4 *B_lane_ptr =
            (int4 *)(B_warp_ptr + tile_k * (size_t)MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * (size_t)K) +
            (lane_id % CHUNK_COPY_LINE_LANES);
        B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < B_smem_iters; ++i) {
            uint32_t B_smem_addr = __cvta_generic_to_shared(
                (int4 *)&smem[B_smem_idx][0] + (lane_id % CHUNK_COPY_LINE_LANES));
            CP_ASYNC_CG(B_smem_addr, B_lane_ptr, 16);

            B_lane_ptr = (int4 *)((half *)B_lane_ptr + (size_t)CHUNK_COPY_LINES_PER_WARP * (size_t)K);
            B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_ALL();

        __syncthreads();

#pragma unroll
        for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
            uint32_t RA[WARP_COL_TILES][4];
            uint32_t RB[WARP_ROW_TILES][2];

#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
                size_t A_idx = (warp_id / BLOCK_ROW_WARPS) * (size_t)WARP_ROWS + i * (size_t)MMA_M;
                uint32_t A_addr = __cvta_generic_to_shared(
                    &smem[A_idx + (lane_id % 16)][k_step * (size_t)MMA_K + (lane_id / 16) * 8]
                );
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], A_addr);
            }

#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t B_idx = B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * (size_t)WARP_COLS + j * (size_t)MMA_N;
                uint32_t B_addr = __cvta_generic_to_shared(
                    &smem[B_idx + (lane_id % 8)][k_step * (size_t)MMA_K + ((lane_id / 8) % 2) * 8]
                );
                LDMATRIX_X2(RB[j][0], RB[j][1], B_addr);
            }

#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
                for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                    size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;
                    HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RC[i][j_s][2], RC[i][j_s][3],
                              RA[i][0], RA[i][1], RA[i][2], RA[i][3],
                              RB[j_s][0], RB[j_s][1]);
                }
            }
        }

        __syncthreads();
    }

    // ---- Store to smem (FP32 -> FP16 conversion) ----
#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            half *lane_ptr0 =
                smem_warp_tile_row_ptr +
                (i * (size_t)MMA_M + lane_id / 4) * (size_t)C_SMEM_STRIDE +
                (warp_id % BLOCK_ROW_WARPS) * (size_t)C_SMEM_OFFSET +
                j * (size_t)MMA_N +
                (lane_id % 4) * (sizeof(uint32_t) / sizeof(half));

            half *lane_ptr1 =
                smem_warp_tile_row_ptr +
                (i * (size_t)MMA_M + lane_id / 4 + 8) * (size_t)C_SMEM_STRIDE +
                (warp_id % BLOCK_ROW_WARPS) * (size_t)C_SMEM_OFFSET +
                j * (size_t)MMA_N +
                (lane_id % 4) * (sizeof(uint32_t) / sizeof(half));

            union { __half2 h2; uint32_t u32; } pack0, pack1;

            pack0.h2 = __floats2half2_rn(RC[i][j][0], RC[i][j][1]);
            pack1.h2 = __floats2half2_rn(RC[i][j][2], RC[i][j][3]);

            *((uint32_t *)(lane_ptr0)) = pack0.u32;
            *((uint32_t *)(lane_ptr1)) = pack1.u32;
        }
    }

    __syncthreads();

    // ---- Stream out ----
#pragma unroll
    for (size_t i = 0; i < MMA_M; ++i) {
        *((int4 *)(dst_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * (size_t)N) + (lane_id % 16)) =
            *((int4 *)(smem_warp_stream_ptr + (i * 2 + lane_id / 16) * (size_t)C_SMEM_STRIDE) + (lane_id % 16));
    }
}

static size_t initMmaBase() {
    int dev_id = 0;
    CUDA_CHECK(cudaGetDevice(&dev_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_id));

    size_t smem_max_size =
        std::max((size_t)(BLOCK_ROWS + BLOCK_COLS) * (size_t)AB_SMEM_STRIDE * sizeof(half),
                 (size_t)BLOCK_ROWS * (size_t)C_SMEM_STRIDE * sizeof(half));

    printf("smem_max_size = %.1f KB (%zu bytes)\n", (double)smem_max_size / 1024.0, smem_max_size);

    CUDA_CHECK(cudaFuncSetAttribute(
        mmaBaseKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_max_size));

    return smem_max_size;
}

static void launchKernel(half *dA, half *dB, half *dC, int M, int N, int K) {
    static size_t smem_max = initMmaBase();
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCK_STRIDE,
              (unsigned)div_ceil(M, (size_t)BLOCK_ROWS),
              (unsigned)div_ceil(N, (size_t)BLOCK_COLS * (size_t)BLOCK_STRIDE));
    mmaBaseKernel<<<grid, block, smem_max>>>(dA, dB, dC, M, N, K);
}
} // namespace gemm_8192x8192_v4

using gemm_8192x8192_v4::launchKernel;
