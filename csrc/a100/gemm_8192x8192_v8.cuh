namespace gemm_8192x8192_v8 {

    #define CUDA_CHECK(call)                                                       \
    do {                                                                           \
        cudaError_t _e = (call);                                                   \
        if (_e != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
                    cudaGetErrorString(_e));                                       \
            std::exit(1);                                                          \
        }                                                                          \
    } while (0)
    
    __host__ __device__ __forceinline__ size_t div_ceil(size_t a, size_t b) {
        return (a + b - 1) / b;
    }
    
    // ------------------------------
    // PTX helpers (ldmatrix + mma)
    // ------------------------------
    #define LDMATRIX_X4(R0, R1, R2, R3, ADDR)                                           \
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" \
                     : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                           \
                     : "r"(ADDR))
    
    #define LDMATRIX_X2(R0, R1, ADDR)                                                 \
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"     \
                     : "=r"(R0), "=r"(R1)                                             \
                     : "r"(ADDR))
    
    // Correct form for mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    // 4 FP32 accumulators per thread.
    #define HMMA16816_F32(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1)                    \
        asm volatile(                                                                 \
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "                      \
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"      \
            : "+f"(C0), "+f"(C1), "+f"(C2), "+f"(C3)                                  \
            : "r"(A0), "r"(A1), "r"(A2), "r"(A3),                                     \
              "r"(B0), "r"(B1))
    
    // ------------------------------
    // cp.async (sm_80): GMEM -> SMEM direct, bypass L1 / register
    // ------------------------------
    #define CP_ASYNC_CG(dst_smem_addr, src_gmem_ptr, bytes)                          \
        asm volatile(                                                                \
            "cp.async.cg.shared.global [%0], [%1], %2;\n"                            \
            :                                                                        \
            : "r"(dst_smem_addr), "l"(src_gmem_ptr), "n"(bytes))
    
    #define CP_ASYNC_COMMIT_GROUP()                                                  \
        asm volatile("cp.async.commit_group;\n" ::)
    
    #define CP_ASYNC_WAIT_GROUP(N)                                                   \
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N))
    
    #define CP_ASYNC_WAIT_ALL()                                                      \
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
    
    #define BLOCK_STRIDE 16
    
    // Pipeline depth
    #define K_STAGE 3
    #define SMEM_STAGE_ROWS (BLOCK_ROWS + BLOCK_COLS)   // = 384
    
    // ==========================================================
    // Shared-memory col swizzle
    // ==========================================================
    __device__ __forceinline__ size_t swz_col(size_t row, size_t col) {
        return (col + ((row & 7) >> 1) * 8) & (size_t)(AB_SMEM_STRIDE - 1);
    }
    
    __global__ void mmaAsyncStageKernel(const half *__restrict__ A,
                                        const half *__restrict__ B,
                                        half *__restrict__ C,
                                        int M, int N, int K) {
        const size_t M_tiles = div_ceil((size_t)M, (size_t)MMA_M);
        const size_t N_tiles = div_ceil((size_t)N, (size_t)MMA_N);
        const size_t K_tiles = div_ceil((size_t)K, (size_t)MMA_K);
    
        const size_t block_tile_i =
            (blockIdx.z % 2) ? ((gridDim.y - blockIdx.y - 1) * (size_t)BLOCK_COL_TILES)
                             : ((size_t)blockIdx.y * (size_t)BLOCK_COL_TILES);
        const size_t block_tile_j =
            ((size_t)blockIdx.z * (size_t)gridDim.x + (size_t)blockIdx.x) * (size_t)BLOCK_ROW_TILES;
    
        if (block_tile_i >= M_tiles || block_tile_j >= N_tiles) return;
    
        extern __shared__ half smem[][AB_SMEM_STRIDE];
    
        const size_t warp_id = threadIdx.x / WARP_SIZE;
        const size_t lane_id = threadIdx.x % WARP_SIZE;
    
        constexpr size_t B_smem_idx_off = BLOCK_ROWS;
    
        // --- Register accumulator ---
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
    
        // --- Global base pointers (per-warp slice of A and B) ---
        const half *A_warp_ptr = &A[block_tile_i * (size_t)MMA_M * (size_t)K] +
                                 ((size_t)BLOCK_ROWS / (size_t)WARPS_PER_BLOCK) * (size_t)K * warp_id;
        const half *B_warp_ptr = &B[block_tile_j * (size_t)MMA_N * (size_t)K] +
                                 ((size_t)BLOCK_COLS / (size_t)WARPS_PER_BLOCK) * (size_t)K * warp_id;
    
        constexpr size_t A_smem_iters = BLOCK_ROWS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);
        constexpr size_t B_smem_iters = BLOCK_COLS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);
    
        const size_t NI = div_ceil(K_tiles, (size_t)CHUNK_K);
    
        // ==========================================================
        // ISSUE_STAGE_A / ISSUE_STAGE_B
        // ==========================================================
    #define ISSUE_STAGE_A(SLOT, TK)                                                            \
        do {                                                                                   \
            const size_t _stage_off = (size_t)(SLOT) * (size_t)SMEM_STAGE_ROWS;                \
            const size_t _tk        = (size_t)(TK);                                            \
            size_t  _A_smem_idx = _stage_off                                                   \
                                + ((size_t)BLOCK_ROWS / (size_t)WARPS_PER_BLOCK) * warp_id     \
                                + (lane_id / CHUNK_COPY_LINE_LANES);                           \
            int4   *_A_lane_ptr = (int4 *)(A_warp_ptr + _tk * (size_t)MMA_K                    \
                                + (lane_id / CHUNK_COPY_LINE_LANES) * (size_t)K)               \
                                + (lane_id % CHUNK_COPY_LINE_LANES);                           \
            _Pragma("unroll")                                                                  \
            for (size_t _ii = 0; _ii < A_smem_iters; ++_ii) {                                  \
                size_t   _col_halfs = (lane_id % CHUNK_COPY_LINE_LANES) * 8;                   \
                size_t   _col_perm  = swz_col(_A_smem_idx, _col_halfs);                        \
                uint32_t _addr = __cvta_generic_to_shared(&smem[_A_smem_idx][_col_perm]);      \
                CP_ASYNC_CG(_addr, _A_lane_ptr, 16);                                           \
                _A_lane_ptr = (int4 *)((half *)_A_lane_ptr                                     \
                                       + (size_t)CHUNK_COPY_LINES_PER_WARP * (size_t)K);       \
                _A_smem_idx += CHUNK_COPY_LINES_PER_WARP;                                      \
            }                                                                                  \
        } while (0)
    
    #define ISSUE_STAGE_B(SLOT, TK)                                                            \
        do {                                                                                   \
            const size_t _stage_off = (size_t)(SLOT) * (size_t)SMEM_STAGE_ROWS;                \
            const size_t _tk        = (size_t)(TK);                                            \
            size_t  _B_smem_idx = _stage_off + (size_t)B_smem_idx_off                          \
                                + ((size_t)BLOCK_COLS / (size_t)WARPS_PER_BLOCK) * warp_id     \
                                + (lane_id / CHUNK_COPY_LINE_LANES);                           \
            int4   *_B_lane_ptr = (int4 *)(B_warp_ptr + _tk * (size_t)MMA_K                    \
                                + (lane_id / CHUNK_COPY_LINE_LANES) * (size_t)K)               \
                                + (lane_id % CHUNK_COPY_LINE_LANES);                           \
            _Pragma("unroll")                                                                  \
            for (size_t _ii = 0; _ii < B_smem_iters; ++_ii) {                                  \
                size_t   _col_halfs = (lane_id % CHUNK_COPY_LINE_LANES) * 8;                   \
                size_t   _col_perm  = swz_col(_B_smem_idx, _col_halfs);                        \
                uint32_t _addr = __cvta_generic_to_shared(&smem[_B_smem_idx][_col_perm]);      \
                CP_ASYNC_CG(_addr, _B_lane_ptr, 16);                                           \
                _B_lane_ptr = (int4 *)((half *)_B_lane_ptr                                     \
                                       + (size_t)CHUNK_COPY_LINES_PER_WARP * (size_t)K);       \
                _B_smem_idx += CHUNK_COPY_LINES_PER_WARP;                                      \
            }                                                                                  \
        } while (0)
    
    #define ISSUE_STAGE(SLOT, TK)                                                              \
        do {                                                                                   \
            ISSUE_STAGE_A(SLOT, TK);                                                           \
            ISSUE_STAGE_B(SLOT, TK);                                                           \
        } while (0)
    
        // ==========================================================
        // LOAD_AB_TO_REG
        // ==========================================================
    #define LOAD_AB_TO_REG(REG_IDX, SMEM_SLOT, K_STEP)                                         \
        do {                                                                                   \
            const size_t _read_off = (size_t)(SMEM_SLOT) * (size_t)SMEM_STAGE_ROWS;            \
            _Pragma("unroll")                                                                  \
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {                                      \
                size_t A_idx = _read_off                                                       \
                             + (warp_id / BLOCK_ROW_WARPS) * (size_t)WARP_ROWS                 \
                             + i * (size_t)MMA_M;                                              \
                size_t _row = A_idx + (lane_id % 16);                                          \
                size_t _col = (size_t)(K_STEP) * (size_t)MMA_K + (lane_id / 16) * 8;           \
                uint32_t A_addr = __cvta_generic_to_shared(                                    \
                    &smem[_row][swz_col(_row, _col)]);                                         \
                LDMATRIX_X4(RA[(REG_IDX)][i][0], RA[(REG_IDX)][i][1],                          \
                            RA[(REG_IDX)][i][2], RA[(REG_IDX)][i][3], A_addr);                 \
            }                                                                                  \
            _Pragma("unroll")                                                                  \
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {                                      \
                size_t B_idx = _read_off + (size_t)B_smem_idx_off                              \
                             + (warp_id % BLOCK_ROW_WARPS) * (size_t)WARP_COLS                 \
                             + j * (size_t)MMA_N;                                              \
                size_t _row = B_idx + (lane_id % 8);                                           \
                size_t _col = (size_t)(K_STEP) * (size_t)MMA_K + ((lane_id / 8) % 2) * 8;      \
                uint32_t B_addr = __cvta_generic_to_shared(                                    \
                    &smem[_row][swz_col(_row, _col)]);                                         \
                LDMATRIX_X2(RB[(REG_IDX)][j][0], RB[(REG_IDX)][j][1], B_addr);                 \
            }                                                                                  \
        } while (0)
    
        // ==========================================================
        // DO_HMMA
        // ==========================================================
    #define DO_HMMA(REG_IDX)                                                                    \
        do {                                                                                    \
            _Pragma("unroll")                                                                   \
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {                                       \
                _Pragma("unroll")                                                               \
                for (size_t j = 0; j < WARP_ROW_TILES; ++j) {                                   \
                    size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;                        \
                    HMMA16816_F32(RC[i][j_s][0], RC[i][j_s][1], RC[i][j_s][2], RC[i][j_s][3],  \
                                  RA[(REG_IDX)][i][0], RA[(REG_IDX)][i][1],                     \
                                  RA[(REG_IDX)][i][2], RA[(REG_IDX)][i][3],                     \
                                  RB[(REG_IDX)][j_s][0], RB[(REG_IDX)][j_s][1]);                \
                }                                                                               \
            }                                                                                   \
        } while (0)
    
        // Register double-buffers for ldmatrix prefetch
        uint32_t RA[2][WARP_COL_TILES][4];
        uint32_t RB[2][WARP_ROW_TILES][2];
    
        // ==========================================================
        // PROLOGUE
        // ==========================================================
    #pragma unroll
        for (int s = 0; s < K_STAGE - 1; ++s) {
            ISSUE_STAGE(s, (size_t)s * CHUNK_K);
            CP_ASYNC_COMMIT_GROUP();
        }
    
        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    
        int read_s  = 0;
        int write_s = K_STAGE - 1;
    
        LOAD_AB_TO_REG(0, read_s, 0);
    
        // ==========================================================
        // MAIN LOOP
        // ==========================================================
        static_assert(CHUNK_K == 2, "main loop is hand-unrolled for CHUNK_K=2");
    
        const size_t main_iters = (NI > (size_t)(K_STAGE - 1)) ? (NI - (size_t)(K_STAGE - 1)) : 0;
    
    #pragma unroll 1
        for (size_t iter = 0; iter < main_iters; ++iter) {
            const size_t issue_tile_k = (iter + (size_t)(K_STAGE - 1)) * (size_t)CHUNK_K;
    
            ISSUE_STAGE_A(write_s, issue_tile_k);
            LOAD_AB_TO_REG(1, read_s, 1);
            DO_HMMA(0);
    
            ISSUE_STAGE_B(write_s, issue_tile_k);
            CP_ASYNC_COMMIT_GROUP();
    
            read_s  = (read_s  + 1) % K_STAGE;
            write_s = (write_s + 1) % K_STAGE;
    
            CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
            __syncthreads();
    
            LOAD_AB_TO_REG(0, read_s, 0);
            DO_HMMA(1);
        }
    
        // ==========================================================
        // EPILOGUE
        // ==========================================================
        static_assert(K_STAGE == 3, "epilogue is hand-unrolled for K_STAGE=3");
    
        LOAD_AB_TO_REG(1, read_s, 1);
        DO_HMMA(0);
    
        read_s = (read_s + 1) % K_STAGE;
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
        LOAD_AB_TO_REG(0, read_s, 0);
        DO_HMMA(1);
    
        LOAD_AB_TO_REG(1, read_s, 1);
        DO_HMMA(0);
        DO_HMMA(1);
        read_s = (read_s + 1) % K_STAGE;
    
        CP_ASYNC_WAIT_ALL();
        __syncthreads();
    
    #undef ISSUE_STAGE
    #undef ISSUE_STAGE_A
    #undef ISSUE_STAGE_B
    #undef LOAD_AB_TO_REG
    #undef DO_HMMA
    
        // ==========================================================
        // C store phase
        //
        // m16n8k16 row.col f32 accumulator mapping per lane:
        //   group = lane_id / 4   in [0, 7]
        //   tid4  = lane_id % 4   in [0, 3]
        //
        //   RC[..][..][0] -> (row + group + 0, col + tid4*2 + 0)
        //   RC[..][..][1] -> (row + group + 0, col + tid4*2 + 1)
        //   RC[..][..][2] -> (row + group + 8, col + tid4*2 + 0)
        //   RC[..][..][3] -> (row + group + 8, col + tid4*2 + 1)
        // ==========================================================
        const size_t warp_row_base = (warp_id / BLOCK_ROW_WARPS) * (size_t)WARP_ROWS;
        const size_t warp_col_base = (warp_id % BLOCK_ROW_WARPS) * (size_t)WARP_COLS;
    
        const size_t lane_group = lane_id >> 2; // 0..7
        const size_t lane_tid4  = lane_id & 3;  // 0..3
    
    #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
    #pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                const size_t tile_row = warp_row_base + i * (size_t)MMA_M;
                const size_t tile_col = warp_col_base + j * (size_t)MMA_N;
    
                const size_t r0 = block_tile_i * (size_t)MMA_M + tile_row + lane_group;
                const size_t r1 = r0 + 8;
                const size_t c0 = block_tile_j * (size_t)MMA_N + tile_col + lane_tid4 * 2;
                const size_t c1 = c0 + 1;
    
                if (r0 < (size_t)M && c0 < (size_t)N) C[r0 * (size_t)N + c0] = __float2half_rn(RC[i][j][0]);
                if (r0 < (size_t)M && c1 < (size_t)N) C[r0 * (size_t)N + c1] = __float2half_rn(RC[i][j][1]);
                if (r1 < (size_t)M && c0 < (size_t)N) C[r1 * (size_t)N + c0] = __float2half_rn(RC[i][j][2]);
                if (r1 < (size_t)M && c1 < (size_t)N) C[r1 * (size_t)N + c1] = __float2half_rn(RC[i][j][3]);
            }
        }
    }
    
    // ---- Host-side launcher ----
    static size_t initMmaAsyncStage3() {
        int dev_id = 0;
        CUDA_CHECK(cudaGetDevice(&dev_id));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_id));
    
        // Only AB pipeline storage is needed in smem now.
        size_t ab_bytes = (size_t)K_STAGE * (size_t)(BLOCK_ROWS + BLOCK_COLS)
                        * (size_t)AB_SMEM_STRIDE * sizeof(half);
    
        printf("smem_size = %.1f KB (%zu bytes)  [AB=%.1fKB  K_STAGE=%d]\n",
               (double)ab_bytes / 1024.0, ab_bytes,
               (double)ab_bytes / 1024.0, K_STAGE);
    
        CUDA_CHECK(cudaFuncSetAttribute(
            mmaAsyncStageKernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ab_bytes));
    
        return ab_bytes;
    }
    
    static void launchKernel(half *dA, half *dB, half *dC,
                             int M, int N, int K) {
        static size_t smem_bytes = initMmaAsyncStage3();
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(BLOCK_STRIDE,
                  (unsigned)div_ceil((size_t)M, (size_t)BLOCK_ROWS),
                  (unsigned)div_ceil((size_t)N, (size_t)BLOCK_COLS * (size_t)BLOCK_STRIDE));
        mmaAsyncStageKernel<<<grid, block, smem_bytes>>>(dA, dB, dC, M, N, K);
    }
    
    } // namespace gemm_8192x8192_v8
    
    using gemm_8192x8192_v8::launchKernel;