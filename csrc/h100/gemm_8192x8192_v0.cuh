namespace gemm_8192x8192_v0 {

using Barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// ===================================
// 1. Atom-Level
// ===================================

__device__ __forceinline__ uint64_t encodeMatrixDescriptor(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

__device__ __forceinline__ uint64_t makeSmemDescriptor(const half *ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= encodeMatrixDescriptor(addr);
    desc |= encodeMatrixDescriptor((uint64_t)16) << 16;
    desc |= encodeMatrixDescriptor((uint64_t)1024) << 32;
    desc |= 1ull << 62;
    return desc;
}

__device__ __forceinline__ void wgmmaFence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmmaCommit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N> __device__ __forceinline__ void wgmmaWait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait groups must be between 0 and 7.");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

template <int ScaleD = 1, int ScaleA = 1, int ScaleB = 1, int TransA = 0, int TransB = 0>
__device__ __forceinline__ void wgmmaM64N64K16(float d[32], const half *sA, const half *sB) {
    uint64_t desc_a = makeSmemDescriptor(sA);
    uint64_t desc_b = makeSmemDescriptor(sB);
    asm volatile("{\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16"
                 "{%0,%1,%2,%3,%4,%5,%6,%7, %8,%9,%10,%11,%12,%13,%14,%15,"
                 " %16,%17,%18,%19,%20,%21,%22,%23, %24,%25,%26,%27,%28,%29,%30,%31},"
                 " %32, %33, %34, %35, %36, %37, %38;\n"
                 "}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                   "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
                   "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]),
                   "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
                   "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
                   "+f"(d[30]), "+f"(d[31])
                 : "l"(desc_a), "l"(desc_b), "n"(ScaleD), "n"(ScaleA), "n"(ScaleB), "n"(TransA),
                   "n"(TransB));
}

// ===================================
// 2. Tile-Level
// ===================================
template <int M, int N, int K, int WGMMA_M, int WGMMA_N, int WGMMA_K, int THREADS> struct BlockShape {
    static constexpr int TileM = M;
    static constexpr int TileN = N;
    static constexpr int TileK = K;
    static constexpr int NumThreads = THREADS;

    static constexpr int WgmmaM = WGMMA_M;
    static constexpr int WgmmaN = WGMMA_N;
    static constexpr int WgmmaK = WGMMA_K;

    static constexpr int NumWgmmaM = TileM / WgmmaM;
    static constexpr int NumWgmmaN = TileN / WgmmaN;
    static constexpr int NumItersK = TileK / WgmmaK;

    static constexpr int SmemElemsA = TileM * TileK;
    static constexpr int SmemElemsB = TileK * TileN;
    static constexpr int StageElems = SmemElemsA + SmemElemsB;
};

__device__ __forceinline__ void mapThreadToEpilogue(int tid, int wgM, int wgN, int w, int &row,
                                                    int &col) {
    const int lane = tid % 32;
    const int warp = tid / 32;
    row = wgM * 64 + warp * 16 + lane / 4;
    col = wgN * 64 + 16 * w + 2 * (tid % 4);
}

// ===================================
// 3. Collective-Level
// ===================================

template <int BlockM, int BlockK>
__host__ CUtensorMap *createTmaMapDevice(const half *src, int blocksHeight, int blocksWidth) {
    CUtensorMap hostMap;
    uint64_t shape[5] = {(uint64_t)BlockK * blocksWidth, (uint64_t)BlockM * blocksHeight, 1, 1, 1};
    uint64_t stride[5] = {sizeof(half), sizeof(half) * BlockK * blocksWidth, 0, 0, 0};
    uint32_t smemShape[5] = {uint32_t(BlockK), uint32_t(BlockM), 1, 1, 1};
    uint32_t smemStride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &hostMap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, (void *)src, shape, stride + 1, smemShape,
        smemStride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    assert(result == CUDA_SUCCESS);

    CUtensorMap *devMap;
    cudaMalloc(&devMap, sizeof(CUtensorMap));
    cudaMemcpy(devMap, &hostMap, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return devMap;
}

template <typename Shape> struct alignas(128) SingleBufferSmem {
    half A[Shape::SmemElemsA];
    half B[Shape::SmemElemsB];
};

__device__ __forceinline__ void tmaFetchAsync(half *dstA, half *dstB, const CUtensorMap *mapA,
                                              const CUtensorMap *mapB, int globalBlockIdxM,
                                              int globalBlockIdxN, int globalBlockKStep, int TileM,
                                              int TileN, int TileK, Barrier &barA, Barrier &barB) {
    cde::cp_async_bulk_tensor_2d_global_to_shared(dstA, mapA, globalBlockKStep * TileK,
                                                  globalBlockIdxM * TileM, barA);
    cde::cp_async_bulk_tensor_2d_global_to_shared(dstB, mapB, globalBlockKStep * TileK,
                                                  globalBlockIdxN * TileN, barB);
}

// ===================================
// 4. Kernel-Level
// ===================================

template <typename Shape>
__global__ void __launch_bounds__(Shape::NumThreads)
    gemm(int M, int N, int K, half *__restrict__ C, const CUtensorMap *__restrict__ tmaA,
             const CUtensorMap *__restrict__ tmaB) {

    __shared__ SingleBufferSmem<Shape> smem;
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ Barrier barA, barB;

    if (threadIdx.x == 0) {
        init(&barA, blockDim.x);
        init(&barB, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    float acc[Shape::NumWgmmaM][Shape::NumWgmmaN][32] = {0};

    const int numBlocksK = K / Shape::TileK;
    const int bIdxN = blockIdx.x % (N / Shape::TileN);
    const int bIdxM = blockIdx.x / (N / Shape::TileN);

    Barrier::arrival_token tokenA, tokenB;

    for (int kStep = 0; kStep < numBlocksK; ++kStep) {
        if (threadIdx.x == 0) {
            tmaFetchAsync(smem.A, smem.B, tmaA, tmaB, bIdxM, bIdxN, kStep, Shape::TileM,
                          Shape::TileN, Shape::TileK, barA, barB);
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(smem.A));
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(smem.B));
        } else {
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }
        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        __syncthreads();

        wgmmaFence();
#pragma unroll
        for (int mi = 0; mi < Shape::NumWgmmaM; ++mi) {
            half *ptrA = &smem.A[0];
            wgmmaM64N64K16(acc[mi][0], &ptrA[0], &smem.B[0]);
            wgmmaM64N64K16(acc[mi][0], &ptrA[Shape::WgmmaK], &smem.B[Shape::WgmmaK]);
            wgmmaM64N64K16(acc[mi][0], &ptrA[2 * Shape::WgmmaK], &smem.B[2 * Shape::WgmmaK]);
            wgmmaM64N64K16(acc[mi][0], &ptrA[3 * Shape::WgmmaK], &smem.B[3 * Shape::WgmmaK]);
        }
        wgmmaCommit();
        wgmmaWait<0>();
    }

    half *blockC = C + (bIdxN * Shape::TileN * M) + (bIdxM * Shape::TileM);
    for (int mi = 0; mi < Shape::NumWgmmaM; ++mi) {
        for (int ni = 0; ni < Shape::NumWgmmaN; ++ni) {
            for (int w = 0; w < Shape::WgmmaN / 16; ++w) {
                int row, col;
                mapThreadToEpilogue(threadIdx.x, mi, ni, w, row, col);

                float *frag = &acc[mi][ni][w * 8];
                auto storeC = [&](int rOff, int cOff, float val) {
                    blockC[(col + cOff) * M + (row + rOff)] =
                        __float2half(val);
                };

                storeC(0, 0, frag[0]);
                storeC(0, 1, frag[1]);
                storeC(8, 0, frag[2]);
                storeC(8, 1, frag[3]);
                storeC(0, 8, frag[4]);
                storeC(0, 9, frag[5]);
                storeC(8, 8, frag[6]);
                storeC(8, 9, frag[7]);
            }
        }
    }
}


// ===================================
// 5. Device-Level
// ===================================

static CUtensorMap *globalTmaMapABase = nullptr;
static CUtensorMap *globalTmaMapBBase = nullptr;
static int prevMBase = 0, prevNBase = 0, prevKBase = 0;

static CUtensorMap *globalTmaMapAOpt = nullptr;
static CUtensorMap *globalTmaMapBOpt = nullptr;
static int prevMOpt = 0, prevNOpt = 0, prevKOpt = 0;

static void launchKernel(half *A, half *B, half *C, int M, int N, int K) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int WGMMA_M = 64;
    constexpr int WGMMA_N = 64;
    constexpr int WGMMA_K = 16;
    constexpr int NUM_THREADS = 128;
    using Shape = BlockShape<BM, BN, BK, WGMMA_M, WGMMA_N, WGMMA_K, NUM_THREADS>;

    if (!globalTmaMapABase || M != prevMBase || N != prevNBase || K != prevKBase) {
        if (globalTmaMapABase)
            cudaFree(globalTmaMapABase);
        if (globalTmaMapBBase)
            cudaFree(globalTmaMapBBase);
        globalTmaMapABase = createTmaMapDevice<Shape::TileM, Shape::TileK>(
            A, M / Shape::TileM, K / Shape::TileK);
        globalTmaMapBBase = createTmaMapDevice<Shape::TileN, Shape::TileK>(
            B, N / Shape::TileN, K / Shape::TileK);
        prevMBase = M;
        prevNBase = N;
        prevKBase = K;
    }

    dim3 grid((M / Shape::TileM) * (N / Shape::TileN));
    dim3 block(Shape::NumThreads);
    gemm<Shape><<<grid, block>>>(M, N, K, C, globalTmaMapABase, globalTmaMapBBase);
}

} // namespace gemm_8192x8192_v0

using gemm_8192x8192_v0::launchKernel;