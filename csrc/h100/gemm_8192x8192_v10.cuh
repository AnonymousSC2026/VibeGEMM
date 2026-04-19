namespace gemm_8192x8192_v10 {

using Barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// ===================================
// 1. Atom-Level
// ===================================

__device__ __forceinline__ uint64_t matrixDescriptorEncode(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

__device__ __forceinline__ uint64_t makeSmemDesc(half* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrixDescriptorEncode(addr);
    desc |= matrixDescriptorEncode((uint64_t)16) << 16;
    desc |= matrixDescriptorEncode((uint64_t)1024) << 32;
    desc |= 1llu << 62;
    return desc;
}

__device__ __forceinline__ void warpgroupArrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroupCommitBatch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void warpgroupWait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmmaM64N256K16(float d[16][8], half* sA, half* sB) {
    uint64_t descA = makeSmemDesc(&sA[0]);
    uint64_t descB = makeSmemDesc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
        " %104, %105, %106, %107, %108, %109, %110, %111,  "
        " %112, %113, %114, %115, %116, %117, %118, %119,  "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130,    %131,  %132,  %133,  %134;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
            "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
            "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
            "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
            "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]), "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]),
            "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[12][4]), "+f"(d[12][5]), "+f"(d[12][6]), "+f"(d[12][7]),
            "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]), "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]),
            "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[14][4]), "+f"(d[14][5]), "+f"(d[14][6]), "+f"(d[14][7]),
            "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]), "+f"(d[15][4]), "+f"(d[15][5]), "+f"(d[15][6]), "+f"(d[15][7])
        : "l"(descA), "l"(descB), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template <uint32_t RegCount>
__device__ __forceinline__ void warpgroupRegAlloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount) : "memory");
}

template <uint32_t RegCount>
__device__ __forceinline__ void warpgroupRegDealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount) : "memory");
}

// ===================================
// 2. Tile-Level
// ===================================

template <int BM, int BN, int BK, int QSIZE, int WGMMA_M, int WGMMA_N, int WGMMA_K, int NUM_THREADS>
struct BlockShape {
    static constexpr int TileM = BM;
    static constexpr int TileN = BN;
    static constexpr int TileK = BK;
    static constexpr int QueueSize = QSIZE;
    static constexpr int WgmmaM = WGMMA_M;
    static constexpr int WgmmaN = WGMMA_N;
    static constexpr int WgmmaK = WGMMA_K;
    static constexpr int NumThreads = NUM_THREADS;
};

template <int BM, int BN, int BK, int QSIZE>
struct SharedStorage {
    alignas(128) half A[BM * BK * QSIZE];
    alignas(128) half B[BK * BN * QSIZE];
};

template<int BM, int BN>
struct Schedule {
    int block, it;
    int totalBlocksM, totalBlocksN;
    int numSm;

    __device__ __forceinline__ Schedule(int M, int N, int _block)
        : block(_block), it(0)
    {
        totalBlocksM = M / BM;
        totalBlocksN = N / BN;
        numSm = (int)gridDim.x;
    }

    __device__ __forceinline__ int next() {
        int idx = it * numSm + block;
        if (idx >= totalBlocksM * totalBlocksN) return -1;
        it++;
        return idx;
    }
};

// ===================================
// 3. Collective-Level
// ===================================

template <int BlockMajorSize, int BlockMinorSize>
__host__ void createTensorMap(CUtensorMap *tmaMap, half* gmemPtr, int blocksHeight, int blocksWidth) {
    void* gmemAddress = (void*)gmemPtr;
    uint64_t gmemProbShape[5]  = {(uint64_t)BlockMinorSize * blocksWidth, (uint64_t)BlockMajorSize * blocksHeight, 1, 1, 1};
    uint64_t gmemProbStride[5] = {sizeof(half), sizeof(half) * BlockMinorSize * blocksWidth, 0, 0, 0};
    uint32_t smemBoxShape[5]   = {uint32_t(BlockMinorSize), uint32_t(BlockMajorSize), 1, 1, 1};
    uint32_t smemBoxStride[5]  = {1, 1, 1, 1, 1};
    cuTensorMapEncodeTiled(tmaMap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, gmemAddress, gmemProbShape,
        gmemProbStride + 1, smemBoxShape, smemBoxStride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

template <int BlockMajorSize, int BlockMinorSize>
__host__ CUtensorMap createTensorMapValue(half* src, int blocksHeight, int blocksWidth) {
    CUtensorMap tmaHost{};
    createTensorMap<BlockMajorSize, BlockMinorSize>(&tmaHost, src, blocksHeight, blocksWidth);
    return tmaHost;
}

// ===================================
// 4. Kernel-Level
// ===================================

template<typename Shape>
__global__ void __launch_bounds__(Shape::NumThreads)
matmulKernel(
    int M, int N, int K, half* __restrict__ C,
    const __grid_constant__ CUtensorMap tensorMapA,
    const __grid_constant__ CUtensorMap tensorMapB
) {
    constexpr int BM = Shape::TileM;
    constexpr int BN = Shape::TileN;
    constexpr int BK = Shape::TileK;
    constexpr int QSIZE = Shape::QueueSize;
    constexpr int WGMMA_M = Shape::WgmmaM;
    constexpr int WGMMA_N = Shape::WgmmaN;
    constexpr int NUM_THREADS = Shape::NumThreads;

    const int wgIdx = threadIdx.x / 128;
    const int tid128 = threadIdx.x & 127;

    constexpr int numConsumers = (NUM_THREADS / 128) - 1;
    static_assert((NUM_THREADS % 128) == 0, "NUM_THREADS must be multiple of 128");
    static_assert(numConsumers >= 1, "num_consumers must be >= 1");
    static_assert((BM % numConsumers) == 0, "BM must be divisible by numConsumers");
    constexpr int B_WG_M = BM / numConsumers;
    static_assert((B_WG_M % WGMMA_M) == 0, "B_WG_M must be divisible by WGMMA_M");

    constexpr int NUM_CONSUMER_THREADS = numConsumers * 128;

    if (wgIdx == 0) {
        constexpr int numRegs = (numConsumers <= 2 ? 24 : 32);
        warpgroupRegDealloc<numRegs>();
    } else {
        constexpr int numRegs = (numConsumers == 1 ? 256 : (numConsumers == 2 ? 240 : 160));
        warpgroupRegAlloc<numRegs>();
    }

    extern __shared__ __align__(128) unsigned char smemRaw[];
    auto* smem = reinterpret_cast<SharedStorage<BM, BN, BK, QSIZE>*>(smemRaw);

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ Barrier full[QSIZE];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ Barrier empty[QSIZE];

    Schedule<BM, BN> schedule(M, N, blockIdx.x);

    for (int numBlock = schedule.next(); numBlock >= 0; numBlock = schedule.next()) {

        if (threadIdx.x == 0) {
            #pragma unroll
            for (int q = 0; q < QSIZE; ++q) {
                init(&full[q], NUM_CONSUMER_THREADS + 1);
                init(&empty[q], NUM_CONSUMER_THREADS + 1);
            }
            cde::fence_proxy_async_shared_cta();
        }
        __syncthreads();

        const int numBlocksK = K / BK;
        const int numBlockN = numBlock % (N / BN);
        const int numBlockM = numBlock / (N / BN);

        if (wgIdx == 0) {
            if (tid128 == 0) {
                for (int kIter = 0; kIter < numBlocksK; ++kIter) {
                    int q = kIter % QSIZE;

                    empty[q].wait(empty[q].arrive());

                    half* sA = &smem->A[q * (BM * BK)];
                    half* sB = &smem->B[q * (BK * BN)];

                    cde::cp_async_bulk_tensor_2d_global_to_shared(
                        sA, &tensorMapA, kIter * BK, numBlockM * BM, full[q]);
                    cde::cp_async_bulk_tensor_2d_global_to_shared(
                        sB, &tensorMapB, kIter * BK, numBlockN * BN, full[q]);

                    uint32_t bytes = (BM * BK + BK * BN) * sizeof(half);
                    cuda::device::barrier_arrive_tx(full[q], 1, bytes);
                }
            }
        } else {
            const int activeConsumerIdx = wgIdx - 1;

            float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8] = {0.0f};

            #pragma unroll
            for (int q = 0; q < QSIZE; ++q) (void)empty[q].arrive();

            for (int kIter = 0; kIter < numBlocksK; ++kIter) {
                int q = kIter % QSIZE;

                full[q].wait(full[q].arrive());

                half* sAPtr = &smem->A[q * (BM * BK)];
                half* sBPtr = &smem->B[q * (BK * BN)];

                warpgroupArrive();

                #pragma unroll
                for (int mIt = 0; mIt < B_WG_M / WGMMA_M; ++mIt) {
                    half* wgmmaSa = sAPtr + (activeConsumerIdx * B_WG_M + mIt * WGMMA_M) * BK;

                    #pragma unroll
                    for (int kStep = 0; kStep < BK / 16; ++kStep) {
                        wgmmaM64N256K16<1, 1, 1, 0, 0>(
                            d[mIt],
                            wgmmaSa + kStep * 16,
                            sBPtr + kStep * 16
                        );
                    }
                }

                warpgroupCommitBatch();
                warpgroupWait<0>();

                (void)empty[q].arrive();
            }

            const int lane = tid128 & 31;
            const int wgWarp = tid128 >> 5;
            const int rowInTileLocal = wgWarp * 16 + (lane >> 2);

            const int rowBase = activeConsumerIdx * B_WG_M;
            half* cBase = C + (numBlockN * BN) * M + (numBlockM * BM);

            auto storeC = [&](int i, int j, float val) {
                cBase[j * M + i] = (half)val;
            };

            #pragma unroll
            for (int mIt = 0; mIt < B_WG_M / WGMMA_M; ++mIt) {
                #pragma unroll
                for (int nIt = 0; nIt < BN / WGMMA_N; ++nIt) {
                    #pragma unroll
                    for (int w = 0; w < WGMMA_N / 16; ++w) {
                        const int colInTile = 16 * w + 2 * (lane & 3);

                        const int i0 = rowBase + mIt * WGMMA_M + rowInTileLocal;
                        const int j0 = nIt * WGMMA_N + colInTile;

                        storeC(i0, j0, d[mIt][w][0]);
                        storeC(i0, j0 + 1, d[mIt][w][1]);
                        storeC(i0 + 8, j0, d[mIt][w][2]);
                        storeC(i0 + 8, j0 + 1, d[mIt][w][3]);
                        storeC(i0, j0 + 8, d[mIt][w][4]);
                        storeC(i0, j0 + 9, d[mIt][w][5]);
                        storeC(i0 + 8, j0 + 8, d[mIt][w][6]);
                        storeC(i0 + 8, j0 + 9, d[mIt][w][7]);
                    }
                }
            }
        }

        __syncthreads();
    }
}

// ===================================
// 5. Device-Level
// ===================================

void launchKernel(half* A, half* B, half* C, int M, int N, int K) {
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 64;
    constexpr int WGMMA_M = 64;
    constexpr int WGMMA_N = 256;
    constexpr int WGMMA_K = 16;
    constexpr int QSIZE = 3;
    constexpr int NUM_THREADS = 384;

    using Shape = BlockShape<BM, BN, BK, QSIZE, WGMMA_M, WGMMA_N, WGMMA_K, NUM_THREADS>;

    CUtensorMap tmaA = createTensorMapValue<BM, BK>(A, M / BM, K / BK);
    CUtensorMap tmaB = createTensorMapValue<BN, BK>(B, N / BN, K / BK);

    size_t sharedBytes = sizeof(SharedStorage<BM, BN, BK, QSIZE>);
    auto kptr = &matmulKernel<Shape>;
    cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sharedBytes);

    int numSm = 0;
    cudaDeviceGetAttribute(&numSm, cudaDevAttrMultiProcessorCount, 0);

    kptr<<<numSm, NUM_THREADS, sharedBytes>>>(M, N, K, C, tmaA, tmaB);
}

}

using gemm_8192x8192_v10::launchKernel;