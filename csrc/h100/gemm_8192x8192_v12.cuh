namespace gemm_8192x8192_v12 {

using Barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
using half = __half;

// ===================================
// 1. Atom-Level
// ===================================

__device__ __forceinline__ uint32_t smemPtrU32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbarrierInit(uint64_t* bar, uint32_t count) {
    uint32_t ptr = smemPtrU32(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(ptr), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarrierArrive(uint64_t* bar) {
    uint32_t ptr = smemPtrU32(bar);
    asm volatile("{\n"
                 ".reg .b64 t;\n"
                 "mbarrier.arrive.shared::cta.b64 t, [%0];\n"
                 "}\n"
                 :
                 : "r"(ptr)
                 : "memory");
}

__device__ __forceinline__ void mbarrierExpectTx(uint64_t* bar, uint32_t bytes) {
    uint32_t ptr = smemPtrU32(bar);
    asm volatile("{\n"
                 ".reg .b64 t;\n"
                 "mbarrier.arrive.expect_tx.shared::cta.b64 t, [%0], %1;\n"
                 "}\n"
                 :
                 : "r"(ptr), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void mbarrierWait(uint64_t* bar, int phaseBit) {
    uint32_t ptr = smemPtrU32(bar);
    asm volatile("{\n"
                 ".reg .pred P1;\n"
                 "LAB_WAIT:\n"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n"
                 "@P1 bra.uni DONE;\n"
                 "bra.uni LAB_WAIT;\n"
                 "DONE:\n"
                 "}\n"
                 :
                 : "r"(ptr), "r"(phaseBit)
                 : "memory");
}

__device__ __forceinline__ uint64_t matrixDescriptorEncode(uint64_t x) {
    return ((x & 0x3FFFF) >> 0x4);
}

__device__ __forceinline__ uint64_t makeSmemDesc(half* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrixDescriptorEncode(addr);
    desc |= matrixDescriptorEncode(static_cast<uint64_t>(16)) << 16;
    desc |= matrixDescriptorEncode(static_cast<uint64_t>(1024)) << 32;
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
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" : : "n"(N) : "memory");
}

template <uint32_t RegCount>
__device__ __forceinline__ void warpgroupRegAlloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount) : "memory");
}

template <uint32_t RegCount>
__device__ __forceinline__ void warpgroupRegDealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount) : "memory");
}

__device__ __forceinline__ void tmaLoad2D(half* smemPtr, const CUtensorMap* map, int coordK,
                                          int coordMn, uint64_t* bar) {
    uint32_t smemU32 = smemPtrU32(smemPtr);
    uint32_t barU32 = smemPtrU32(bar);
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :
                 : "r"(smemU32), "l"(map), "r"(coordK), "r"(coordMn), "r"(barU32)
                 : "memory");
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
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
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
          "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
          "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
          "+f"(d[3][6]), "+f"(d[3][7]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
          "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]), "+f"(d[5][0]), "+f"(d[5][1]),
          "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]),
          "+f"(d[6][6]), "+f"(d[6][7]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
          "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]), "+f"(d[8][0]), "+f"(d[8][1]),
          "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
          "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[9][4]), "+f"(d[9][5]),
          "+f"(d[9][6]), "+f"(d[9][7]), "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]),
          "+f"(d[10][3]), "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
          "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]), "+f"(d[11][4]),
          "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]), "+f"(d[12][0]), "+f"(d[12][1]),
          "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[12][4]), "+f"(d[12][5]), "+f"(d[12][6]),
          "+f"(d[12][7]), "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]),
          "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]), "+f"(d[14][0]),
          "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[14][4]), "+f"(d[14][5]),
          "+f"(d[14][6]), "+f"(d[14][7]), "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]),
          "+f"(d[15][3]), "+f"(d[15][4]), "+f"(d[15][5]), "+f"(d[15][6]), "+f"(d[15][7])
        : "l"(descA), "l"(descB), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
          "n"(int32_t(TransA)), "n"(int32_t(TransB)));
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

template <int BM, int BN, int TM = 4, int TN = 4>
struct Schedule {
    int block;
    int it;
    int totalBlocksM;
    int totalBlocksN;
    int numSm;

    __device__ __forceinline__ Schedule(int M, int N, int blockIdxValue)
        : block(blockIdxValue), it(0) {
        totalBlocksM = M / BM;
        totalBlocksN = N / BN;
        numSm = static_cast<int>(gridDim.x);
    }

    __device__ __forceinline__ int next() {
        int total = totalBlocksM * totalBlocksN;
        if (total <= 0) return -1;

        constexpr int K_TM = (TM > 0 ? TM : 1);
        constexpr int K_TN = (TN > 0 ? TN : 1);

        int tilesM = (totalBlocksM + K_TM - 1) / K_TM;
        int tilesN = (totalBlocksN + K_TN - 1) / K_TN;
        int tilesTotal = tilesM * tilesN;
        int virtualTotal = tilesTotal * K_TM * K_TN;

        int num = it * numSm + block;
        if (num >= virtualTotal) return -1;

        while (num < virtualTotal) {
            int curTile = num / (K_TM * K_TN);
            int curPos = num - curTile * (K_TM * K_TN);

            int tileM = curTile / tilesN;
            int tileN = curTile - tileM * tilesN;

            int m0 = tileM * K_TM;
            int n0 = tileN * K_TN;

            int dm = curPos / K_TN;
            int dn = curPos - dm * K_TN;

            int m = m0 + dm;
            int n = n0 + dn;

            ++it;

            if (m < totalBlocksM && n < totalBlocksN) {
                return m * totalBlocksN + n;
            }

            num = it * numSm + block;
        }

        return -1;
    }
};

// ===================================
// 3. Collective-Level
// ===================================

template <int BlockMajorSize, int BlockMinorSize>
__host__ void createTensorMap(CUtensorMap* tmaMap, half* gmemPtr, int blocksHeight, int blocksWidth) {
    void* gmemAddress = reinterpret_cast<void*>(gmemPtr);
    uint64_t gmemProbShape[5] = {static_cast<uint64_t>(BlockMinorSize) * blocksWidth,
                                 static_cast<uint64_t>(BlockMajorSize) * blocksHeight, 1, 1, 1};
    uint64_t gmemProbStride[5] = {sizeof(half), sizeof(half) * BlockMinorSize * blocksWidth, 0, 0, 0};
    uint32_t smemBoxShape[5] = {static_cast<uint32_t>(BlockMinorSize),
                                static_cast<uint32_t>(BlockMajorSize), 1, 1, 1};
    uint32_t smemBoxStride[5] = {1, 1, 1, 1, 1};

    cuTensorMapEncodeTiled(tmaMap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, gmemAddress, gmemProbShape,
                           gmemProbStride + 1, smemBoxShape, smemBoxStride,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                           CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

template <int BlockMajorSize, int BlockMinorSize>
__host__ CUtensorMap createTensorMapValue(half* gmemPtr, int blocksHeight, int blocksWidth) {
    CUtensorMap hostMap{};
    createTensorMap<BlockMajorSize, BlockMinorSize>(&hostMap, gmemPtr, blocksHeight, blocksWidth);
    return hostMap;
}

// ===================================
// 4. Kernel-Level
// ===================================

template <typename Shape>
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
    
    constexpr int NUM_CONSUMERS = (NUM_THREADS / 128) - 1;
    constexpr int NUM_CONSUMER_THREADS = NUM_CONSUMERS * 128;
    constexpr int B_WG_M = BM / NUM_CONSUMERS;

    static_assert((NUM_THREADS % 128) == 0, "NUM_THREADS must be multiple of 128");
    static_assert(NUM_CONSUMERS >= 1, "numConsumers must be >= 1");
    static_assert((BM % NUM_CONSUMERS) == 0, "BM must be divisible by NUM_CONSUMERS");
    static_assert((B_WG_M % WGMMA_M) == 0, "B_WG_M must be divisible by WGMMA_M");

    const int wgIdx = threadIdx.x / 128;
    const int tid128 = threadIdx.x & 127;

    if (wgIdx == 0) {
        constexpr int NUM_REGS = (NUM_CONSUMERS <= 2 ? 24 : 32);
        warpgroupRegDealloc<NUM_REGS>();
    } else {
        constexpr int NUM_REGS = (NUM_CONSUMERS == 1 ? 256 : (NUM_CONSUMERS == 2 ? 240 : 160));
        warpgroupRegAlloc<NUM_REGS>();
    }

    extern __shared__ __align__(128) unsigned char smemRaw[];
    auto* shared = reinterpret_cast<SharedStorage<BM, BN, BK, QSIZE>*>(smemRaw);

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ alignas(8) uint64_t full[QSIZE];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ alignas(8) uint64_t empty[QSIZE];

    Schedule<BM, BN, 4, 4> schedule(M, N, blockIdx.x);

    for (int numBlock = schedule.next(); numBlock >= 0; numBlock = schedule.next()) {
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int q = 0; q < QSIZE; ++q) {
                mbarrierInit(&full[q], 1);
                mbarrierInit(&empty[q], NUM_CONSUMER_THREADS);
            }
            cde::fence_proxy_async_shared_cta();
        }
        __syncthreads();

        const int NUM_BLOCKS_K = K / BK;
        const int NUM_BLOCK_N = numBlock % (N / BN);
        const int NUM_BLOCK_M = numBlock / (N / BN);

        if (wgIdx == 0) {
            int pEmpty[QSIZE];
            int pFull[QSIZE];

            #pragma unroll
            for (int q = 0; q < QSIZE; ++q) {
                pEmpty[q] = 0;
                pFull[q] = 0;
            }

            if (tid128 == 0) {
                for (int kIter = 0; kIter < NUM_BLOCKS_K; ++kIter) {
                    int q = kIter % QSIZE;

                    mbarrierWait(&empty[q], pEmpty[q]);
                    pEmpty[q] ^= 1;

                    half* sA = &shared->A[q * (BM * BK)];
                    half* sB = &shared->B[q * (BK * BN)];

                    tmaLoad2D(sA, &tensorMapA, kIter * BK, NUM_BLOCK_M * BM, &full[q]);
                    tmaLoad2D(sB, &tensorMapB, kIter * BK, NUM_BLOCK_N * BN, &full[q]);

                    uint32_t bytes = (BM * BK + BK * BN) * sizeof(half);
                    mbarrierExpectTx(&full[q], bytes);
                    pFull[q] ^= 1;
                }
            }
        } else {
            const int activeConsumerIdx = wgIdx - 1;
            int pEmpty[QSIZE];
            int pFull[QSIZE];

            #pragma unroll
            for (int q = 0; q < QSIZE; ++q) {
                pEmpty[q] = 0;
                pFull[q] = 0;
            }

            float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8] = {0.0f};

            #pragma unroll
            for (int q = 0; q < QSIZE; ++q) {
                mbarrierArrive(&empty[q]);
                pEmpty[q] ^= 1;
            }

            for (int kIter = 0; kIter < NUM_BLOCKS_K; ++kIter) {
                int q = kIter % QSIZE;

                mbarrierWait(&full[q], pFull[q]);
                pFull[q] ^= 1;

                half* sAPtr = &shared->A[q * (BM * BK)];
                half* sBPtr = &shared->B[q * (BK * BN)];

                warpgroupArrive();

                #pragma unroll
                for (int mIt = 0; mIt < B_WG_M / WGMMA_M; ++mIt) {
                    half* wgmmaSA = sAPtr + (activeConsumerIdx * B_WG_M + mIt * WGMMA_M) * BK;

                    #pragma unroll
                    for (int kStep = 0; kStep < BK / 16; ++kStep) {
                        wgmmaM64N256K16<1, 1, 1, 0, 0>(
                            d[mIt],
                            wgmmaSA + kStep * 16,
                            sBPtr + kStep * 16
                        );
                    }
                }

                warpgroupCommitBatch();
                warpgroupWait<0>();

                mbarrierArrive(&empty[q]);
                pEmpty[q] ^= 1;
            }

            const int lane = tid128 & 31;
            const int wgWarp = tid128 >> 5;
            const int rowInTileLocal = wgWarp * 16 + (lane >> 2);
            const int rowBase = activeConsumerIdx * B_WG_M;
            half* cBase = C + (NUM_BLOCK_N * BN) * M + (NUM_BLOCK_M * BM);

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

CUtensorMap globalTmaMapA;
CUtensorMap globalTmaMapB;
bool tmaInitialized = false;
int prevM = 0;
int prevN = 0;
int prevK = 0;

void launchKernel(half* A, half* B, half* C, int M, int N, int K) {
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 64;
    constexpr int WGMMA_M = 64;
    constexpr int WGMMA_N = 256;
    constexpr int WGMMA_K = 16;
    constexpr int QSIZE = 3;
    constexpr int NUM_THREADS = 384;
    constexpr int NUM_SM = 128;

    using Shape = BlockShape<BM, BN, BK, QSIZE, WGMMA_M, WGMMA_N, WGMMA_K, NUM_THREADS>;

    if (!tmaInitialized || M != prevM || N != prevN || K != prevK) {
        globalTmaMapA = createTensorMapValue<BM, BK>(A, M / BM, K / BK);
        globalTmaMapB = createTensorMapValue<BN, BK>(B, N / BN, K / BK);
        tmaInitialized = true;
        prevM = M;
        prevN = N;
        prevK = K;
    }

    size_t sharedBytes = sizeof(SharedStorage<BM, BN, BK, QSIZE>);
    auto kernelPtr = &matmulKernel<Shape>;
    cudaFuncSetAttribute(kernelPtr, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sharedBytes));
    
    kernelPtr<<<NUM_SM, NUM_THREADS, sharedBytes>>>(M, N, K, C, globalTmaMapA, globalTmaMapB);
}

} // namespace gemm_8192x8192_v12

using gemm_8192x8192_v12::launchKernel;