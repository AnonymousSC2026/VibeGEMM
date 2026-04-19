namespace gemm_8192x8192_v20 {

namespace cde = cuda::device::experimental;

// ===================================
// 1. Atom-Level
// ===================================

__device__ __forceinline__ uint32_t smemPtrU32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbarrierInit(uint64_t* bar, int threadCount, int transactionCount) {
    uint32_t barPtr = smemPtrU32(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :: "r"(barPtr), "r"(threadCount + transactionCount));
}

__device__ __forceinline__ void mbarrierInit(uint64_t* bar, uint32_t count) {
    uint32_t ptr = smemPtrU32(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 : : "r"(ptr), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarrierArrive(uint64_t* bar) {
    uint32_t ptr = smemPtrU32(bar);
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
                 : : "r"(ptr) : "memory");
}

__device__ __forceinline__ void mbarrierExpectTx(uint64_t* bar, uint32_t bytes) {
    uint32_t ptr = smemPtrU32(bar);
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
                 : : "r"(ptr), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarrierWait(uint64_t* bar, int phaseBit) {
    uint32_t ptr = smemPtrU32(bar);
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1 bra.uni DONE;\n"
        "bra.uni LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        : : "r"(ptr), "r"(phaseBit) : "memory"
    );
}

__device__ __forceinline__ void arriveCluster(uint64_t* bar, uint32_t ctaId, uint32_t count = 1) {
    uint32_t smemAddr = smemPtrU32(bar);
    asm volatile(
        "{\n\t"
        ".reg .b32 remAddr32;\n\t"
        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
        "mbarrier.arrive.shared::cluster.b64 _, [remAddr32], %2;\n\t"
        "}"
        : : "r"(smemAddr), "r"(ctaId), "r"(count)
    );
}

__device__ __forceinline__ void loadAsyncMulticast(
    half* dst, void const* const srcTma, uint64_t* bar,
    int globalColIdx, int globalRowIdx, uint16_t clusterMask) {
    uint64_t tmaPtr = reinterpret_cast<uint64_t>(srcTma);
    uint32_t mbarPtr = smemPtrU32(bar);
    uint32_t dstPtr  = smemPtrU32(dst);
    asm volatile (
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster "
        "[%0], [%1, {%3, %4, %5}], [%2], %6;"
        : : "r"(dstPtr), "l"(tmaPtr), "r"(mbarPtr), "n"(0), "r"(globalRowIdx), "r"(globalColIdx / 64), "h"(clusterMask)
        : "memory"
    );
}

__device__ __forceinline__ void loadAsync(
    half* dst, void const* const srcTma, uint64_t* bar,
    int globalColIdx, int globalRowIdx) {
    uint64_t tmaPtr = reinterpret_cast<uint64_t>(srcTma);
    uint32_t mbarPtr = smemPtrU32(bar);
    uint32_t dstPtr  = smemPtrU32(dst);
    asm volatile (
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4, %5}], [%2];"
        : : "r"(dstPtr), "l"(tmaPtr), "r"(mbarPtr), "n"(0), "r"(globalRowIdx), "r"(globalColIdx / 64)
        : "memory"
    );
}

__device__ __forceinline__ void storeAsync(
    void const* dstTmaMap, half* src, int globalColIdx, int globalRowIdx) {
    uint64_t tmaPtr = reinterpret_cast<uint64_t>(dstTmaMap);
    uint32_t srcPtr = smemPtrU32(src);
    asm volatile (
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group "
        "[%0, {%2, %3, %4}], [%1];"
        : : "l"(tmaPtr), "r"(srcPtr), "n"(0), "r"(globalRowIdx), "r"(globalColIdx / 64)
        : "memory"
    );
}

__device__ __forceinline__ uint64_t matrixDescriptorEncode(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

__device__ __forceinline__ uint64_t makeSmemDesc(half* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0;
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
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        "%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,"
        "%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,"
        "%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,"
        "%56,%57,%58,%59,%60,%61,%62,%63,"
        "%64,%65,%66,%67,%68,%69,%70,%71,"
        "%72,%73,%74,%75,%76,%77,%78,%79,"
        "%80,%81,%82,%83,%84,%85,%86,%87,"
        "%88,%89,%90,%91,%92,%93,%94,%95,"
        "%96,%97,%98,%99,%100,%101,%102,%103,"
        "%104,%105,%106,%107,%108,%109,%110,%111,"
        "%112,%113,%114,%115,%116,%117,%118,%119,"
        "%120,%121,%122,%123,%124,%125,%126,%127},"
        "%128,%129,%130,%131,%132,%133,%134;\n"
        "}\n"
        : "+f"(d[0][0]),"+f"(d[0][1]),"+f"(d[0][2]),"+f"(d[0][3]),"+f"(d[0][4]),"+f"(d[0][5]),"+f"(d[0][6]),"+f"(d[0][7]),
          "+f"(d[1][0]),"+f"(d[1][1]),"+f"(d[1][2]),"+f"(d[1][3]),"+f"(d[1][4]),"+f"(d[1][5]),"+f"(d[1][6]),"+f"(d[1][7]),
          "+f"(d[2][0]),"+f"(d[2][1]),"+f"(d[2][2]),"+f"(d[2][3]),"+f"(d[2][4]),"+f"(d[2][5]),"+f"(d[2][6]),"+f"(d[2][7]),
          "+f"(d[3][0]),"+f"(d[3][1]),"+f"(d[3][2]),"+f"(d[3][3]),"+f"(d[3][4]),"+f"(d[3][5]),"+f"(d[3][6]),"+f"(d[3][7]),
          "+f"(d[4][0]),"+f"(d[4][1]),"+f"(d[4][2]),"+f"(d[4][3]),"+f"(d[4][4]),"+f"(d[4][5]),"+f"(d[4][6]),"+f"(d[4][7]),
          "+f"(d[5][0]),"+f"(d[5][1]),"+f"(d[5][2]),"+f"(d[5][3]),"+f"(d[5][4]),"+f"(d[5][5]),"+f"(d[5][6]),"+f"(d[5][7]),
          "+f"(d[6][0]),"+f"(d[6][1]),"+f"(d[6][2]),"+f"(d[6][3]),"+f"(d[6][4]),"+f"(d[6][5]),"+f"(d[6][6]),"+f"(d[6][7]),
          "+f"(d[7][0]),"+f"(d[7][1]),"+f"(d[7][2]),"+f"(d[7][3]),"+f"(d[7][4]),"+f"(d[7][5]),"+f"(d[7][6]),"+f"(d[7][7]),
          "+f"(d[8][0]),"+f"(d[8][1]),"+f"(d[8][2]),"+f"(d[8][3]),"+f"(d[8][4]),"+f"(d[8][5]),"+f"(d[8][6]),"+f"(d[8][7]),
          "+f"(d[9][0]),"+f"(d[9][1]),"+f"(d[9][2]),"+f"(d[9][3]),"+f"(d[9][4]),"+f"(d[9][5]),"+f"(d[9][6]),"+f"(d[9][7]),
          "+f"(d[10][0]),"+f"(d[10][1]),"+f"(d[10][2]),"+f"(d[10][3]),"+f"(d[10][4]),"+f"(d[10][5]),"+f"(d[10][6]),"+f"(d[10][7]),
          "+f"(d[11][0]),"+f"(d[11][1]),"+f"(d[11][2]),"+f"(d[11][3]),"+f"(d[11][4]),"+f"(d[11][5]),"+f"(d[11][6]),"+f"(d[11][7]),
          "+f"(d[12][0]),"+f"(d[12][1]),"+f"(d[12][2]),"+f"(d[12][3]),"+f"(d[12][4]),"+f"(d[12][5]),"+f"(d[12][6]),"+f"(d[12][7]),
          "+f"(d[13][0]),"+f"(d[13][1]),"+f"(d[13][2]),"+f"(d[13][3]),"+f"(d[13][4]),"+f"(d[13][5]),"+f"(d[13][6]),"+f"(d[13][7]),
          "+f"(d[14][0]),"+f"(d[14][1]),"+f"(d[14][2]),"+f"(d[14][3]),"+f"(d[14][4]),"+f"(d[14][5]),"+f"(d[14][6]),"+f"(d[14][7]),
          "+f"(d[15][0]),"+f"(d[15][1]),"+f"(d[15][2]),"+f"(d[15][3]),"+f"(d[15][4]),"+f"(d[15][5]),"+f"(d[15][6]),"+f"(d[15][7])
        : "l"(descA), "l"(descB),
          "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
          "n"(int32_t(TransA)), "n"(int32_t(TransB)));
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

template <int BM, int BN, int BK, int QSIZE, int WGMMA_M, int WGMMA_N, int WGMMA_K, int NUM_THREADS, int CLUSTER_M, int CLUSTER_N>
struct BlockShape {
    static constexpr int TileM = BM;
    static constexpr int TileN = BN;
    static constexpr int TileK = BK;
    static constexpr int QueueSize = QSIZE;
    static constexpr int WgmmaM = WGMMA_M;
    static constexpr int WgmmaN = WGMMA_N;
    static constexpr int WgmmaK = WGMMA_K;
    static constexpr int NumThreads = NUM_THREADS;
    static constexpr int ClusterM = CLUSTER_M;
    static constexpr int ClusterN = CLUSTER_N;
    static constexpr int ClusterSize = CLUSTER_M * CLUSTER_N;
};

template <int BM, int BN, int BK, int QSIZE>
struct SharedStorage {
    alignas(128) half A[BM * BK * QSIZE];
    alignas(128) half B[BK * BN * QSIZE];
    alignas(128) half C[BN * BM];
    alignas(8) uint64_t full[QSIZE];
    alignas(8) uint64_t empty[QSIZE];
    alignas(16) int space[128];
};

struct HilbertSchedule {
    const int* space;
    int it;

    __device__ __forceinline__ HilbertSchedule(const int* _space) : space(_space), it(0) {}

    __device__ __forceinline__ bool next(int& blockM, int& blockN) {
        if (it >= 128) return false;
        int v = space[it++];
        if (v < 0) return false;
        blockM = (v >> 16) & 0xFFFF;
        blockN = v & 0xFFFF;
        return true;
    }
};

// ===================================
// 3. Collective-Level
// ===================================

template <int BlockMajorSize, int BlockMinorSize, bool Swizzle = true>
__host__ void createTensorMap(CUtensorMap* tmaMap, half* gmemPtr, int globalHeight, int globalWidth) {
    void* gmemAddress = (void*)gmemPtr;
    uint64_t gmemProbShape[5] = {64, (uint64_t)globalHeight, (uint64_t)(globalWidth / 64), 1, 1};
    uint64_t gmemProbStride[5] = {sizeof(half) * globalWidth, 64 * sizeof(half), 0, 0, 0};
    uint32_t smemBoxShape[5] = {64, (uint32_t)BlockMajorSize, (uint32_t)(BlockMinorSize / 64), 1, 1};
    uint32_t smemBoxStride[5] = {1, 1, 1, 1, 1};
    cuTensorMapEncodeTiled(tmaMap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, gmemAddress, gmemProbShape,
        gmemProbStride, smemBoxShape, smemBoxStride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        Swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

template <int BlockMajorSize, int BlockMinorSize, bool Swizzle = true>
__host__ CUtensorMap createTensorMapValue(half* src, int globalHeight, int globalWidth) {
    CUtensorMap tmaHost{};
    createTensorMap<BlockMajorSize, BlockMinorSize, Swizzle>(&tmaHost, src, globalHeight, globalWidth);
    return tmaHost;
}

// ===================================
// 4. Kernel-Level
// ===================================

template<typename Shape>
__global__ void __launch_bounds__(Shape::NumThreads)
__cluster_dims__(Shape::ClusterSize, 1, 1)
matmulKernel(
    int M, int N, int K,
    const __grid_constant__ CUtensorMap tensorMapC,
    const __grid_constant__ CUtensorMap tensorMapA,
    const __grid_constant__ CUtensorMap tensorMapB,
    const int* __restrict__ dspace
) {
    constexpr int BM = Shape::TileM;
    constexpr int BN = Shape::TileN;
    constexpr int BK = Shape::TileK;
    constexpr int QSIZE = Shape::QueueSize;
    constexpr int WGMMA_M = Shape::WgmmaM;
    constexpr int WGMMA_N = Shape::WgmmaN;
    constexpr int WGMMA_K = Shape::WgmmaK;
    constexpr int NUM_THREADS = Shape::NumThreads;

    constexpr int numConsumers = (NUM_THREADS / 128) - 1;
    constexpr int B_WG_M = BM / numConsumers;

    const int wgIdx = threadIdx.x / 128;
    const int tid = threadIdx.x % 128;

    extern __shared__ __align__(128) uint8_t smemRaw[];
    auto& smem = *reinterpret_cast<SharedStorage<BM, BN, BK, QSIZE>*>(smemRaw);
    half* sA = smem.A;
    half* sB = smem.B;
    half* sC = smem.C;
    uint64_t* full = smem.full;
    uint64_t* empty = smem.empty;

    uint32_t clusterId;
    asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(clusterId) :);

    const int numBlocksK = K / BK;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < QSIZE; ++i) {
            mbarrierInit(&full[i], 0, 1);
            mbarrierInit(&empty[i], 0, numConsumers * Shape::ClusterSize);
        }
    }

    if (threadIdx.x < 128) {
        smem.space[threadIdx.x] = dspace[clusterId * 128 + threadIdx.x];
    }

    asm volatile("barrier.cluster.arrive;\n" : :);
    asm volatile("barrier.cluster.wait;\n" : :);

    HilbertSchedule schedule(smem.space);

    uint32_t rank;
    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(rank) :);
    uint32_t rankM = rank / Shape::ClusterN;
    uint32_t rankN = rank % Shape::ClusterN;

    if (wgIdx == 0) {
        constexpr int numRegs = (numConsumers <= 2 ? 24 : 32);
        warpgroupRegDealloc<numRegs>();

        if (tid == 0) {
            int p = 0, qidx = 0;
            uint16_t bMask = 0;

            if constexpr (Shape::ClusterM > 1) {
                #pragma unroll
                for (int rm = 0; rm < Shape::ClusterM; ++rm) {
                    bMask |= (uint16_t)(1u << (rm * Shape::ClusterN + rankN));
                }
            }

            int numBlockM, numBlockN;
            while (schedule.next(numBlockM, numBlockN)) {
                numBlockN = numBlockN * Shape::ClusterN + rankN;
                numBlockM = numBlockM * Shape::ClusterM + rankM;
                
                for (int blockKIter = 0; blockKIter < numBlocksK; ++blockKIter, ++qidx) {
                    if (qidx == QSIZE) { qidx = 0; p ^= 1; }
                    mbarrierWait(&empty[qidx], p);
                    
                    mbarrierExpectTx(&full[qidx], (BK * BN + BK * BM) * sizeof(half));
                    
                    if constexpr (Shape::ClusterN > 1) {
                        uint32_t mask = ((1u << Shape::ClusterN) - 1u) << (rankM * Shape::ClusterN);
                        if (rankN == 0) {
                            loadAsyncMulticast(&sA[qidx * BK * BM], &tensorMapA, &full[qidx], blockKIter * BK, numBlockM * BM, (uint16_t)mask);
                        }
                    } else {
                        loadAsync(&sA[qidx * BK * BM], &tensorMapA, &full[qidx], blockKIter * BK, numBlockM * BM);
                    }

                    if constexpr (Shape::ClusterM > 1) {
                        if (rankM == 0) {
                            loadAsyncMulticast(&sB[qidx * BK * BN], &tensorMapB, &full[qidx], blockKIter * BK, numBlockN * BN, bMask);
                        }
                    } else {
                        loadAsync(&sB[qidx * BK * BN], &tensorMapB, &full[qidx], blockKIter * BK, numBlockN * BN);
                    }
                }
            }
        }
    } else {
        warpgroupRegAlloc<240>();

        const int activeConsumerIdx = wgIdx - 1;
        float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];

        #pragma unroll
        for (int qidx = 0; qidx < QSIZE; ++qidx) {
            if (tid < Shape::ClusterSize) arriveCluster(&empty[qidx], tid);
        }

        int p = 0, qidx = 0;
        int numBlockM, numBlockN;

        while (schedule.next(numBlockM, numBlockN)) {
            numBlockN = numBlockN * Shape::ClusterN + rankN;
            numBlockM = numBlockM * Shape::ClusterM + rankM;

            {
                if (qidx == QSIZE) { qidx = 0; p ^= 1; }
                mbarrierWait(&full[qidx], p);
                warpgroupArrive();

                #pragma unroll
                for (int mIt = 0; mIt < B_WG_M / WGMMA_M; ++mIt) {
                    half* wgmmaSa = sA + qidx * BK * BM + 64 * (mIt + activeConsumerIdx * (B_WG_M / WGMMA_M)) * WGMMA_M;
                    half* wgmmaSb = sB + qidx * BK * BN;
                    {
                        wgmmaM64N256K16<0, 1, 1, 0, 0>(d[mIt], &wgmmaSa[0], &wgmmaSb[0]);
                        #pragma unroll
                        for (int kIt = 1; kIt < 64 / WGMMA_K; ++kIt) {
                            wgmmaM64N256K16<1, 1, 1, 0, 0>(d[mIt], &wgmmaSa[kIt * WGMMA_K], &wgmmaSb[kIt * WGMMA_K]);
                        }
                        wgmmaSa += 64 * BM;
                        wgmmaSb += 64 * BN;
                    }
                    #pragma unroll
                    for (int bk = 64; bk < BK; bk += 64) {
                        #pragma unroll
                        for (int kIt = 0; kIt < 64 / WGMMA_K; ++kIt) {
                            wgmmaM64N256K16<1, 1, 1, 0, 0>(d[mIt], &wgmmaSa[kIt * WGMMA_K], &wgmmaSb[kIt * WGMMA_K]);
                        }
                        wgmmaSa += 64 * BM;
                        wgmmaSb += 64 * BN;
                    }
                }
                warpgroupCommitBatch();
                warpgroupWait<0>();

                if (tid < Shape::ClusterSize) arriveCluster(&empty[qidx], tid);
                ++qidx;
            }

            for (int blockKIter = 1; blockKIter < numBlocksK; ++blockKIter, ++qidx) {
                if (qidx == QSIZE) { qidx = 0; p ^= 1; }
                mbarrierWait(&full[qidx], p);
                warpgroupArrive();

                #pragma unroll
                for (int mIt = 0; mIt < B_WG_M / WGMMA_M; ++mIt) {
                    half* wgmmaSa = sA + qidx * BK * BM + 64 * (mIt + activeConsumerIdx * (B_WG_M / WGMMA_M)) * WGMMA_M;
                    half* wgmmaSb = sB + qidx * BK * BN;

                    #pragma unroll
                    for (int bk = 0; bk < BK; bk += 64) {
                        #pragma unroll
                        for (int kIt = 0; kIt < 64 / WGMMA_K; ++kIt) {
                            wgmmaM64N256K16<1, 1, 1, 0, 0>(d[mIt], &wgmmaSa[kIt * WGMMA_K], &wgmmaSb[kIt * WGMMA_K]);
                        }
                        wgmmaSa += 64 * BM;
                        wgmmaSb += 64 * BN;
                    }
                }
                warpgroupCommitBatch();
                warpgroupWait<0>();

                if (tid < Shape::ClusterSize) arriveCluster(&empty[qidx], tid);
            }

            asm volatile("cp.async.bulk.wait_group 0;");

            const int lane = tid % 32;
            const int warp = tid / 32;
            const int row  = warp * 16 + lane / 4;
            const int colBase = 2 * (tid % 4);

            half* blockSc = sC + activeConsumerIdx * B_WG_M * BN;

            auto storeC = [&](int i, int j, float val) {
                blockSc[j * B_WG_M + i] = (half)val;
            };

            #pragma unroll
            for (int mIt = 0; mIt < B_WG_M / 64; ++mIt) {
                int rowOffset = mIt * 64;
                #pragma unroll
                for (int nIt = 0; nIt < BN / 16; ++nIt) {
                    int colOffset = nIt * 16;
                    
                    storeC(rowOffset + row + 0, colOffset + colBase,     d[mIt][nIt][0]);
                    storeC(rowOffset + row + 8, colOffset + colBase,     d[mIt][nIt][2]);
                    storeC(rowOffset + row + 0, colOffset + colBase + 1, d[mIt][nIt][1]);
                    storeC(rowOffset + row + 8, colOffset + colBase + 1, d[mIt][nIt][3]);
                    storeC(rowOffset + row + 0, colOffset + colBase + 8, d[mIt][nIt][4]);
                    storeC(rowOffset + row + 8, colOffset + colBase + 8, d[mIt][nIt][6]);
                    storeC(rowOffset + row + 0, colOffset + colBase + 9, d[mIt][nIt][5]);
                    storeC(rowOffset + row + 8, colOffset + colBase + 9, d[mIt][nIt][7]);
                }
            }

            asm volatile("bar.sync 10, 256;\n");
            
            if (threadIdx.x == 128) {
                storeAsync(&tensorMapC, &sC[0], numBlockM * BM, numBlockN * BN);
                asm volatile("cp.async.bulk.commit_group;");
            }
        }

        warpgroupRegDealloc<240>();
    }
}

// ===================================
// 5. Device-Level
// ===================================

CUtensorMap globalTmaMapA;
CUtensorMap globalTmaMapB;
CUtensorMap globalTmaMapC;
bool tmaInitialized = false;
int prevM = 0;
int prevN = 0;
int prevK = 0;

int* globalDspace = nullptr;
int spaceMTiles = 0;
int spaceNTiles = 0;
int spaceNumClusters = 0;


static int* buildHilbertTaskTable(int mTiles, int nTiles, int numClusters) {
    constexpr int kDepth = 128;

    const int totalTasks = mTiles * nTiles;

    int side = 1;
    while (side < ((mTiles > nTiles) ? mTiles : nTiles)) {
        side <<= 1;
    }

    int* table = (int*)malloc((size_t)numClusters * kDepth * sizeof(int));
    if (!table) {
        return nullptr;
    }

    for (int i = 0; i < numClusters * kDepth; ++i) {
        table[i] = -1;
    }

    auto decodeHilbertIndex = [](int orderSide, int index, int& x, int& y) {
        auto rotate = [](int n, int& x, int& y, int rx, int ry) {
            if (ry == 0) {
                if (rx == 1) {
                    x = n - 1 - x;
                    y = n - 1 - y;
                }
                int tmp = x;
                x = y;
                y = tmp;
            }
        };

        x = 0;
        y = 0;
        int t = index;

        for (int s = 1; s < orderSide; s <<= 1) {
            int rx = 1 & (t >> 1);
            int ry = 1 & (t ^ rx);
            rotate(s, x, y, rx, ry);
            x += s * rx;
            y += s * ry;
            t >>= 2;
        }
    };

    int taskId = 0;
    const int maxHilbertPoints = side * side;

    for (int d = 0; d < maxHilbertPoints && taskId < totalTasks; ++d) {
        int x, y;
        decodeHilbertIndex(side, d, x, y);

        if (y >= mTiles || x >= nTiles) {
            continue;
        }

        const int blockM = y;
        const int blockN = x;

        const int cluster = taskId % numClusters;
        const int slot = taskId / numClusters;

        if (slot < kDepth) {
            table[cluster * kDepth + slot] =
                (blockM << 16) | (blockN & 0xFFFF);
        }

        ++taskId;
    }

    return table;
}


void launchKernel(half* A, half* B, half* C, int M, int N, int K) {
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 64;
    constexpr int QSIZE = 3;
    constexpr int WGMMA_M = 64;
    constexpr int WGMMA_N = 256;
    constexpr int WGMMA_K = 16;
    constexpr int NUM_THREADS = 384;
    constexpr int NUM_SM = 128;
    constexpr int CLUSTER_M = 2;
    constexpr int CLUSTER_N = 1;

    using Shape = BlockShape<BM, BN, BK, QSIZE, WGMMA_M, WGMMA_N, WGMMA_K, NUM_THREADS, CLUSTER_M, CLUSTER_N>;

    if (!tmaInitialized || M != prevM || N != prevN || K != prevK) {
        globalTmaMapA = createTensorMapValue<BM, BK>(A, M, K);
        globalTmaMapB = createTensorMapValue<BN, BK>(B, N, K);
        globalTmaMapC = createTensorMapValue<BN, BM, false>(C, N, M);
        tmaInitialized = true;
        prevM = M;
        prevN = N;
        prevK = K;
    }

    const int numClusters = NUM_SM / Shape::ClusterSize;
    const int mTiles = (M + BM * CLUSTER_M - 1) / (BM * CLUSTER_M);
    const int nTiles = (N + BN * CLUSTER_N - 1) / (BN * CLUSTER_N);

    const bool needRegen = 
        (globalDspace == nullptr) || 
        (spaceMTiles != mTiles) || 
        (spaceNTiles != nTiles) || 
        (spaceNumClusters != numClusters);

    if (needRegen) {
        if (globalDspace) {
            cudaFree(globalDspace);
            globalDspace = nullptr;
        }
        int* hSpace = buildHilbertTaskTable(mTiles, nTiles, numClusters);
        cudaMalloc(&globalDspace, (size_t)numClusters * 128 * sizeof(int));
        cudaMemcpy(globalDspace, hSpace, (size_t)numClusters * 128 * sizeof(int), cudaMemcpyHostToDevice);
        free(hSpace);

        spaceMTiles = mTiles;
        spaceNTiles = nTiles;
        spaceNumClusters = numClusters;
    }

    size_t sharedBytes = sizeof(SharedStorage<BM, BN, BK, QSIZE>);
    auto kptr = &matmulKernel<Shape>;
    cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sharedBytes);

    kptr<<<NUM_SM, NUM_THREADS, sharedBytes>>>(M, N, K, globalTmaMapC, globalTmaMapA, globalTmaMapB, globalDspace);
}

}

using gemm_8192x8192_v20::launchKernel;