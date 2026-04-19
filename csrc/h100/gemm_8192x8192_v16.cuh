namespace gemm_8192x8192_v16 {

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
    int gCol, int gRow, uint16_t clusterMask) {
    uint64_t tmaPtr = reinterpret_cast<uint64_t>(srcTma);
    uint32_t mbarPtr = smemPtrU32(bar);
    uint32_t dstPtr  = smemPtrU32(dst);
    asm volatile (
        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster "
        "[%0], [%1, {0, %3, %4, 0, 0}], [%2], %5;"
        : : "r"(dstPtr), "l"(tmaPtr), "r"(mbarPtr), "r"(gRow), "r"(gCol), "h"(clusterMask)
        : "memory"
    );
}

__device__ __forceinline__ void tmaLoad5D(
    half* smemPtr, const CUtensorMap* map, int coordK64, int coordMn, uint64_t* bar) {
    uint32_t smemU32 = smemPtrU32(smemPtr);
    uint32_t barU32  = smemPtrU32(bar);
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {0, %3, %4, 0, 0}], [%2];"
        : : "r"(smemU32), "l"(map), "r"(barU32), "r"(coordMn), "r"(coordK64)
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
};

template<int BM, int BN, int TM = 16, int TN = 8>
struct Schedule {
    int block, it;
    int totalBlocksM, totalBlocksN;
    int numSm;

    __device__ __forceinline__ Schedule(int M, int N, int _block, int _numSm)
        : block(_block), it(0), numSm(_numSm)
    {
        totalBlocksM = M / BM;
        totalBlocksN = N / BN;
    }

    __device__ __forceinline__ bool next(int& blockM, int& blockN) {
        constexpr int KTM = (TM > 0 ? TM : 1);
        constexpr int KTN = (TN > 0 ? TN : 1);

        int total = totalBlocksM * totalBlocksN;
        if (total <= 0) return false;

        int tilesM = (totalBlocksM + KTM - 1) / KTM;
        int tilesN = (totalBlocksN + KTN - 1) / KTN;
        int tilesTotal = tilesM * tilesN;
        int virtualTotal = tilesTotal * KTM * KTN;

        int num = it * numSm + block;
        if (num >= virtualTotal) return false;

        while (num < virtualTotal) {
            int curTile = num / (KTM * KTN);
            int curPos  = num - curTile * (KTM * KTN);
            int tileM = curTile / tilesN;
            int tileN = curTile - tileM * tilesN;

            int m0 = tileM * KTM;
            int n0 = tileN * KTN;

            int dm = curPos / KTN;
            int dn = curPos - dm * KTN;

            blockM = m0 + dm;
            blockN = n0 + dn;
            ++it;

            if (blockM < totalBlocksM && blockN < totalBlocksN) {
                return true;
            }

            num = it * numSm + block;
        }

        return false;
    }
};

// ===================================
// 3. Collective-Level
// ===================================

template <int BlockMajorSize, int BlockMinorSize>
__host__ void createTensorMap(CUtensorMap* tmaMap, half* gmemPtr, int globalHeight, int globalWidth) {
    void* gmemAddress = (void*)gmemPtr;
    uint64_t gmemProbShape[5] = {64, (uint64_t)globalHeight, (uint64_t)(globalWidth / 64), 1, 1};
    uint64_t gmemProbStride[5] = {sizeof(half) * globalWidth, 64 * sizeof(half), 0, 0, 0};
    uint32_t smemBoxShape[5] = {64, (uint32_t)BlockMajorSize, (uint32_t)(BlockMinorSize / 64), 1, 1};
    uint32_t smemBoxStride[5] = {1, 1, 1, 1, 1};
    cuTensorMapEncodeTiled(tmaMap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 5, gmemAddress, gmemProbShape,
        gmemProbStride, smemBoxShape, smemBoxStride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

template <int BlockMajorSize, int BlockMinorSize>
__host__ CUtensorMap createTensorMapValue(half* src, int globalHeight, int globalWidth) {
    CUtensorMap tmaHost{};
    createTensorMap<BlockMajorSize, BlockMinorSize>(&tmaHost, src, globalHeight, globalWidth);
    return tmaHost;
}

// ===================================
// 4. Kernel-Level
// ===================================

template<typename Shape>
__global__ void __launch_bounds__(Shape::NumThreads)
__cluster_dims__(Shape::ClusterSize, 1, 1)
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
    constexpr int WGMMA_K = Shape::WgmmaK;
    constexpr int NUM_THREADS = Shape::NumThreads;

    constexpr int numConsumers = (NUM_THREADS / 128) - 1;
    constexpr int B_WG_M = BM / numConsumers;

    const int wgIdx = threadIdx.x / 128;
    const int tid = threadIdx.x % 128;

    if (wgIdx == 0) {
        constexpr int numRegs = (numConsumers <= 2 ? 24 : 32);
        warpgroupRegDealloc<numRegs>();
    } else {
        constexpr int numRegs = (numConsumers == 1 ? 256 : (numConsumers == 2 ? 240 : 160));
        warpgroupRegAlloc<numRegs>();
    }

    extern __shared__ __align__(128) uint8_t smemRaw[];
    auto& smem = *reinterpret_cast<SharedStorage<BM, BN, BK, QSIZE>*>(smemRaw);

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ alignas(8) uint64_t full[QSIZE];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ alignas(8) uint64_t empty[QSIZE];

    const int numBlocksK = K / BK;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < QSIZE; ++i) {
            mbarrierInit(&full[i], 0, 1);
            mbarrierInit(&empty[i], 0, numConsumers * Shape::ClusterSize);
        }
    }

    asm volatile("barrier.cluster.arrive;\n" : :);
    asm volatile("barrier.cluster.wait;\n" : :);

    const int clusterId = blockIdx.x / Shape::ClusterSize;
    const int clusterCount = gridDim.x / Shape::ClusterSize;
    Schedule<BM, BN, 16, 8> schedule(M, N, clusterId, clusterCount);

    if (wgIdx == 0) {
        if (tid == 0) {
            int clusterRank = blockIdx.x % Shape::ClusterSize;
            int rankM = clusterRank / Shape::ClusterN;
            int rankN = clusterRank % Shape::ClusterN;

            int p = 0, qidx = 0;
            int blockM, blockN;

            while (schedule.next(blockM, blockN)) {
                for (int kIter = 0; kIter < numBlocksK; ++kIter, ++qidx) {
                    if (qidx == QSIZE) { qidx = 0; p ^= 1; }

                    mbarrierWait(&empty[qidx], p);

                    uint32_t bytes = (BM * BK + BK * BN) * sizeof(half);
                    mbarrierExpectTx(&full[qidx], bytes);

                    if constexpr (Shape::ClusterN > 1) {
                        uint32_t mask = ((1 << Shape::ClusterN) - 1) << (rankM * Shape::ClusterN);
                        if (rankN == 0) {
                            loadAsyncMulticast(
                                &smem.A[qidx * BM * BK],
                                &tensorMapA,
                                &full[qidx],
                                kIter,
                                blockM * BM,
                                mask
                            );
                        }
                    } else {
                        tmaLoad5D(&smem.A[qidx * BM * BK], &tensorMapA, kIter, blockM * BM, &full[qidx]);
                    }

                    tmaLoad5D(&smem.B[qidx * BK * BN], &tensorMapB, kIter, blockN * BN, &full[qidx]);
                }
            }
        }
    } else {
        const int consumerIdx = wgIdx - 1;
        float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];

        #pragma unroll
        for (int i = 0; i < QSIZE; ++i) {
            if (tid < Shape::ClusterSize) {
                arriveCluster(&empty[i], tid);
            }
        }

        int p = 0, qidx = 0;
        int blockM, blockN;

        while (schedule.next(blockM, blockN)) {
            #pragma unroll
            for(int i = 0; i < B_WG_M / WGMMA_M; ++i) {
                #pragma unroll
                for(int j = 0; j < WGMMA_N / 16; ++j) {
                    #pragma unroll
                    for(int k = 0; k < 8; ++k) {
                        d[i][j][k] = 0.0f;
                    }
                }
            }

            for (int kIter = 0; kIter < numBlocksK; ++kIter, ++qidx) {
                if (qidx == QSIZE) { qidx = 0; p ^= 1; }

                mbarrierWait(&full[qidx], p);

                half* sABase = &smem.A[qidx * BM * BK];
                half* sBBase = &smem.B[qidx * BK * BN];

                warpgroupArrive();

                #pragma unroll
                for (int mIt = 0; mIt < B_WG_M / WGMMA_M; ++mIt) {
                    half* wgmmaSa = sABase + 64 * (mIt + consumerIdx * (B_WG_M / WGMMA_M)) * WGMMA_M;
                    half* wgmmaSb = sBBase;

                    #pragma unroll
                    for (int bk = 0; bk < BK; bk += 64) {
                        #pragma unroll
                        for (int kIt = 0; kIt < 64 / WGMMA_K; ++kIt) {
                            wgmmaM64N256K16<1, 1, 1, 0, 0>(
                                d[mIt],
                                wgmmaSa + kIt * WGMMA_K,
                                wgmmaSb + kIt * WGMMA_K
                            );
                        }
                        wgmmaSa += 64 * BM;
                        wgmmaSb += 64 * BN;
                    }
                }

                warpgroupCommitBatch();
                warpgroupWait<0>();

                if (tid < Shape::ClusterSize) {
                    arriveCluster(&empty[qidx], tid);
                }
            }

            const int lane = tid % 32;
            const int warp = tid / 32;
            const int row  = warp * 16 + lane / 4;

            half* cBlock = C + (blockN * BN) * M + (blockM * BM);

            auto storeC = [&](int i, int j, float val) {
                cBlock[j * M + i] = (half)val;
            };

            #pragma unroll
            for (int mIt = 0; mIt < B_WG_M / WGMMA_M; ++mIt) {
                int rowOff = mIt * WGMMA_M + consumerIdx * B_WG_M;
                #pragma unroll
                for (int w = 0; w < WGMMA_N / 16; ++w) {
                    int col = w * 16 + 2 * (lane % 4);
                    int i0 = row + rowOff;
                    int j0 = col;

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
    constexpr int CLUSTER_M = 1;
    constexpr int CLUSTER_N = 1;

    using Shape = BlockShape<BM, BN, BK, QSIZE, WGMMA_M, WGMMA_N, WGMMA_K, NUM_THREADS, CLUSTER_M, CLUSTER_N>;

    if (!tmaInitialized || M != prevM || N != prevN || K != prevK) {
        globalTmaMapA = createTensorMapValue<BM, BK>(A, M, K);
        globalTmaMapB = createTensorMapValue<BN, BK>(B, N, K);
        tmaInitialized = true;
        prevM = M;
        prevN = N;
        prevK = K;
    }

    size_t sharedBytes = sizeof(SharedStorage<BM, BN, BK, QSIZE>);
    auto kptr = &matmulKernel<Shape>;
    cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sharedBytes);

    kptr<<<NUM_SM, NUM_THREADS, sharedBytes>>>(M, N, K, C, globalTmaMapA, globalTmaMapB);
}

}

using gemm_8192x8192_v16::launchKernel;