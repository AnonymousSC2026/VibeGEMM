namespace gemm_8192x8192_v6 {

using Barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// ===================================
// 1. Atom-Level
// ===================================

__device__ __forceinline__ uint64_t encodeMatrixDescriptor(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

__device__ __forceinline__ uint64_t makeSmemDescriptor(half *ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= encodeMatrixDescriptor(addr);
    desc |= encodeMatrixDescriptor((uint64_t)16) << 16;
    desc |= encodeMatrixDescriptor((uint64_t)1024) << 32;
    desc |= 1llu << 62; // 128B swizzle
    return desc;
}

__device__ __forceinline__ void wgmmaFence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmmaCommit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void wgmmaWait() {
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

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmmaM64N128K16(float d[8][8], half *sA, half *sB) {
    uint64_t descA = makeSmemDescriptor(sA);
    uint64_t descB = makeSmemDescriptor(sB);

    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},"
        " %64, %65, %66, %67, %68, %69, %70;\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
          "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
          "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
          "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
          "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        : "l"(descA), "l"(descB), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
          "n"(int32_t(TransA)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

// ===================================
// 2. Tile-Level
// ===================================

template <int BM, int BN, int BK, int QSIZE>
struct SharedStorage {
    alignas(128) half A[BM * BK * QSIZE];
    alignas(128) half B[BK * BN * QSIZE];
};

template<int BM, int BN, int BK, int QSIZE, int WGMMA_M, int WGMMA_N, int WGMMA_K, int NUM_THREADS>
struct BlockShape {
    static constexpr int TileM = BM;
    static constexpr int TileN = BN;
    static constexpr int TileK = BK;
    static constexpr int QueueSize = QSIZE;

    static constexpr int WgmmaM = WGMMA_M;
    static constexpr int WgmmaN = WGMMA_N;
    static constexpr int WgmmaK = WGMMA_K;

    static constexpr int NumThreads = NUM_THREADS;
    static constexpr int ProducerWarpGroup = 0;

    static constexpr int NumConsumers = (NUM_THREADS / 128) - 1;
    static constexpr int ConsumerTileM = BM / NumConsumers;
    static constexpr int NumConsumerThreads = NumConsumers * 128;
};

template <typename Shape>
__device__ __forceinline__ void configureWarpgroupRegisters(int wgIdx) {
    if (wgIdx == Shape::ProducerWarpGroup) {
        constexpr int numRegs = (Shape::NumConsumers <= 2 ? 24 : 32);
        warpgroupRegDealloc<numRegs>();
    } else {
        constexpr int numRegs = (Shape::NumConsumers == 1 ? 256 : (Shape::NumConsumers == 2 ? 240 : 160));
        warpgroupRegAlloc<numRegs>();
    }
}

template <typename Shape>
__device__ __forceinline__ void mapThreadToEpilogue(
    int tid128,
    int activeConsumerIdx,
    int mIt,
    int nIt,
    int w,
    int &row,
    int &col) {

    const int lane = tid128 & 31;
    const int wgWarp = tid128 >> 5;
    const int rowInTileLocal = wgWarp * 16 + (lane >> 2);
    const int rowBase = activeConsumerIdx * Shape::ConsumerTileM;
    const int colInTile = 16 * w + 2 * (lane & 3);

    row = rowBase + mIt * Shape::WgmmaM + rowInTileLocal;
    col = nIt * Shape::WgmmaN + colInTile;
}

// ===================================
// 3. Collective-Level
// ===================================

template <int BlockMajorSize, int BlockMinorSize>
__host__ void createTmaMap(CUtensorMap *tmaMap, half *src, int blocksHeight, int blocksWidth) {
    void *gmemAddress = (void *)src;
    uint64_t gmemProbShape[5] = {
        (uint64_t)BlockMinorSize * blocksWidth,
        (uint64_t)BlockMajorSize * blocksHeight,
        1, 1, 1
    };
    uint64_t gmemProbStride[5] = {
        sizeof(half),
        sizeof(half) * BlockMinorSize * blocksWidth,
        0, 0, 0
    };
    uint32_t smemBoxShape[5] = {
        uint32_t(BlockMinorSize),
        uint32_t(BlockMajorSize),
        1, 1, 1
    };
    uint32_t smemBoxStride[5] = {1, 1, 1, 1, 1};

    cuTensorMapEncodeTiled(
        tmaMap,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        2,
        gmemAddress,
        gmemProbShape,
        gmemProbStride + 1,
        smemBoxShape,
        smemBoxStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

template <int BlockMajorSize, int BlockMinorSize>
__host__ __forceinline__ CUtensorMap *createTmaMapDevice(half *src, int blocksHeight, int blocksWidth) {
    CUtensorMap *tmaMapDevice;
    cudaMalloc(&tmaMapDevice, sizeof(CUtensorMap));

    CUtensorMap tmaMapHost;
    createTmaMap<BlockMajorSize, BlockMinorSize>(&tmaMapHost, src, blocksHeight, blocksWidth);
    cudaMemcpy(tmaMapDevice, &tmaMapHost, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

    return tmaMapDevice;
}

template <typename Shape>
__device__ __forceinline__ half *getStageA(SharedStorage<Shape::TileM, Shape::TileN, Shape::TileK, Shape::QueueSize> *smem, int q) {
    return &smem->A[q * (Shape::TileM * Shape::TileK)];
}

template <typename Shape>
__device__ __forceinline__ half *getStageB(SharedStorage<Shape::TileM, Shape::TileN, Shape::TileK, Shape::QueueSize> *smem, int q) {
    return &smem->B[q * (Shape::TileK * Shape::TileN)];
}

template <typename Shape>
__device__ __forceinline__ void tmaFetchAsync(
    SharedStorage<Shape::TileM, Shape::TileN, Shape::TileK, Shape::QueueSize> *smem,
    int q,
    const CUtensorMap *tensorMapA,
    const CUtensorMap *tensorMapB,
    int blockIdxM,
    int blockIdxN,
    int kIter,
    Barrier &full) {

    half *sA = getStageA<Shape>(smem, q);
    half *sB = getStageB<Shape>(smem, q);

    cde::cp_async_bulk_tensor_2d_global_to_shared(
        sA, tensorMapA, kIter * Shape::TileK, blockIdxM * Shape::TileM, full);
    cde::cp_async_bulk_tensor_2d_global_to_shared(
        sB, tensorMapB, kIter * Shape::TileK, blockIdxN * Shape::TileN, full);
}

// ===================================
// 4. Kernel-Level
// ===================================

template <typename Shape>
__global__ void __launch_bounds__(Shape::NumThreads)
gemm(int M, int N, int K, half *C, const CUtensorMap *tensorMapA, const CUtensorMap *tensorMapB) {
    const int wgIdx = threadIdx.x / 128;
    const int tid128 = threadIdx.x & 127;

    constexpr int numConsumers = Shape::NumConsumers;
    constexpr int consumerTileM = Shape::ConsumerTileM;
    constexpr int numConsumerThreads = Shape::NumConsumerThreads;

    static_assert((Shape::NumThreads % 128) == 0, "NUM_THREADS must be multiple of 128");
    static_assert(numConsumers >= 1, "num_consumers must be >= 1");
    static_assert((Shape::TileM % numConsumers) == 0, "BM must be divisible by num_consumers");
    static_assert((consumerTileM % Shape::WgmmaM) == 0, "B_WG_M must be divisible by WGMMA_M");

    configureWarpgroupRegisters<Shape>(wgIdx);

    extern __shared__ __align__(128) unsigned char smemRaw[];
    auto *smem = reinterpret_cast<SharedStorage<Shape::TileM, Shape::TileN, Shape::TileK, Shape::QueueSize> *>(smemRaw);

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ Barrier full[Shape::QueueSize];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ Barrier empty[Shape::QueueSize];

    if (threadIdx.x == 0) {
        for (int q = 0; q < Shape::QueueSize; ++q) {
            init(&full[q], numConsumerThreads + 1);
            init(&empty[q], numConsumerThreads + 1);
        }
    }
    __syncthreads();

    const int numBlocksK = K / Shape::TileK;
    const int blockIdxN = blockIdx.x % (N / Shape::TileN);
    const int blockIdxM = blockIdx.x / (N / Shape::TileN);

    if (wgIdx == Shape::ProducerWarpGroup) {
        if (tid128 == 0) {
            for (int kIter = 0; kIter < numBlocksK; ++kIter) {
                int q = kIter % Shape::QueueSize;

                empty[q].wait(empty[q].arrive());

                half *sA = &smem->A[q * (Shape::TileM * Shape::TileK)];
                half *sB = &smem->B[q * (Shape::TileK * Shape::TileN)];

                cde::cp_async_bulk_tensor_2d_global_to_shared(
                    sA, tensorMapA, kIter * Shape::TileK, blockIdxM * Shape::TileM, full[q]);
                cde::cp_async_bulk_tensor_2d_global_to_shared(
                    sB, tensorMapB, kIter * Shape::TileK, blockIdxN * Shape::TileN, full[q]);

                uint32_t bytes = (Shape::TileM * Shape::TileK + Shape::TileK * Shape::TileN) * sizeof(half);
                cuda::device::barrier_arrive_tx(full[q], 1, bytes);
            }
        }
    } else {
        const int activeConsumerIdx = wgIdx - 1;
        float d[Shape::ConsumerTileM / Shape::WgmmaM][Shape::WgmmaN / 16][8] = {0.0f};

#pragma unroll
        for (int q = 0; q < Shape::QueueSize; ++q) {
            (void)empty[q].arrive();
        }

        for (int kIter = 0; kIter < numBlocksK; ++kIter) {
            int q = kIter % Shape::QueueSize;
            full[q].wait(full[q].arrive());

            half *sA = getStageA<Shape>(smem, q);
            half *sB = getStageB<Shape>(smem, q);

            wgmmaFence();

#pragma unroll
            for (int mIt = 0; mIt < Shape::ConsumerTileM / Shape::WgmmaM; ++mIt) {
                half *wgmmaSA = sA + (activeConsumerIdx * Shape::ConsumerTileM + mIt * Shape::WgmmaM) * Shape::TileK;

#pragma unroll
                for (int kStep = 0; kStep < Shape::TileK / 16; ++kStep) {
                    wgmmaM64N128K16<1, 1, 1, 0, 0>(
                        reinterpret_cast<float (*)[8]>(d[mIt]),
                        wgmmaSA + kStep * 16,
                        sB + kStep * 16
                    );
                }
            }

            wgmmaCommit();
            wgmmaWait<0>();
            (void)empty[q].arrive();
        }

        half *blockC = C + (blockIdxN * Shape::TileN) * M + (blockIdxM * Shape::TileM);

        for (int mIt = 0; mIt < Shape::ConsumerTileM / Shape::WgmmaM; ++mIt) {
            for (int nIt = 0; nIt < Shape::TileN / Shape::WgmmaN; ++nIt) {
                for (int w = 0; w < Shape::WgmmaN / 16; ++w) {
                    int row, col;
                    mapThreadToEpilogue<Shape>(tid128, activeConsumerIdx, mIt, nIt, w, row, col);

                    float *frag = &d[mIt][w][0];
                    auto storeC = [&](int rOff, int cOff, float val) {
                        blockC[(col + cOff) * M + (row + rOff)] = (half)val;
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
}

// ===================================
// 5. Device-Level
// ===================================

static CUtensorMap *globalTmaMapA = 0;
static CUtensorMap *globalTmaMapB = 0;
static int prevM = 0;
static int prevN = 0;
static int prevK = 0;

void launchKernel(half *A, half *B, half *C, int M, int N, int K) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int WGMMA_M = 64;
    constexpr int WGMMA_N = 128;
    constexpr int WGMMA_K = 16;
    constexpr int QSIZE = 5;
    constexpr int NUM_THREADS = 384;

    using Shape = BlockShape<BM, BN, BK, QSIZE, WGMMA_M, WGMMA_N, WGMMA_K, NUM_THREADS>;

    if (!globalTmaMapA || M != prevM || N != prevN || K != prevK) {
        if (globalTmaMapA) cudaFree(globalTmaMapA);
        if (globalTmaMapB) cudaFree(globalTmaMapB);

        globalTmaMapA = createTmaMapDevice<BM, BK>(A, M / BM, K / BK);
        globalTmaMapB = createTmaMapDevice<BN, BK>(B, N / BN, K / BK);

        prevM = M;
        prevN = N;
        prevK = K;
    }

    size_t sharedBytes = sizeof(SharedStorage<BM, BN, BK, QSIZE>);
    auto kernelPtr = &gemm<Shape>;

    cudaFuncSetAttribute(kernelPtr, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sharedBytes);
    kernelPtr<<<(M / BM) * (N / BN), NUM_THREADS, sharedBytes>>>(M, N, K, C, globalTmaMapA, globalTmaMapB);
}

} // namespace gemm_8192x8192_v6

using gemm_8192x8192_v6::launchKernel;