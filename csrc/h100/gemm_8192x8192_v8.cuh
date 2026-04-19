namespace gemm_8192x8192_v8 {

using Barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// ===================================
// 1. Atom-Level
// ===================================

__device__ __forceinline__ uint64_t encodeMatrixDescriptor(uint64_t x) {
    return (x & 0x3FFFF) >> 4;
}

__device__ __forceinline__ uint64_t makeSmemDescriptor(const half* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0;
    desc |= encodeMatrixDescriptor(addr);
    desc |= encodeMatrixDescriptor(uint64_t(16)) << 16;
    desc |= encodeMatrixDescriptor(uint64_t(1024)) << 32;
    desc |= 1ull << 62;
    return desc;
}

__device__ __forceinline__ void wgmmaFence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmmaCommit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int Groups>
__device__ __forceinline__ void wgmmaWait() {
    static_assert(Groups >= 0 && Groups <= 7, "WGMMA wait groups must be in [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" : : "n"(Groups) : "memory");
}

template <uint32_t RegCount>
__device__ __forceinline__ void warpgroupRegAlloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount) : "memory");
}

template <uint32_t RegCount>
__device__ __forceinline__ void warpgroupRegDealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount) : "memory");
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmmaM64n256k16(float d[16][8], half* sA, half* sB) {
    uint64_t descA = makeSmemDescriptor(sA);
    uint64_t descB = makeSmemDescriptor(sB);
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
        " %130, %131, %132, %133, %134;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
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
        : "l"(descA), "l"(descB), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

// ===================================
// 2. Tile-Level
// ===================================

template <int M, int N, int K, int Qsize, int THREADS>
struct BlockShape {
    static constexpr int TileM = M;
    static constexpr int TileN = N;
    static constexpr int TileK = K;
    static constexpr int QueueSize = Qsize;
    static constexpr int NumThreads = THREADS;

    static constexpr int WgmmaM = 64;
    static constexpr int WgmmaN = 256;
    static constexpr int WgmmaK = 16;

    static constexpr int WarpGroupSize = 128;
    static constexpr int NumWarpGroups = NumThreads / WarpGroupSize;
    static constexpr int NumConsumerGroups = NumWarpGroups - 1;
    static constexpr int ConsumerTileM = TileM / NumConsumerGroups;

    static constexpr int NumWgmmaM = ConsumerTileM / WgmmaM;
    static constexpr int NumWgmmaN = TileN / WgmmaN;
    static constexpr int NumKSteps = TileK / WgmmaK;

    static constexpr int SmemElemsA = TileM * TileK * QueueSize;
    static constexpr int SmemElemsB = TileK * TileN * QueueSize;

    static_assert((NumThreads % WarpGroupSize) == 0, "NumThreads must be multiple of 128");
    static_assert(NumConsumerGroups >= 1, "At least one consumer warpgroup is required");
    static_assert((TileM % NumConsumerGroups) == 0, "TileM must be divisible by number of consumers");
    static_assert((ConsumerTileM % WgmmaM) == 0, "ConsumerTileM must be divisible by WgmmaM");
};

template <typename Shape>
struct alignas(128) SharedStorage {
    half A[Shape::SmemElemsA];
    half B[Shape::SmemElemsB];
};

__device__ __forceinline__ void mapThreadToEpilogue(int tid, int consumerGroupIdx, int mIter,
                                                    int nIter, int wIter, int& row, int& col,
                                                    int consumerTileM, int wgmmaM,
                                                    int wgmmaN) {
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int rowInTile = warp * 16 + (lane >> 2);
    const int colInTile = 16 * wIter + 2 * (lane & 3);

    row = consumerGroupIdx * consumerTileM + mIter * wgmmaM + rowInTile;
    col = nIter * wgmmaN + colInTile;
}

// ===================================
// 3. Collective-Level
// ===================================

template <int BlockMajor, int BlockMinor>
__host__ void createTensorMap(CUtensorMap* tmaMap, half* src, int blocksHeight, int blocksK,
                              int leadingDimK) {
    void* gmemAddress = static_cast<void*>(src);
    uint64_t globalShape[5] = {
        uint64_t(BlockMinor) * uint64_t(blocksK),
        uint64_t(BlockMajor) * uint64_t(blocksHeight),
        1,
        1,
        1,
    };
    uint64_t globalStride[5] = {
        sizeof(half),
        uint64_t(sizeof(half)) * uint64_t(leadingDimK),
        0,
        0,
        0,
    };
    uint32_t smemShape[5] = {uint32_t(BlockMinor), uint32_t(BlockMajor), 1, 1, 1};
    uint32_t smemStride[5] = {1, 1, 1, 1, 1};

    cuTensorMapEncodeTiled(
        tmaMap,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        2,
        gmemAddress,
        globalShape,
        globalStride + 1,
        smemShape,
        smemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
}

template <int BlockMajor, int BlockMinor>
__host__ CUtensorMap* createTensorMapDevice(half* src, int blocksHeight, int blocksK,
                                            int leadingDimK) {
    CUtensorMap hostMap;
    createTensorMap<BlockMajor, BlockMinor>(&hostMap, src, blocksHeight, blocksK, leadingDimK);

    CUtensorMap* deviceMap = nullptr;
    cudaMalloc(&deviceMap, sizeof(CUtensorMap));
    cudaMemcpy(deviceMap, &hostMap, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return deviceMap;
}

template <typename Shape>
__device__ __forceinline__ void initPipelineBarriers(Barrier full[Shape::QueueSize],
                                                     Barrier empty[Shape::QueueSize]) {
    constexpr int consumerThreads = Shape::NumConsumerGroups * Shape::WarpGroupSize;
    if (threadIdx.x == 0) {
        for (int q = 0; q < Shape::QueueSize; ++q) {
            init(&full[q], consumerThreads + 1);
            init(&empty[q], consumerThreads + 1);
        }
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();
}

template <typename Shape>
__device__ __forceinline__ void producerFetchTile(SharedStorage<Shape>* smem,
                                                  const CUtensorMap* mapA,
                                                  const CUtensorMap* mapB,
                                                  int blockM,
                                                  int blockN,
                                                  int kIter,
                                                  Barrier& full,
                                                  Barrier& empty) {
    empty.wait(empty.arrive());

    half* dstA = &smem->A[kIter % Shape::QueueSize * (Shape::TileM * Shape::TileK)];
    half* dstB = &smem->B[kIter % Shape::QueueSize * (Shape::TileK * Shape::TileN)];

    cde::cp_async_bulk_tensor_2d_global_to_shared(dstA, mapA, kIter * Shape::TileK,
                                                  blockM * Shape::TileM, full);
    cde::cp_async_bulk_tensor_2d_global_to_shared(dstB, mapB, kIter * Shape::TileK,
                                                  blockN * Shape::TileN, full);

    const uint32_t bytes = (Shape::TileM * Shape::TileK + Shape::TileK * Shape::TileN) * sizeof(half);
    cuda::device::barrier_arrive_tx(full, 1, bytes);
}

// ===================================
// 4. Kernel-Level
// ===================================

template <typename Shape>
__global__ void __launch_bounds__(Shape::NumThreads)
    gemm(int M, int N, int K, half* __restrict__ C, const CUtensorMap* __restrict__ tmaA,
         const CUtensorMap* __restrict__ tmaB) {

    const int warpGroupIdx = threadIdx.x / Shape::WarpGroupSize;
    const int warpGroupTid = threadIdx.x & (Shape::WarpGroupSize - 1);

    if (warpGroupIdx == 0) {
        constexpr int producerRegs = (Shape::NumConsumerGroups <= 2 ? 24 : 32);
        warpgroupRegDealloc<producerRegs>();
    } else {
        constexpr int consumerRegs =
            (Shape::NumConsumerGroups == 1 ? 256 : (Shape::NumConsumerGroups == 2 ? 240 : 160));
        warpgroupRegAlloc<consumerRegs>();
    }

    extern __shared__ __align__(128) unsigned char smemRaw[];
    auto* smem = reinterpret_cast<SharedStorage<Shape>*>(smemRaw);

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ Barrier full[Shape::QueueSize];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ Barrier empty[Shape::QueueSize];

    initPipelineBarriers<Shape>(full, empty);

    const int numBlocksK = (K + Shape::TileK - 1) / Shape::TileK;
    const int blockIdxN = blockIdx.x % (N / Shape::TileN);
    const int blockIdxM = blockIdx.x / (N / Shape::TileN);

    if (warpGroupIdx == 0) {
        if (warpGroupTid == 0) {
            for (int kIter = 0; kIter < numBlocksK; ++kIter) {
                producerFetchTile<Shape>(smem, tmaA, tmaB, blockIdxM, blockIdxN, kIter,
                                         full[kIter % Shape::QueueSize],
                                         empty[kIter % Shape::QueueSize]);
            }
        }
        return;
    }

    const int consumerGroupIdx = warpGroupIdx - 1;
    float acc[Shape::NumWgmmaM][Shape::WgmmaN / 16][8] = {0.0f};

#pragma unroll
    for (int q = 0; q < Shape::QueueSize; ++q) {
        (void)empty[q].arrive();
    }

    for (int kIter = 0; kIter < numBlocksK; ++kIter) {
        const int q = kIter % Shape::QueueSize;
        full[q].wait(full[q].arrive());

        half* tileA = &smem->A[q * (Shape::TileM * Shape::TileK)];
        half* tileB = &smem->B[q * (Shape::TileK * Shape::TileN)];

        wgmmaFence();

        int kTail = K - kIter * Shape::TileK;
        if (kTail > Shape::TileK) {
            kTail = Shape::TileK;
        }
        const int numKSteps = (kTail + Shape::WgmmaK - 1) / Shape::WgmmaK;

#pragma unroll
        for (int mIter = 0; mIter < Shape::NumWgmmaM; ++mIter) {
            half* aPtr = tileA + (consumerGroupIdx * Shape::ConsumerTileM + mIter * Shape::WgmmaM) * Shape::TileK;
#pragma unroll
            for (int kStep = 0; kStep < numKSteps; ++kStep) {
                wgmmaM64n256k16<1, 1, 1, 0, 0>(acc[mIter], aPtr + kStep * Shape::WgmmaK,
                                                    tileB + kStep * Shape::WgmmaK);
            }
        }

        wgmmaCommit();
        wgmmaWait<0>();
        (void)empty[q].arrive();
    }

    half* blockC = C + (blockIdxN * Shape::TileN) * M + (blockIdxM * Shape::TileM);

    for (int mIter = 0; mIter < Shape::NumWgmmaM; ++mIter) {
        for (int nIter = 0; nIter < Shape::NumWgmmaN; ++nIter) {
            for (int wIter = 0; wIter < Shape::WgmmaN / 16; ++wIter) {
                int row, col;
                mapThreadToEpilogue(warpGroupTid, consumerGroupIdx, mIter, nIter, wIter, row, col,
                                    Shape::ConsumerTileM, Shape::WgmmaM, Shape::WgmmaN);

                float* frag = &acc[mIter][wIter][0];
                auto storeC = [&](int rowOffset, int colOffset, float value) {
                    blockC[(col + colOffset) * M + (row + rowOffset)] = __float2half(value);
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

__global__ void fallbackGemmKernel(int M, int N, int K, const half* A, const half* B, half* C) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) {
        return;
    }

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += float(__ldg(A + k + m * K)) * float(__ldg(B + k + n * K));
    }
    C[m + n * M] = __float2half(acc);
}

// ===================================
// 5. Device-Level
// ===================================

static CUtensorMap* globalTmaMapA = nullptr;
static CUtensorMap* globalTmaMapB = nullptr;
static int prevM = 0;
static int prevN = 0;
static int prevK = 0;

void launchKernel(half* A, half* B, half* C, int M, int N, int K) {
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 64;
    constexpr int QSIZE = 3;
    constexpr int NUM_THREADS = 384;
    using Shape = BlockShape<BM, BN, BK, QSIZE, NUM_THREADS>;

    if (K <= 0) {
        return;
    }

    if (K % 16 != 0) {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        fallbackGemmKernel<<<grid, block>>>(M, N, K, A, B, C);
        return;
    }

    const int blocksK = (K + Shape::TileK - 1) / Shape::TileK;
    if (!globalTmaMapA || M != prevM || N != prevN || K != prevK) {
        if (globalTmaMapA) cudaFree(globalTmaMapA);
        if (globalTmaMapB) cudaFree(globalTmaMapB);

        globalTmaMapA = createTensorMapDevice<Shape::TileM, Shape::TileK>(A, M / Shape::TileM,
                                                                           blocksK, K);
        globalTmaMapB = createTensorMapDevice<Shape::TileN, Shape::TileK>(B, N / Shape::TileN,
                                                                           blocksK, K);
        prevM = M;
        prevN = N;
        prevK = K;
    }

    size_t sharedBytes = sizeof(SharedStorage<Shape>);
    auto kernel = &gemm<Shape>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         int(sharedBytes));
    kernel<<<(M / Shape::TileM) * (N / Shape::TileN), Shape::NumThreads, sharedBytes>>>(
        M, N, K, C, globalTmaMapA, globalTmaMapB);
}

}  // namespace gemm_8192x8192_v8

using gemm_8192x8192_v8::launchKernel;
