namespace gemm_8192x8192_v4 {

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
    static_assert(N >= 0 && N <= 7, "wgmmaWait<N>: N must be in [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" : : "n"(N) : "memory");
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmmaM64N64K16(float d[4][8], const half *sA, const half *sB) {
    uint64_t descA = makeSmemDescriptor(sA);
    uint64_t descB = makeSmemDescriptor(sB);

    asm volatile("{\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
                 "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                 " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                 " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                 " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
                 " %32,"
                 " %33,"
                 " %34, %35, %36, %37, %38;\n"
                 "}\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]),
                   "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]),
                   "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]),
                   "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
                   "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]), "+f"(d[3][0]),
                   "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
                   "+f"(d[3][6]), "+f"(d[3][7])
                 : "l"(descA), "l"(descB), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
                   "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmmaM64N128K16(float d[8][8], const half *sA, const half *sB) {
    uint64_t descA = makeSmemDescriptor(sA);
    uint64_t descB = makeSmemDescriptor(sB);

    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63},"
        " %64,"
        " %65,"
        " %66, %67, %68, %69, %70;\n"
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
          "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        : "l"(descA), "l"(descB), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
          "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}


// ===================================
// 2. Tile-Level
// ===================================

template <int M, int N, int K, int QSIZE, int THREADS> struct BlockShape {
    static constexpr int TileM = M;
    static constexpr int TileN = N;
    static constexpr int TileK = K;
    static constexpr int QueueSize = QSIZE;
    static constexpr int NumThreads = THREADS;

    static constexpr int WgmmaM = 64;
    static constexpr int WgmmaK = 16;
    static constexpr int ProducerWarpGroup = 0;
    static constexpr int ConsumerWarpGroup = 1;

    static constexpr int NumWgmmaM = TileM / WgmmaM;
    static constexpr int NumItersK = TileK / WgmmaK;
};

template <typename Shape> struct WgmmaShape;

template <int M, int K, int QSIZE, int THREADS>
struct WgmmaShape<BlockShape<M, 64, K, QSIZE, THREADS>> {
    static constexpr int WgmmaN = 64;
};

template <int M, int K, int QSIZE, int THREADS>
struct WgmmaShape<BlockShape<M, 128, K, QSIZE, THREADS>> {
    static constexpr int WgmmaN = 128;
};

template <typename Shape>
__device__ __forceinline__ void mapThreadToEpilogue(int tid128, int mIt, int nIt, int w, int &row,
                                                    int &col) {
    const int lane = tid128 & 31;
    const int wgWarp = tid128 >> 5;
    const int rowInTile = wgWarp * 16 + (lane >> 2);
    const int rowOffset = mIt * Shape::WgmmaM;
    const int colInTile = 16 * w + 2 * (lane & 3);

    row = rowOffset + rowInTile;
    col = nIt * WgmmaShape<Shape>::WgmmaN + colInTile;
}

// ===================================
// 3. Collective-Level
// ===================================

template <typename Shape> struct alignas(128) RingBufferSmem {
    half A[Shape::TileM * Shape::TileK * Shape::QueueSize];
    half B[Shape::TileK * Shape::TileN * Shape::QueueSize];
};

template <int BlockMajorSize, int BlockMinorSize>
__host__ void createTmaMap(CUtensorMap *tmaMap, const half *src, int blocksHeight,
                           int blocksWidth) {
    uint64_t shape[5] = {(uint64_t)BlockMinorSize * blocksWidth,
                         (uint64_t)BlockMajorSize * blocksHeight, 1, 1, 1};
    uint64_t stride[5] = {sizeof(half), sizeof(half) * BlockMinorSize * blocksWidth, 0, 0, 0};
    uint32_t smemShape[5] = {uint32_t(BlockMinorSize), uint32_t(BlockMajorSize), 1, 1, 1};
    uint32_t smemStride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        tmaMap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, (void *)src, shape, stride + 1, smemShape,
        smemStride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);
}

template <int BlockMajorSize, int BlockMinorSize>
__host__ CUtensorMap *createTmaMapDevice(const half *src, int blocksHeight, int blocksWidth) {
    CUtensorMap hostMap;
    createTmaMap<BlockMajorSize, BlockMinorSize>(&hostMap, src, blocksHeight, blocksWidth);

    CUtensorMap *devMap;
    cudaMalloc(&devMap, sizeof(CUtensorMap));
    cudaMemcpy(devMap, &hostMap, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return devMap;
}

template <typename Shape>
__device__ __forceinline__ half *getStageA(RingBufferSmem<Shape> *smem, int q) {
    return &smem->A[q * (Shape::TileM * Shape::TileK)];
}

template <typename Shape>
__device__ __forceinline__ half *getStageB(RingBufferSmem<Shape> *smem, int q) {
    return &smem->B[q * (Shape::TileK * Shape::TileN)];
}

template <typename Shape>
__device__ __forceinline__ void
tmaFetchAsync(RingBufferSmem<Shape> *smem, int q, const CUtensorMap *mapA, const CUtensorMap *mapB,
              int blockIdxM, int blockIdxN, int kIter, Barrier &fullA, Barrier &fullB) {

    half *dstA = getStageA(smem, q);
    half *dstB = getStageB(smem, q);

    cde::cp_async_bulk_tensor_2d_global_to_shared(dstA, mapA, kIter * Shape::TileK,
                                                  blockIdxM * Shape::TileM, fullA);

    cde::cp_async_bulk_tensor_2d_global_to_shared(dstB, mapB, kIter * Shape::TileK,
                                                  blockIdxN * Shape::TileN, fullB);
}

// ===================================
// 4. Kernel-Level
// ===================================

template <typename Shape>
__global__ void __launch_bounds__(Shape::NumThreads)
    gemm(int M, int N, int K, half *__restrict__ C, const CUtensorMap *__restrict__ tmaA,
         const CUtensorMap *__restrict__ tmaB) {

    constexpr int WgmmaN = WgmmaShape<Shape>::WgmmaN;
    constexpr int BarrierCount = 129;
    constexpr int ConsumerThreads = 128;

    const int wgIdx = threadIdx.x / 128;
    const int tid128 = threadIdx.x & 127;

    extern __shared__ __align__(128) unsigned char smemRaw[];
    auto *smem = reinterpret_cast<RingBufferSmem<Shape> *>(smemRaw);

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ Barrier fullA[Shape::QueueSize];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ Barrier fullB[Shape::QueueSize];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ Barrier empty[Shape::QueueSize];

    const int numBlocksK = K / Shape::TileK;
    const int bIdxN = blockIdx.x % (N / Shape::TileN);
    const int bIdxM = blockIdx.x / (N / Shape::TileN);

    float acc[Shape::NumWgmmaM][WgmmaN / 16][8];
    static_assert(sizeof(acc) * ConsumerThreads == Shape::TileM * Shape::TileN * sizeof(float),
                  "Accumulator layout must cover exactly one output tile.");

    if (threadIdx.x == 0) {
#pragma unroll
        for (int q = 0; q < Shape::QueueSize; ++q) {
            init(&fullA[q], BarrierCount);
            init(&fullB[q], BarrierCount);
            init(&empty[q], BarrierCount);
        }
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    if (wgIdx == Shape::ConsumerWarpGroup) {
#pragma unroll
        for (int q = 0; q < Shape::QueueSize; ++q) {
            (void)empty[q].arrive();
        }
    }
    __syncthreads();

    if (wgIdx == Shape::ProducerWarpGroup && tid128 == 0) {
        for (int kIter = 0; kIter < numBlocksK; ++kIter) {
            const int q = kIter % Shape::QueueSize;

            empty[q].wait(empty[q].arrive());

            tmaFetchAsync<Shape>(smem, q, tmaA, tmaB, bIdxM, bIdxN, kIter, fullA[q], fullB[q]);
            (void)cuda::device::barrier_arrive_tx(fullA[q], 1,
                                                  Shape::TileM * Shape::TileK * sizeof(half));
            (void)cuda::device::barrier_arrive_tx(fullB[q], 1,
                                                  Shape::TileK * Shape::TileN * sizeof(half));
        }
    }

    if (wgIdx == Shape::ConsumerWarpGroup) {
        for (int kIter = 0; kIter < numBlocksK; ++kIter) {
            const int q = kIter % Shape::QueueSize;

            fullA[q].wait(fullA[q].arrive());
            fullB[q].wait(fullB[q].arrive());

            half *stageA = getStageA(smem, q);
            half *stageB = getStageB(smem, q);

            wgmmaFence();

#pragma unroll
            for (int mi = 0; mi < Shape::NumWgmmaM; ++mi) {
                half *tileA = &stageA[mi * Shape::WgmmaM * Shape::TileK];

#pragma unroll
                for (int ki = 0; ki < Shape::NumItersK; ++ki) {
                    half *ptrA = &tileA[ki * Shape::WgmmaK];
                    half *ptrB = &stageB[ki * Shape::WgmmaK];

                    if (kIter == 0 && ki == 0) {
                        wgmmaM64N128K16<0, 1, 1, 0, 0>(acc[mi], ptrA, ptrB);
                    } else {
                        wgmmaM64N128K16<1, 1, 1, 0, 0>(acc[mi], ptrA, ptrB);
                    }
                }
            }

            wgmmaCommit();
            wgmmaWait<0>();

            (void)empty[q].arrive();
        }

        half *blockC = C + (bIdxN * Shape::TileN) * M + (bIdxM * Shape::TileM);

        for (int mi = 0; mi < Shape::NumWgmmaM; ++mi) {
            for (int ni = 0; ni < Shape::TileN / WgmmaN; ++ni) {
                for (int w = 0; w < WgmmaN / 16; ++w) {
                    int row, col;
                    mapThreadToEpilogue<Shape>(tid128, mi, ni, w, row, col);

                    float *frag = &acc[mi][w][0];
                    auto storeC = [&](int rOff, int cOff, float val) {
                        blockC[(col + cOff) * M + (row + rOff)] = __float2half(val);
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

static CUtensorMap *globalTmaMapA = nullptr;
static CUtensorMap *globalTmaMapB = nullptr;
static int prevM = 0; static int prevN = 0; static int prevK = 0;

void launchKernel(half *A, half *B, half *C, int M, int N, int K) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int QSIZE = 5;
    constexpr int NUM_THREADS = 256;
    using Shape = BlockShape<BM, BN, BK, QSIZE, NUM_THREADS>;

    if (!globalTmaMapA || M != prevM || N != prevN || K != prevK) {
        if (globalTmaMapA)
            cudaFree(globalTmaMapA);
        if (globalTmaMapB)
            cudaFree(globalTmaMapB);

        globalTmaMapA =
            createTmaMapDevice<Shape::TileM, Shape::TileK>(A, M / Shape::TileM, K / Shape::TileK);
        globalTmaMapB =
            createTmaMapDevice<Shape::TileN, Shape::TileK>(B, N / Shape::TileN, K / Shape::TileK);

        prevM = M;
        prevN = N;
        prevK = K;
    }

    constexpr size_t sharedBytes = size_t(Shape::TileM * Shape::TileK * Shape::QueueSize +
                                          Shape::TileK * Shape::TileN * Shape::QueueSize) *
                                   sizeof(half);

    using KernelT = decltype(&gemm<Shape>);
    KernelT kernelPtr = &gemm<Shape>;

    cudaError_t result = cudaFuncSetAttribute(
        kernelPtr, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sharedBytes));
    assert(result == cudaSuccess);

    dim3 grid((M / Shape::TileM) * (N / Shape::TileN));
    dim3 block(Shape::NumThreads);

    gemm<Shape><<<grid, block, sharedBytes>>>(M, N, K, C, globalTmaMapA, globalTmaMapB);
}

} // namespace gemm_8192x8192_v4

using gemm_8192x8192_v4::launchKernel;