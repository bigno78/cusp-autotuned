

__device__ int* row_counter;


// Modulo operation assuming mod is a value of this form: mod = 2^exp.
template<typename T, typename S>
__device__
inline auto mod2exp(T v, S mod)
{
    return v & (mod - 1);
}


__device__
int assign_row(const int prev_row, const int worker_idx, const int total_workers)
{
    const int lane = mod2exp(threadIdx.x, 32);

    if (prev_row == -1)
        return worker_idx;

#if THREADS_PER_ROW == 32
    int row = 0;
    if (lane == 0) row = total_workers + atomicAdd(row_counter, 1);

    constexpr unsigned mask = 0xffffffff;
    int got = __shfl_sync(mask, row, 0);
    return got;

#elif THREADS_PER_ROW == 1

    int row = 0;
    if (lane == 0) row = total_workers + atomicAdd(row_counter, 32);

    constexpr unsigned mask = 0xffffffff;
    int got = lane + __shfl_sync(mask, row, 0);
    return got;

#elif THREADS_PER_ROW == 0

    const int idx_in_blk = threadIdx.x;
    __shared__ int sh_row;

    if (idx_in_blk == 0) sh_row = total_workers + atomicAdd(row_counter, 1);
    __syncthreads();

    return sh_row;

#else
    int row = 0;
    if (lane == 0) row = total_workers + atomicAdd(row_counter, 32 / THREADS_PER_ROW);

    constexpr unsigned mask = 0xffffffff;
    int got = __shfl_sync(mask, row, 0);

    return got + lane / THREADS_PER_ROW;

#endif

}



template<typename T>
__device__
T load_no_cache(const T* val)
{
#if SPECIAL_LOADS != 0
    return __ldcs(val);
#else
    return *val;
#endif
}

template<typename T>
__device__
T load_cache(const T* val)
{
#if SPECIAL_LOADS != 0
    return __ldg(val);
#else
    return *val;
#endif
}


template<int Total,
         typename Val1,
         typename Val2,
         typename Idx>
__device__
Val2 accumulate(Val2 value, const Idx lane, const Idx row_begin, const Idx row_end,
                const Idx*   __restrict__ Ac,
                const Val1*  __restrict__ Ax,
                const Val2*  __restrict__ x)
{
    auto get = [&](Idx i)
    {
        auto mat = load_no_cache(Ax + i);
        auto col = load_no_cache(Ac + i);
        auto val = load_cache(x + col);
        return mat * val;
    };

#if ALIGNED == 1
    if (Total % 32 == 0 && row_end - row_begin > 32)
    {
        Idx aligned_start = row_begin - mod2exp(row_begin, 32) + lane;

        if (int i = aligned_start; i >= row_begin && i < row_end)
            value += get(i);

        for (int i = aligned_start + Total; i < row_end; i += Total)
            value += get(i);
    }
    else
#endif
    {
        for (int i = row_begin + lane; i < row_end; i += Total)
            // value += Ax[ i ] * x[ Ac[ i ] ];
            value += get(i);
    }

    return value;
}



template<typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void csr_kernel_naive(const unsigned int num_rows,
                const Idx*   __restrict__ Ar,
                const Idx*   __restrict__ Ac,
                const Val1*  __restrict__ Ax,
                const Val2*  __restrict__ x,
                Val3*        __restrict__ y)
{
    int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int total_threads = BLOCK_SIZE * gridDim.x;
    const Idx idx_in_blk = threadIdx.x;

    Val1 sum = 0;
#if DYNAMIC != 0
    int row = -1;
    while ( ( row = assign_row(row, idx, total_threads) ) < num_rows )
#else
    for (Idx row = idx; row < num_rows; row += total_threads)
#endif
    {
        sum = 0;
        // TODO: read this coalesed with the whole block, might be faster
        Idx row_start = Ar[row];
        Idx row_end = Ar[row + 1];

        // TODO: Didn't seem to be faster.
        // constexpr int SHIFT = 17;
        // sh_row_info[ idx_in_blk + idx_in_blk / SHIFT ] = Ar[ row ];
        // if (idx_in_blk == BLOCK_SIZE - 1)
        //     sh_row_info[ BLOCK_SIZE + BLOCK_SIZE / SHIFT ] = Ar[ row + 1 ];
        // __syncthreads();
        // Idx row_start = sh_row_info[ idx_in_blk + idx_in_blk / SHIFT ];
        // Idx row_end   = sh_row_info[ (idx_in_blk + 1) + ((idx_in_blk + 1) / SHIFT) ];

        sum = accumulate<1, Val1, Val2, Idx>(0, 0, row_start, row_end, Ac, Ax, x);
        y[row] = sum;
    }
}


template<typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void csr_kernel_warp(const unsigned int num_rows,
                const Idx*   __restrict__ Ar,
                const Idx*   __restrict__ Ac,
                const Val1*  __restrict__ Ax,
                const Val2*  __restrict__ x,
                Val3*        __restrict__ y)
{
    const int ti         = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int lane       = mod2exp(threadIdx.x, THREADS_PER_ROW);
    const int idx_in_blk = threadIdx.x;
    const int blk_idx    = blockIdx.x;
    const int worker_idx = threadIdx.x / THREADS_PER_ROW;

    const int vectors_per_block = BLOCK_SIZE / THREADS_PER_ROW;
    const int vector_count = gridDim.x * vectors_per_block;

    __shared__ Idx sh_row_info[vectors_per_block][2];

#if DYNAMIC != 0
    int row = -1;
    while ( ( row = assign_row(row, ti / THREADS_PER_ROW, vector_count) ) < num_rows )
#else
    for (Idx row = ti / THREADS_PER_ROW; row < num_rows; row += vector_count)
#endif
    {
    // TODO: This.
    if (lane < 2)
        sh_row_info[ worker_idx ][ lane ] = Ar[ row + lane ];
    __syncthreads();
    Idx row_start = sh_row_info[ worker_idx ][ 0 ];
    Idx row_end   = sh_row_info[ worker_idx ][ 1 ];


    Val3 value = 0;

    value = accumulate<THREADS_PER_ROW, Val1, Val2, Idx>(0, lane, row_start, row_end, Ac, Ax, x);

    constexpr unsigned mask = 0xffffffff;
    #pragma unroll
    for (int off = THREADS_PER_ROW / 2; off >= 1; off /= 2)
        value += __shfl_down_sync(mask, value, off, THREADS_PER_ROW);

    if (lane == 0)
        y[ row ] = value;

    }
}








template<typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void csr_kernel_block(const unsigned int num_rows,
                const Idx*   __restrict__ Ar,
                const Idx*   __restrict__ Ac,
                const Val1*  __restrict__ Ax,
                const Val2*  __restrict__ x,
                Val3*        __restrict__ y)
{
    const int WARP_SIZE = 32;

    const int lane = mod2exp(threadIdx.x, WARP_SIZE);
    const int blk_idx = blockIdx.x;
    const int idx_in_blk = threadIdx.x;
    const int warp_in_block = idx_in_blk / WARP_SIZE;

    const int BLOCK_COUNT = gridDim.x;

    __shared__ Val3 sh_sums[ BLOCK_SIZE / WARP_SIZE ];
    __shared__ Idx sh_row_info[2];

    // TODO: better fetching of row info
    // for (int row = blk_idx * ROWS_PER_BLOCK; row < blk_idx * ROWS_PER_BLOCK + ROWS_PER_BLOCK; ++row)

    int begin = blk_idx;

#if DYNAMIC != 0
    int row = -1;
    while ( ( row = assign_row(row, begin, BLOCK_COUNT) ) < num_rows )
#else
    for (unsigned row = begin; row < num_rows; row += BLOCK_COUNT)
#endif
    {
    // TODO: check if this correct and faster
    if (idx_in_blk < 2)
        sh_row_info[ idx_in_blk ] = Ar[ row + idx_in_blk ];
    __syncthreads();
    const Idx row_start = sh_row_info[ 0 ];
    const Idx row_end   = sh_row_info[ 1 ];


    Val3 value = 0;

    value = accumulate<BLOCK_SIZE, Val1, Val2, Idx>(0, idx_in_blk, row_start, row_end, Ac, Ax, x);

    const unsigned mask = 0xffffffff;
    value += __shfl_down_sync(mask, value, 16);
    value += __shfl_down_sync(mask, value, 8);
    value += __shfl_down_sync(mask, value, 4);
    value += __shfl_down_sync(mask, value, 2);
    value += __shfl_down_sync(mask, value, 1);

    if (lane == 0)
        sh_sums[ warp_in_block ] = value;

    __syncthreads();

    if (idx_in_blk == 0)
    {
        Val1 total_sum = 0;
        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE / WARP_SIZE; ++j)
            total_sum += sh_sums[ j ];
        y[ row ] = total_sum;
    }

    // TODO: these is some bug in this that makes it loop forever
    // const int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
    // if (idx_in_blk < WARPS_PER_BLOCK / 2)
    // {
    //     for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset /= 2)
    //     {
    //         auto my_val = sh_sums[ warp_in_block ];
    //         auto friend_val = sh_sums[ warp_in_block + offset ];
    //         sh_sums[ warp_in_block ] = my_val + friend_val;
    //         __syncthreads();
    //     }
    // }
    // if (idx_in_blk == 0)
    //     y[ row ] = sh_sums[ 0 ];
    }
}


template<typename T, typename U>
__device__
inline auto divide_into(T value, U chunk)
{
    auto div = value / chunk;
    if (value % chunk == 0)
        return div;

    return div + 1;
}



template<typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void csr_kernel_balanced(const unsigned int num_rows,
                         const Idx*   __restrict__ Ar,
                         const Idx*   __restrict__ Ac,
                         const Val1*  __restrict__ Ax,
                         const Val2*  __restrict__ x,
                         Val3*        __restrict__ y,
                         const unsigned int num_entries,
                         const int* __restrict__ row_starts)
{
    const int ti         = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    // global “worker” index (a worker in this case is a warp that processes the given interval)
    const int worker_idx = ti / THREADS_PER_ROW;
    const int lane       = mod2exp(threadIdx.x, THREADS_PER_ROW);

    const int worker_count = gridDim.x * BLOCK_SIZE / THREADS_PER_ROW;
    const int worker_chunk = divide_into(num_entries, worker_count);

    const int begin =       worker_idx      * worker_chunk;
    const int end   = min( (worker_idx + 1) * worker_chunk, num_entries );

    if (begin >= end)
        return;

    int real_row_begin = 0;
    for (int row = row_starts[ worker_idx ]; ( real_row_begin = Ar[ row ] ) < end; ++row)
    {
        const int real_row_end = Ar[ row + 1 ];

        const int row_begin = max( real_row_begin, begin );
        const int row_end   = min( real_row_end, end );

        Val1 value = 0;

        value = accumulate<THREADS_PER_ROW, Val1, Val2, Idx>(0, lane, row_begin, row_end, Ac, Ax, x);

        constexpr unsigned mask = 0xffffffff;
        #pragma unroll
        for (int off = THREADS_PER_ROW / 2; off >= 1; off /= 2)
            value += __shfl_down_sync(mask, value, off, THREADS_PER_ROW);

        if (lane == 0)
        {
#if AVOID_ATOMIC == 1
            // This worker processed the whole row, no need for atomic op.
            if ( (row_begin == real_row_begin) + (row_end == real_row_end) == 2 )
                y[ row ] = value;
            else
#endif
                atomicAdd(&y[ row ], value);
        }
    }
}



template<typename Idx, typename Val1, typename Val2, typename Val3>
__launch_bounds__(BLOCK_SIZE, 1)
__global__
void csr_spmv(const unsigned int num_rows,
              const Idx*   __restrict__ Ar,
              const Idx*   __restrict__ Ac,
              const Val1*  __restrict__ Ax,
              const Val2*  __restrict__ x,
              Val3*        __restrict__ y,
              const unsigned int num_entries,
              int* __restrict__ row_counter,
              const int* __restrict__ row_starts)
{
    ::row_counter = row_counter;

#if DYNAMIC == 2

    csr_kernel_balanced<Idx, Val1, Val2, Val3>(num_rows, Ar, Ac, Ax, x, y, num_entries, row_starts);

#else

    if constexpr (THREADS_PER_ROW == 0)
        csr_kernel_block<Idx, Val1, Val2, Val3>(num_rows, Ar, Ac, Ax, x, y);
    else if constexpr (THREADS_PER_ROW == 1)
        csr_kernel_naive<Idx, Val1, Val2, Val3>(num_rows, Ar, Ac, Ax, x, y);
    else if constexpr (THREADS_PER_ROW <= 32)
        csr_kernel_warp<Idx, Val1, Val2, Val3>(num_rows, Ar, Ac, Ax, x, y);
    else
        printf("invalid THREADS_PER_ROW value\n"), assert(false);

#endif
}
