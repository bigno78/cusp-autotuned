
#if DISABLE_UNROLL == 1
#define UNROLL 1
#else
#define UNROLL
#endif


template<typename T>
__device__ T load(const T* addr)
{
#if UNCACHED_LOADS == 0
    return *addr;
#elif UNCACHED_LOADS == 1
    return __ldcv(addr);
#elif UNCACHED_LOADS == 2
    return __ldcs(addr);
#elif UNCACHED_LOADS == 3
    return __ldlu(addr);
#endif
}


template<typename IndexType,
         typename ValueType,
         int K>
struct Prefetcher : Prefetcher<IndexType, ValueType,  K - 1>
{
    using parent = Prefetcher<IndexType, ValueType, K - 1>;

    IndexType col;
    ValueType A_val = 0;
    ValueType x_val = 0;

    __device__
    void prefetch_cols(const IndexType* __restrict__ Aj,
                      IndexType offset,
                      IndexType pitch,
                      IndexType i)
    {
        col = load(Aj + offset + i*pitch);
        parent::prefetch_cols(Aj, offset, pitch, i + 1);
    }

    __device__
    void prefetch_vals(const ValueType* __restrict__ Ax,
                       const ValueType* __restrict__ x,
                       IndexType offset,
                       IndexType pitch,
                       IndexType i)
    {
        if (col >= 0)
        {
            A_val = load(Ax + offset + i*pitch);
            x_val = x[col];
        }
        parent::prefetch_vals(Ax, x, offset, pitch, i + 1);
    }

    __device__ ValueType accumulate_results()
    {
        return x_val*A_val + parent::accumulate_results();
    }
};

template<typename IndexType,
         typename ValueType>
struct Prefetcher<IndexType, ValueType, 0>
{
    __device__
    void prefetch_cols(const IndexType* __restrict__ Aj, IndexType offset,
                       IndexType pitch, IndexType i) { }

    __device__
    void prefetch_vals(const ValueType* __restrict__ Ax,
                       const ValueType* __restrict__ x, IndexType offset,
                       IndexType pitch, IndexType i) { }

    __device__ ValueType accumulate_results()
    {
        return 0;
    }
};


template <typename IndexType,
          typename ValueType>
__device__ void
ktt_ell_kernel_basic(const IndexType num_rows,
                     const IndexType num_cols,
                     const IndexType num_cols_per_row,
                     const IndexType pitch,
                     const IndexType* __restrict__ Aj,
                     const ValueType* __restrict__ Ax,
                     const ValueType* __restrict__ x,
                     ValueType* __restrict__ y)
{

#if THREADS_PER_ROW > 1
    const IndexType thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const IndexType warp_id = thread_id/32;
    const IndexType stripe_id = warp_id/THREADS_PER_ROW;
    const IndexType warp_lane = thread_id % 32;
    const IndexType stripe_lane = warp_id % THREADS_PER_ROW;
    const IndexType block_lane = threadIdx.x/(32*THREADS_PER_ROW);

    const IndexType row = stripe_id*32 + warp_lane;
    const IndexType start_offset = row + stripe_lane * pitch;
    const IndexType stride = THREADS_PER_ROW * pitch;

    const IndexType rounded_up_count =
        (num_cols_per_row + THREADS_PER_ROW - 1)/THREADS_PER_ROW;
    const IndexType remainder = num_cols_per_row % THREADS_PER_ROW;

    const IndexType elem_count =
        remainder == 0 || stripe_lane < remainder
            ? rounded_up_count
            : rounded_up_count - 1;

    //if (row == 1)
    //    printf("%d -> %d %d %d %d\n", thread_id, rounded_up_count, remainder, stripe_lane, elem_count);

    const int stripes_in_block = BLOCK_SIZE/(32*THREADS_PER_ROW);
    __shared__ ValueType sums[stripes_in_block][32];

    if (stripe_lane == 0)
    {
        sums[block_lane][warp_lane] = 0;
    }
    __syncthreads();
#else
    const IndexType row = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const IndexType start_offset = row;
    const IndexType stride = pitch;
    const IndexType elem_count = num_cols_per_row;
#endif

    if (row < num_rows)
    {
        ValueType sum = 0;
        IndexType offset = start_offset;

#if PREFETCH_FACTOR > 0
        IndexType num_iters = elem_count/PREFETCH_FACTOR;

        #pragma unroll UNROLL
        for(IndexType n = 0; n < num_iters; ++n)
        {
            Prefetcher<IndexType, ValueType, PREFETCH_FACTOR> prefetcher;

            prefetcher.prefetch_cols(Aj, offset, stride, 0);
            prefetcher.prefetch_vals(Ax, x, offset, stride, 0);
            sum += prefetcher.accumulate_results();

            offset += PREFETCH_FACTOR * stride;
        }

        IndexType remaining_iters = elem_count % PREFETCH_FACTOR;
#else
        IndexType remaining_iters = elem_count;
#endif

        #pragma unroll UNROLL
        for (int n = 0; n < remaining_iters; ++n)
        {
            const IndexType col = load(Aj + offset);

            // This assumes that
            // cusp::ell_matrix<...>::invalid_index is always < 0
            if (col >= 0)
            {
                const ValueType x_j = x[col];
                const ValueType A_ij = load(Ax + offset);

                sum += A_ij * x_j;
            }
            else
            {
#if BREAK == 1
                break;
#endif
            }

            offset += stride;
        }

#if THREADS_PER_ROW > 1
        atomicAdd(&sums[block_lane][warp_lane], sum);

        __syncthreads();

        if (stripe_lane == 0)
        {
            y[row] = sums[block_lane][warp_lane];
        }
#else
        y[row] = sum;
#endif
    }
}


template <typename IndexType,
          typename ValueType>
__launch_bounds__(BLOCK_SIZE, 1) __global__ void
ktt_ell_kernel(const IndexType num_rows,
               const IndexType num_cols,
               const IndexType num_cols_per_row,
               const IndexType pitch,
               const IndexType* __restrict__ Aj,
               const ValueType* __restrict__ Ax,
               const ValueType* __restrict__ x,
               ValueType* __restrict__ y)
{
    ktt_ell_kernel_basic(num_rows, num_cols, num_cols_per_row, pitch,
                         Aj, Ax, x, y);
}
