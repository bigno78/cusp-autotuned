
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
    const IndexType row = blockDim.x*blockIdx.x + threadIdx.x;
    const IndexType stride = THREADS_PER_ROW * pitch;

#if THREADS_PER_ROW > 1
    __shared__ ValueType sums[BLOCK_SIZE/THREADS_PER_ROW];

    if (threadIdx.y == 0)
    {
        sums[threadIdx.x] = 0;
    }

    __syncthreads();
#endif

    if (row < num_rows)
    {
        ValueType sum = 0;
        IndexType n = threadIdx.y;
        IndexType offset = row + n * pitch;

#if PREFETCH_FACTOR > 0
        IndexType bound = num_cols_per_row
                            - (PREFETCH_FACTOR - 1)*THREADS_PER_ROW;
        IndexType step = PREFETCH_FACTOR * THREADS_PER_ROW;

        #pragma unroll UNROLL
        for(; n < bound; n += step)
        {
            Prefetcher<IndexType, ValueType, PREFETCH_FACTOR> prefetcher;

            prefetcher.prefetch_cols(Aj, offset, stride, 0);
            prefetcher.prefetch_vals(Ax, x, offset, stride, 0);
            sum += prefetcher.accumulate_results();

            offset += PREFETCH_FACTOR * stride;
        }
#endif

        #pragma unroll UNROLL
        for (; n < num_cols_per_row; n += THREADS_PER_ROW)
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
        atomicAdd(&sums[threadIdx.x], sum);

        __syncthreads();

        if (threadIdx.y == 0)
        {
            y[row] = sums[threadIdx.x];
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
