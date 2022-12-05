#pragma once


#if REGISTER_PREFETCH_FACTOR > 0
#define PREFETCH_FACTOR REGISTER_PREFETCH_FACTOR
#elif SHARED_PREFETCH_FACTOR > 0
#define PREFETCH_FACTOR SHARED_PREFETCH_FACTOR
#else
#define PREFETCH_FACTOR 0
#endif


template<typename T>
__device__ T min(T a, T b) {
    return a < b ? a : b;
}

template<typename T>
__device__ T max(T a, T b) {
    return a > b ? a : b;
}


template<typename T>
__device__ T load_diag_val(const T* val)
{
#if LOAD_TYPE == 0
    return *val;
#elif LOAD_TYPE == 1
    // Cache streaming, likely to be accessed once.
    // The ld.cs load cached streaming operation allocates global lines with evict-first policy in L1 and L2
    // to limit cache pollution by temporary streaming data
    return __ldcs(val);
#endif
}

template<typename T>
__device__ T load_x_val(const T* val)
{
#if LOAD_TYPE == 0
    return *val;
#else
    // Needs compute capability at least 3.5
    return __ldg(val);
#endif
}


template<typename IndexType,
         typename ValueType1,
         typename ValueType2,
         int K>
struct UnrolledLoop : UnrolledLoop<IndexType, ValueType1, ValueType2, K - 1>
{
    using parent = UnrolledLoop<IndexType, ValueType1, ValueType2, K - 1>;

    IndexType col;
    ValueType1 diag_val = 0;
    ValueType2 x_val = 0;

    __device__ void prefetch(const IndexType* offsets,
                             const ValueType1* values,
                             const ValueType2* x,
                             int pitch,
                             int cols,
                             IndexType row,
                             IndexType offset_base,
                             IndexType i)
    {
        col = offsets[i] + row;

        if (col >= 0 && col < cols)
        {
            diag_val = load_diag_val(&values[(offset_base + i)*pitch + row]);
            x_val = load_x_val(&x[col]);
        }

        parent::prefetch(offsets, values, x, pitch, cols, row, offset_base, i + 1);
    }

    template<typename ValueType3>
    __device__ ValueType3 do_iter()
    {
        return static_cast<ValueType3>(diag_val*x_val) + parent::template do_iter<ValueType3>();
    }
};

template<typename IndexType,
         typename ValueType1,
         typename ValueType2>
struct UnrolledLoop<IndexType, ValueType1, ValueType2, 0>
{
    __device__ void prefetch(const IndexType* offsets,
                             const ValueType1* values,
                             const ValueType2* x,
                             int pitch,
                             int cols,
                             IndexType row,
                             IndexType offset_base,
                             IndexType i) {}

    template<typename ValueType3>
    __device__ ValueType3 do_iter()
    {
        return 0;
    }
};


template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3,
          unsigned int BLOCK_SIZE>
__device__ void
naive_dia_kernel(const int num_rows,
                 const int num_cols,
                 const int num_diagonals,
                 const int pitch,
                 const IndexType* diagonal_offsets,
                 const ValueType1* values,
                 const ValueType2* x,
                 ValueType3* y)
{
    const IndexType thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (thread_id < num_rows)
    {
        ValueType3 sum = ValueType3(0);
        for (IndexType i = 0; i < num_diagonals; ++i)
        {
            IndexType col = diagonal_offsets[i] + thread_id;
            if (col >= 0 && col < num_cols)
            {
                auto diag_val = load_diag_val(&values[i*pitch + thread_id]);
                auto x_val = load_x_val(&x[col]);

                sum += diag_val * x_val;
            }
        }
        y[thread_id] = sum;
    }
}

#define REGISTER_PREFETCH_ITERATION(iter) \
    IndexType col##iter       = offsets[i + iter] + thread_id; \
    ValueType1 diag_val##iter = 0; \
    ValueType2 x_val##iter    = 0; \
    if (col##iter >= 0 && col##iter < num_cols) \
    { \
        diag_val##iter = load_diag_val(&values[ (offset_base + i + iter)*pitch + thread_id ]); \
        x_val##iter    = load_x_val(&x[col##iter]); \
    }

template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3,
          unsigned int BLOCK_SIZE>
__device__ void
blocked_offsets_dia_kernel(const int num_rows,
                           const int num_cols,
                           const int num_diagonals,
                           const int pitch,
                           const IndexType* diagonal_offsets,
                           const ValueType1* values,
                           const ValueType2* x,
                           ValueType3* y)
{
    const IndexType thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x;

#if SHARED_PREFETCH_FACTOR > 0
    __shared__ ValueType1 values_prefetched[SHARED_PREFETCH_FACTOR*BLOCK_SIZE];
    __shared__ ValueType2 x_prefetched[SHARED_PREFETCH_FACTOR*BLOCK_SIZE];
#endif

    __shared__ IndexType offsets[BLOCK_SIZE];
    ValueType3 sum = ValueType3(0);

    for (int offset_base = 0; offset_base < num_diagonals; offset_base += BLOCK_SIZE)
    {
        if (offset_base + threadIdx.x < num_diagonals)
        {
            offsets[threadIdx.x] = diagonal_offsets[offset_base + threadIdx.x];
        }

        __syncthreads();

        int batch_size = BLOCK_SIZE > num_diagonals - offset_base ? num_diagonals - offset_base : BLOCK_SIZE;

        if (thread_id < num_rows)
        {
#if SHARED_PREFETCH_FACTOR == 0 && REGISTER_PREFETCH_FACTOR == 0
            for (IndexType i = 0; i < batch_size; ++i)
            {
                IndexType col = offsets[i] + thread_id;
                if (col >= 0 && col < num_cols)
                {
                    auto diag_val = load_diag_val(&values[ (offset_base + i)*pitch + thread_id ]);
                    auto x_val = load_x_val(&x[col]);

                    sum += diag_val*x_val;
                }
            }
#else // SHARED_PREFETCH_FACTOR == 0 && REGISTER_PREFETCH_FACTOR == 0
            int end = batch_size - (batch_size % PREFETCH_FACTOR);

            for (IndexType i = 0; i < end; i += PREFETCH_FACTOR)
            {
#if REGISTER_PREFETCH_FACTOR > 0
#if REGISTER_PREFETCH_TYPE == 0

                UnrolledLoop<IndexType, ValueType1, ValueType2, REGISTER_PREFETCH_FACTOR> loop;
                loop.prefetch(offsets, values, x, pitch, num_cols, thread_id, offset_base, i);
                sum += loop.template do_iter<ValueType3>();

#elif REGISTER_PREFETCH_TYPE == 1

                REGISTER_PREFETCH_ITERATION(0);
                REGISTER_PREFETCH_ITERATION(1);
#if REGISTER_PREFETCH_FACTOR > 2
                REGISTER_PREFETCH_ITERATION(2);
#endif
#if REGISTER_PREFETCH_FACTOR > 3
                REGISTER_PREFETCH_ITERATION(3);
#endif

                sum += diag_val0 * x_val0;
                sum += diag_val1 * x_val1;
#if REGISTER_PREFETCH_FACTOR > 2
                sum += diag_val2 * x_val2;
#endif
#if REGISTER_PREFETCH_FACTOR > 3
                sum += diag_val3 * x_val3;
#endif

#endif // REGISTER_PREFETCH_TYPE == 1
#elif SHARED_PREFETCH_FACTOR > 0

                for (int j = 0; j < SHARED_PREFETCH_FACTOR; ++j)
                {
                    IndexType col = offsets[i + j] + thread_id;
                    if (col >= 0 && col < num_cols)
                    {
                        auto diag_val = load_diag_val(&values[ (offset_base + i + j)*pitch + thread_id ]);
                        auto x_val = load_x_val(&x[col]);

                        values_prefetched[j*BLOCK_SIZE + threadIdx.x] = diag_val;
                        x_prefetched[j*BLOCK_SIZE + threadIdx.x] = x_val;
                    }
                    else
                    {
                        values_prefetched[j*BLOCK_SIZE + threadIdx.x] = ValueType1(0);
                        x_prefetched[j*BLOCK_SIZE + threadIdx.x] = ValueType2(0);
                    }
                }

                for (int j = 0; j < SHARED_PREFETCH_FACTOR; ++j)
                {
                    sum += values_prefetched[j*BLOCK_SIZE + threadIdx.x] * x_prefetched[j*BLOCK_SIZE + threadIdx.x];
                }
#endif // SHARED_PREFETCH_FACTOR > 0
            }

            for (IndexType i = end; i < batch_size; ++i)
            {
                IndexType col = offsets[i] + thread_id;
                if (col >= 0 && col < num_cols)
                {
                    auto diag_val = load_diag_val(&values[ (offset_base + i)*pitch + thread_id ]);
                    auto x_val = load_x_val(&x[col]);

                    sum += diag_val*x_val;
                }
            }
#endif // SHARED_PREFETCH_FACTOR == 0 && REGISTER_PREFETCH_FACTOR == 0
        }

        __syncthreads();
    }

    if (thread_id < num_rows)
    {
        y[thread_id] = sum;
    }
}

template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3,
          unsigned int BLOCK_SIZE>
__device__ void
cached_x_dia_kernel(const int num_rows,
                    const int num_cols,
                    const int num_diagonals,
                    const int pitch,
                    const IndexType* diagonal_offsets,
                    const ValueType1* values,
                    const ValueType2* x,
                    ValueType3* y)
{
    using ValueType = ValueType3;

    const IndexType thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    __shared__ IndexType offsets[BLOCK_SIZE];
    __shared__ ValueType x_cache[BLOCK_SIZE];
    IndexType cache_start = -IndexType(BLOCK_SIZE);
    IndexType cache_end = 0;

    IndexType first_row = BLOCK_SIZE * blockIdx.x;
    IndexType end_row = min(first_row + BLOCK_SIZE, num_rows);

    ValueType sum = ValueType(0);

    for (int offset_base = 0; offset_base < num_diagonals; offset_base += BLOCK_SIZE) {
        if (offset_base + threadIdx.x < num_diagonals) {
            offsets[threadIdx.x] = diagonal_offsets[offset_base + threadIdx.x];
        }

        __syncthreads();

        int batch_size = BLOCK_SIZE > num_diagonals - offset_base ? num_diagonals - offset_base : BLOCK_SIZE;

        if (thread_id < num_rows)
        {
            for (IndexType i = 0; i < batch_size; ++i)
            {
                IndexType col = offsets[i] + thread_id;
                ValueType x_val = -1;

                if (col >= 0 && col < num_cols)
                {
                    x_val = col < cache_end ? x_cache[col - cache_start] : x[col];
                    ValueType diag_val = values[(offset_base + i)*pitch + thread_id];
                    sum += x_val * diag_val;
                }

                __syncthreads();

                x_cache[threadIdx.x] = x_val;
                cache_start = first_row + offsets[i];
                cache_end = end_row + offsets[i];

                __syncthreads();
            }
        }

        __syncthreads();
    }

    if (thread_id < num_rows) {
        y[thread_id] = sum;
    }
}

template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3,
          unsigned int BLOCK_SIZE>
__device__ void
experimental_dia_kernel(const int num_rows,
                    const int num_cols,
                    const int num_diagonals,
                    const int pitch,
                    const IndexType* diagonal_offsets,
                    const ValueType1* values,
                    const ValueType2* x,
                    ValueType3* y)
{
    using ValueType = ValueType3;

    const IndexType thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    __shared__ IndexType offsets[BLOCK_SIZE];
    __shared__ ValueType x_cache[2*BLOCK_SIZE];

    IndexType first_row = BLOCK_SIZE * blockIdx.x;
    IndexType first_col = offsets[0] + first_row;

    if (0 <= first_col + threadIdx.x && first_col + threadIdx.x < num_cols) {
        x_cache[threadIdx.x] = x[first_col + threadIdx.x];
    }

    if (0 <= first_col + BLOCK_SIZE + threadIdx.x && first_col + BLOCK_SIZE + threadIdx.x < num_cols) {
        x_cache[BLOCK_SIZE + threadIdx.x] = x[first_col + BLOCK_SIZE + threadIdx.x];
    }

    ValueType sum = ValueType(0);

    for (int offset_base = 0; offset_base < num_diagonals; offset_base += BLOCK_SIZE) {
        if (offset_base + threadIdx.x < num_diagonals) {
            offsets[threadIdx.x] = diagonal_offsets[offset_base + threadIdx.x];
        }

        __syncthreads();

        int batch_size = BLOCK_SIZE > num_diagonals - offset_base ? num_diagonals - offset_base : BLOCK_SIZE;

        if (thread_id < num_rows)
        {
            for (IndexType i = 0; i < batch_size; ++i)
            {
                IndexType col = offsets[i] + thread_id;

                if (col >= 0 && col < num_cols)
                {
                    ValueType x_val = x_cache[col - first_col];
                    ValueType diag_val = values[(offset_base + i)*pitch + thread_id];
                    sum += x_val * diag_val;
                }
            }
        }

        __syncthreads();
    }

    if (thread_id < num_rows) {
        y[thread_id] = sum;
    }
}

template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3,
          unsigned int BLOCK_SIZE>
__device__ void
cached_x_naive_dia_kernel(const int num_rows,
                 const int num_cols,
                 const int num_diagonals,
                 const int pitch,
                 const IndexType* diagonal_offsets,
                 const ValueType1* values,
                 const ValueType2* x,
                 ValueType3* y)
{
    using ValueType = ValueType3;

    const IndexType thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    __shared__ ValueType x_cache[2*BLOCK_SIZE];

    IndexType first_row = BLOCK_SIZE * blockIdx.x;
    IndexType first_col = diagonal_offsets[0] + first_row;

    if (0 <= first_col + threadIdx.x && first_col + threadIdx.x < num_cols) {
        x_cache[threadIdx.x] = x[first_col + threadIdx.x];
    }

    if (0 <= first_col + BLOCK_SIZE + threadIdx.x && first_col + BLOCK_SIZE + threadIdx.x < num_cols) {
        x_cache[BLOCK_SIZE + threadIdx.x] = x[first_col + BLOCK_SIZE + threadIdx.x];
    }


    if (thread_id < num_rows) {
        ValueType sum = ValueType(0);
        for (IndexType i = 0; i < num_diagonals; ++i) {
            IndexType col = diagonal_offsets[i] + thread_id;
            if (col >= 0 && col < num_cols) {
                auto diag_val = values[i*pitch + thread_id];
                auto x_val = x_cache[col - first_col];
                sum += diag_val * x_val;
            }
        }
        y[thread_id] = sum;
    }
}

template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3,
          unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE,1) __global__ void
ktt_dia_vector_kernel(
                const int num_rows,
                const int num_cols,
                const int num_diagonals,
                const int pitch,
                const IndexType* diagonal_offsets,
                const ValueType1* values,
                const ValueType2* x,
                ValueType3* y)
{
#if KERNEL_TYPE == 0
    naive_dia_kernel<IndexType, ValueType1, ValueType2, ValueType3, BLOCK_SIZE>
        (num_rows, num_cols, num_diagonals, pitch, diagonal_offsets, values, x, y);
#elif KERNEL_TYPE == 1
    blocked_offsets_dia_kernel<IndexType, ValueType1, ValueType2, ValueType3, BLOCK_SIZE>
        (num_rows, num_cols, num_diagonals, pitch, diagonal_offsets, values, x, y);
#elif KERNEL_TYPE == 2
    cached_x_dia_kernel<IndexType, ValueType1, ValueType2, ValueType3, BLOCK_SIZE>
        (num_rows, num_cols, num_diagonals, pitch, diagonal_offsets, values, x, y);
#elif KERNEL_TYPE == 3
    experimental_dia_kernel<IndexType, ValueType1, ValueType2, ValueType3, BLOCK_SIZE>
        (num_rows, num_cols, num_diagonals, pitch, diagonal_offsets, values, x, y);
// #elif KERNEL_TYPE == 4
//     cached_x_naive_dia_kernel<IndexType, ValueType1, ValueType2, ValueType3, BLOCK_SIZE>
//         (num_rows, num_cols, num_diagonals, pitch, diagonal_offsets, values, x, y);
#endif
}
