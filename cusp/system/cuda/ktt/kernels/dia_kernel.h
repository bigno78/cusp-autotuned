
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
#if SPECIAL_LOADS == 0
    return *val;
#elif SPECIAL_LOADS == 1
    // Cache streaming, likely to be accessed once.
    // The ld.cs load cached streaming operation allocates global lines with
    // evict-first policy in L1 and L2
    // to limit cache pollution by temporary streaming data
    return __ldcs(val);
#endif
}

template<typename T>
__device__ T load_x_val(const T* val)
{
#if SPECIAL_LOADS == 0
    return *val;
#elif SPECIAL_LOADS == 1
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

    __device__ void prefetch_vals(const IndexType* offsets,
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

        parent::prefetch_vals(offsets, values, x, pitch, cols, row,
                              offset_base, i + 1);
    }

    template<typename ValueType3>
    __device__ ValueType3 accumulate()
    {
        return static_cast<ValueType3>(diag_val*x_val)
                + parent::template accumulate<ValueType3>();
    }
};

template<typename IndexType,
         typename ValueType1,
         typename ValueType2>
struct UnrolledLoop<IndexType, ValueType1, ValueType2, 0>
{
    __device__ void prefetch_vals(const IndexType* offsets,
                                  const ValueType1* values,
                                  const ValueType2* x,
                                  int pitch,
                                  int cols,
                                  IndexType row,
                                  IndexType offset_base,
                                  IndexType i) {}

    template<typename ValueType3>
    __device__ ValueType3 accumulate() { return 0; }
};


#define PREFETCH_VALS_ITERATION(iter)                                          \
    IndexType col##iter       = offsets[i + iter - 1] + row;             \
    ValueType1 diag_val##iter = 0;                                             \
    ValueType2 x_val##iter    = 0;                                             \
    if (col##iter >= 0 && col##iter < num_cols)                                \
    {                                                                          \
        IndexType idx = (offset_base + i + iter - 1)*pitch + row    ;        \
        diag_val##iter = load_diag_val(&values[idx]);                          \
        x_val##iter    = load_x_val(&x[col##iter]);                            \
    }

#define ACCUMULATE_ITERATION(iter) \
    sum += diag_val##iter * x_val##iter;

#define REPEAT1(ITER) ITER(1);
#define REPEAT2(ITER) ITER(2); REPEAT1(ITER)
#define REPEAT3(ITER) ITER(3); REPEAT2(ITER)
#define REPEAT4(ITER) ITER(4); REPEAT3(ITER)
#define REPEAT5(ITER) ITER(5); REPEAT4(ITER)
#define REPEAT6(ITER) ITER(6); REPEAT5(ITER)
#define REPEAT7(ITER) ITER(7); REPEAT6(ITER)
#define REPEAT8(ITER) ITER(8); REPEAT7(ITER)

#define PREFETCH_VALS_IMPL(factor) REPEAT##factor(PREFETCH_VALS_ITERATION)
#define ACCUMULATE_IMPL(factor) REPEAT##factor(ACCUMULATE_ITERATION)

#define PREFETCH_VALS(factor) PREFETCH_VALS_IMPL(factor)
#define ACCUMULATE(factor) ACCUMULATE_IMPL(factor)


template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3>
__device__ void
blocked_offsets_dia_kernel(const int num_rows,
                           const int num_cols,
                           const int num_diagonals,
                           const int pitch,
                           const IndexType* __restrict__ diagonal_offsets,
                           const ValueType1* __restrict__ values,
                           const ValueType2*__restrict__  x,
                           ValueType3* __restrict__ y)
{
    const IndexType row = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    __shared__ IndexType offsets[BLOCK_SIZE];
    ValueType3 sum = ValueType3(0);

    for (int offset_base = 0; offset_base < num_diagonals; offset_base += BLOCK_SIZE)
    {
        if (offset_base + threadIdx.x < num_diagonals)
            offsets[threadIdx.x] = diagonal_offsets[offset_base + threadIdx.x];

        __syncthreads();

        int batch_size = BLOCK_SIZE > num_diagonals - offset_base
                            ? num_diagonals - offset_base
                            : BLOCK_SIZE;

        if (row < num_rows)
        {
            IndexType i = 0;

#if PREFETCH_FACTOR > 0
            for (; i < batch_size - PREFETCH_FACTOR + 1; i += PREFETCH_FACTOR)
            {
#if PREFETCH_TYPE == 0
                UnrolledLoop<IndexType, ValueType1, ValueType2, PREFETCH_FACTOR> loop;
                loop.prefetch_vals(offsets, values, x, pitch, num_cols,
                                   row, offset_base, i);
                sum += loop.template accumulate<ValueType3>();
#elif PREFETCH_TYPE == 1
                PREFETCH_VALS(PREFETCH_FACTOR);
                ACCUMULATE(PREFETCH_FACTOR);
#endif
            }
#endif

            for (; i < batch_size; ++i)
            {
                IndexType col = offsets[i] + row;
                if (col >= 0 && col < num_cols)
                {
                    IndexType idx = (offset_base + i)*pitch + row;

                    auto diag_val = load_diag_val(&values[idx]);
                    auto x_val = load_x_val(&x[col]);

                    sum += diag_val*x_val;
                }
            }
        }

        __syncthreads();
    }

    if (row < num_rows)
        y[row] = sum;
}


template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3>
__device__ void
naive_dia_kernel(const int num_rows,
                 const int num_cols,
                 const int num_diagonals,
                 const int pitch,
                 const IndexType* __restrict__ diagonal_offsets,
                 const ValueType1* __restrict__ values,
                 const ValueType2* __restrict__ x,
                 ValueType3* __restrict__ y)
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


template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3>
__launch_bounds__(BLOCK_SIZE, 1) __global__ void
ktt_dia_vector_kernel(const int num_rows,
                      const int num_cols,
                      const int num_diagonals,
                      const int pitch,
                      const IndexType* __restrict__ diagonal_offsets,
                      const ValueType1* __restrict__ values,
                      const ValueType2 *__restrict__ x,
                      ValueType3* __restrict__ y)
{
    blocked_offsets_dia_kernel<IndexType, ValueType1, ValueType2, ValueType3>
        (num_rows, num_cols, num_diagonals, pitch, diagonal_offsets, values, x, y);
}
