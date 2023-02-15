
template<typename T>
__device__ T load_uncached(const T* addr)
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
    const IndexType row = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (row < num_rows)
    {
        ValueType sum = 0;
        IndexType offset = row;

        for(IndexType n = 0; n < num_cols_per_row; n++)
        {
            const IndexType col = load_uncached(Aj + offset);

            // This assumes that
            // cusp::ell_matrix<...>::invalid_index is always < 0
            if (col >= 0)
            {
                const ValueType x_j = x[col];
                const ValueType A_ij = load_uncached(Ax + offset);

                sum += A_ij * x_j;
            }

            offset += pitch;
        }

        y[row] = sum;
    }
}
