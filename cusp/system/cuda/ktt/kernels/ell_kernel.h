
template<typename T>
__device__ T load_uncached(const T* addr)
{
#if UNCACHED_LOADS == 0
    return *addr;
#else
    return __ldcv(addr);
#endif
}

template <typename IndexType,
          typename ValueType>
__launch_bounds__(BLOCK_SIZE, 1) __global__ void
ktt_ell_kernel(const IndexType num_rows,
                const IndexType num_cols,
                const IndexType num_cols_per_row,
                const IndexType pitch,
                const IndexType * Aj,
                const ValueType * Ax,
                const ValueType * x,
                ValueType * y)
{
    const IndexType row = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    // if (row == 0)
    // {
    //     printf("Hello from ELL.\n");
    // }

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
