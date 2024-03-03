
template<typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void coo_kernel(const Idx* __restrict__ row_indices,
                const Idx* __restrict__ column_indices,
                const Val1* __restrict__ values,
                const int num_entries,
                const Val2* __restrict__ x,
                Val3* __restrict__ y)
{
    // const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    // if (idx == 0)
    // {
    //     for (Idx n = 0; n < num_entries; n++)
    //     {
    //         y[row_indices[n]] += values[n] * x[column_indices[n]];
    //     }
    // }

#if VALUES_PER_THREAD == 1

    const int n = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    Val1 value;
    if (n < num_entries)
    {
        value = values[n] * x[column_indices[n]];
        auto* ptr = &y[row_indices[n]];
        atomicAdd(ptr, value);
    }

#else

    const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int begin = idx * VALUES_PER_THREAD;

    const int end = min(num_entries, begin + VALUES_PER_THREAD);

    // no work left for this thread
    if (begin >= end)
        return;

    Val1 value = 0;
    Idx row = row_indices[ begin ];

    for (int i = begin; i < end; ++i)
    {
        Idx cur = row_indices[ i ];

        if (row != cur)
        {
            auto* ptr = &y[ row ];
            atomicAdd(ptr, value);
            value = 0;
        }

        value += values[ i ] * x[ column_indices[ i ] ];
        row = cur;
    }

    auto* ptr = &y[ row ];
    atomicAdd(ptr, value);

#endif
}

template<typename Idx, typename Val1, typename Val2, typename Val3>
__global__
void coo_spmv(const Idx* __restrict__ row_indices,
               const Idx* __restrict__ column_indices,
               const Val1* __restrict__ values,
               const int num_entries,
               const Val2* __restrict__ x,
               Val3* __restrict__ y)
{
    coo_kernel<Idx, Val1, Val2, Val3>
              (row_indices, column_indices, values, num_entries, x, y);
}
