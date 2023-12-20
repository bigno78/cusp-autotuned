
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

    const int n = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    Val1 value;
    if (n < num_entries)
    {
        value = values[n] * x[column_indices[n]];
        auto* ptr = &y[row_indices[n]];
        atomicAdd(ptr, value);
    }
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
