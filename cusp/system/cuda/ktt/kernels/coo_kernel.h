
template<typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void coo_kernel(const Idx* __restrict__ row_indices,
                const Idx* __restrict__ column_indices,
                const Val1* __restrict__ values,
                const int num_entries,
                const Val2* __restrict__ x,
                Val3* __restrict__ y,
                const int vec_size)
{
    // if (BLOCK_SIZE * blockIdx.x + threadIdx.x == 0)
    //     printf("... ktt coo_kernel ...\n");

    // const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    // if (idx == 0)
    // {
    //     for (Idx n = 0; n < num_entries; n++)
    //     {
    //         y[row_indices[n]] += values[n] * x[column_indices[n]];
    //     }
    // }

    // set output vector to all zeroes
    const unsigned ti = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const unsigned space = BLOCK_SIZE * gridDim.x;
    for (unsigned i = ti; i < vec_size; i += space)
    {
        y[ i ] = 0;
    }

#if VALUES_PER_THREAD == 1

    const int n = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    Val1 value;
    if (n < num_entries)
    {
        value = values[n] * x[column_indices[n]];
        auto* ptr = &y[row_indices[n]];
        atomicAdd(ptr, value);
    }

#elif VALUES_PER_THREAD != 500

    const int idx = ti;
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

#else

    // const unsigned ti = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const unsigned idx_in_blk = threadIdx.x;

    __shared__ Idx  sh_rows[ BLOCK_SIZE + 2 + 10 ];
    __shared__ Val1 sh_vals[ BLOCK_SIZE + 2 + 10 ];

    if (ti < num_entries)
    {
        if (idx_in_blk == 0 || idx_in_blk == BLOCK_SIZE-1)
        {
            sh_rows[ idx_in_blk ] = -1;
            sh_vals[ idx_in_blk ] = 0;
            sh_rows[ idx_in_blk + 2 ] = -1;
            sh_vals[ idx_in_blk + 2 ] = 0;
        }

        sh_rows[ idx_in_blk + 1 ] = row_indices[ ti ];
        sh_vals[ idx_in_blk + 1 ] = values[ ti ] * x[ column_indices[ ti ] ];

        // __syncthreads();
        __threadfence_block();

        Idx prv_row = sh_rows[ idx_in_blk ];
        Idx cur_row = sh_rows[ idx_in_blk + 1 ];
        Idx nxt_row = sh_rows[ idx_in_blk + 2 ];

        if (prv_row != cur_row && cur_row != nxt_row)
        {
            y[ cur_row ] = sh_vals[ idx_in_blk + 1 ];
        }
        else if (prv_row != cur_row && cur_row == nxt_row)
        {
            Val1 sum = 0;
            int i = idx_in_blk + 1;
            for (; sh_rows[ i ] == cur_row && i < BLOCK_SIZE; ++i)
            {
                sum += sh_vals[ i ];
            }

            if (i == BLOCK_SIZE - 1 || idx_in_blk == 0)
            {
                // atomic add
                atomicAdd(&y[ cur_row ], sum);
            }
            else
            {
                y[ cur_row ] = sum;
            }
        }

        // value = values[n] * x[column_indices[n]];
        // auto* ptr = &y[row_indices[n]];

        // y[ sh_rows[ idx_in_blk + 1 ] ] = sh_vals[ idx_in_blk + 1 ];
    }

#endif
}

template<typename Idx, typename Val1, typename Val2, typename Val3>
__global__
void coo_spmv(const Idx* __restrict__ row_indices,
               const Idx* __restrict__ column_indices,
               const Val1* __restrict__ values,
               const int num_entries,
               const Val2* __restrict__ x,
               Val3* __restrict__ y,
               const int vec_size)
{
    coo_kernel<Idx, Val1, Val2, Val3>
              (row_indices, column_indices, values, num_entries, x, y, vec_size);
}
