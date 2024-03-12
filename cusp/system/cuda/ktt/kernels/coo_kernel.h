

template<typename Idx, typename Val1, typename Val2, typename Val3>
__global__
void zero_output(const Idx* __restrict__ row_indices,
                 const Idx* __restrict__ col_indices,
                 const Val1* __restrict__ values,
                 const int num_entries,
                 const Val2* __restrict__ x,
                 Val3* __restrict__ y,
                 const int y_size)
{
    // set output vector to all zeroes
    const unsigned ti = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const unsigned space = BLOCK_SIZE * gridDim.x;
    for (unsigned i = ti; i < y_size; i += space)
        y[ i ] = 0.0f;

    // const unsigned ti = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    // if (ti < y_size)
    //     y[ ti ] = 0.0f;
}



template<typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void naive_coo_kernel(const Idx* __restrict__ row_indices,
                      const Idx* __restrict__ col_indices,
                      const Val1* __restrict__ values,
                      const int num_entries,
                      const Val2* __restrict__ x,
                      Val3* __restrict__ y,
                      const int y_size)
{
    const int n = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    // if (n == 0) printf("... naive_coo_kernel ...\n");
    if (n < num_entries)
    {
        Val1 value = values[n] * x[col_indices[n]];
        auto* ptr = &y[row_indices[n]];
        atomicAdd(ptr, value);
    }
}



template<typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void serial_coo_kernel(const Idx* __restrict__ row_indices,
                      const Idx* __restrict__ col_indices,
                      const Val1* __restrict__ values,
                      const int num_entries,
                      const Val2* __restrict__ x,
                      Val3* __restrict__ y,
                      const int y_size)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx == 0)
    {
        printf("... SLOW naive_coo_kernel ...\n");
        for (Idx n = 0; n < num_entries; n++)
        {
            y[row_indices[n]] += values[n] * x[col_indices[n]];
        }
    }
}


template<typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void naive_multi(const Idx* __restrict__ row_indices,
                 const Idx* __restrict__ col_indices,
                 const Val1* __restrict__ values,
                 const int num_entries,
                 const Val2* __restrict__ x,
                 Val3* __restrict__ y,
                 const int y_size)
{
    const unsigned idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int begin = idx * VALUES_PER_THREAD;
    const int end = min(num_entries, begin + VALUES_PER_THREAD);

    // if (idx == 0) printf("... naive_multi ...\n");

    // no work left for this thread
    if (begin >= end)
        return;

    float value = 0;
    Idx row = row_indices[ begin ];
    // bool first = true;
    for (int i = begin; i < end; ++i)
    {
        Idx cur = row_indices[ i ];
        if (row != cur)
        {
            // if (first)
                atomicAdd(&y[ row ], value);
            // else
            //     y[ row ] = value;
            value = 0;
            // first = false;
        }
        value += values[ i ] * x[ col_indices[ i ] ];
        row = cur;
    }
    auto* ptr = &y[ row ];
    atomicAdd(ptr, value);

    // const unsigned idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    // const int begin = idx * VALUES_PER_THREAD;
    // const int end = min(num_entries, begin + VALUES_PER_THREAD);
    // if (begin >= end)
    //     return;
    // for (int i = begin; i < end; ++i)
    // {
    //     Idx row = row_indices[ i ];
    //     Val1 value = values[ i ] * x[ col_indices[ i ] ];
    //     atomicAdd(&( y[ row ] ), value);
    // }
}



template<typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void naive_multi_shared(const Idx* __restrict__ row_indices,
                        const Idx* __restrict__ col_indices,
                        const Val1* __restrict__ values,
                        const int num_entries,
                        const Val2* __restrict__ x,
                        Val3* __restrict__ y,
                        const int y_size)
{
    const unsigned idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const unsigned idx_in_blk = threadIdx.x;
// if (idx == 0) printf("... naive_multi_shared ...\n");
    const int begin = idx * VALUES_PER_THREAD;
    const int end = min(num_entries, begin + VALUES_PER_THREAD);

    __shared__ Idx  sh_rows[ BLOCK_SIZE * VALUES_PER_THREAD ];
    __shared__ Val1 sh_vals[ BLOCK_SIZE * VALUES_PER_THREAD ];

    if (idx >= num_entries)
        return;

    // no work left for this thread
    if (begin >= end)
        return;

    int it = 0;
    for (int i = begin; i < end; ++i, ++it)
    {
        sh_rows[ idx_in_blk * VALUES_PER_THREAD + it ] = row_indices[ i ];
        sh_vals[ idx_in_blk * VALUES_PER_THREAD + it ] = values[ i ] * x[ col_indices[ i ] ];
    }

    Idx row = sh_rows[ idx_in_blk * VALUES_PER_THREAD ];
    Val1 value = 0;

    for (int i = 0; i < end - begin; ++i)
    {
        Idx cur = sh_rows[ idx_in_blk * VALUES_PER_THREAD + i ];
        if (row != cur)
        {
            atomicAdd(&( y[ row ] ), value);
            value = 0;
        }
        value += sh_vals[ idx_in_blk * VALUES_PER_THREAD + i ];
        row = cur;
    }

    // Add the last one.
    atomicAdd( &( y[ row ] ), value );
}




template<typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void shared_single(const Idx* __restrict__ row_indices,
                   const Idx* __restrict__ col_indices,
                   const Val1* __restrict__ values,
                   const int num_entries,
                   const Val2* __restrict__ x,
                   Val3* __restrict__ y,
                   const int y_size);


template<typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void coo_kernel(const Idx* __restrict__ row_indices,
                const Idx* __restrict__ col_indices,
                const Val1* __restrict__ values,
                const int num_entries,
                const Val2* __restrict__ x,
                Val3* __restrict__ y,
                const int y_size)
{
// #if IMPL == 0
//     naive_coo_kernel(row_indices, col_indices, values, num_entries, x, y, y_size);
// #elif IMPL == 1
//     shared_single(row_indices, col_indices, values, num_entries, x, y, y_size);
// #elif IMPL == 2
//     naive_multi_shared(row_indices, col_indices, values, num_entries, x, y, y_size);
// #elif IMPL == 3
//     naive_multi(row_indices, col_indices, values, num_entries, x, y, y_size);
// #else
//     serial_coo_kernel(row_indices, col_indices, values, num_entries, x, y, y_size);
// #endif


#if VALUES_PER_THREAD == 1
    naive_coo_kernel(   row_indices, col_indices, values, num_entries, x, y, y_size);
#elif VALUES_PER_THREAD == 500
    shared_single(      row_indices, col_indices, values, num_entries, x, y, y_size);
#else
    #ifdef SHARED == 1
        naive_multi_shared( row_indices, col_indices, values, num_entries, x, y, y_size);
    #else
        naive_multi(row_indices, col_indices, values, num_entries, x, y, y_size);
    #endif
#endif
}


template<typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void shared_single(const Idx* __restrict__ row_indices,
                   const Idx* __restrict__ col_indices,
                   const Val1* __restrict__ values,
                   const int num_entries,
                   const Val2* __restrict__ x,
                   Val3* __restrict__ y,
                   const int y_size)
{
    const unsigned ti = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const unsigned idx_in_blk = threadIdx.x;

// if (ti == 0) printf("... shared_single ...\n");

    __shared__ Idx  sh_rows[ BLOCK_SIZE + 2 ];
    __shared__ Val1 sh_vals[ BLOCK_SIZE + 2 ];

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
        sh_vals[ idx_in_blk + 1 ] = values[ ti ] * x[ col_indices[ ti ] ];

        __syncthreads();
        // __threadfence_block();

        Idx prv_row = sh_rows[ idx_in_blk ];
        Idx cur_row = sh_rows[ idx_in_blk + 1 ];
        Idx nxt_row = sh_rows[ idx_in_blk + 2 ];

        if (prv_row != cur_row && cur_row != nxt_row)
        {
            auto value = sh_vals[ idx_in_blk + 1 ];

            if (idx_in_blk == 0 || idx_in_blk == BLOCK_SIZE-1)
                atomicAdd( &( y[ cur_row ] ), value );
            else
                y[ cur_row ] = value;
        }
        else if (prv_row != cur_row && cur_row == nxt_row)
        {
            Val1 sum = 0;
            int i = idx_in_blk + 1;
            for (; sh_rows[ i ] == cur_row && i < BLOCK_SIZE + 2; ++i)
            {
                sum += sh_vals[ i ];
            }

            if (idx_in_blk == 0 || i == BLOCK_SIZE + 1)
                atomicAdd( &y[ cur_row ], sum );
            else
                y[ cur_row ] = sum;
        }
    }
}


template<typename Idx, typename Val1, typename Val2, typename Val3>
__global__
void coo_spmv(const Idx* __restrict__ row_indices,
               const Idx* __restrict__ col_indices,
               const Val1* __restrict__ values,
               const int num_entries,
               const Val2* __restrict__ x,
               Val3* __restrict__ y,
               const int y_size)
{
    coo_kernel<Idx, Val1, Val2, Val3>
              (row_indices, col_indices, values, num_entries, x, y, y_size);
}
