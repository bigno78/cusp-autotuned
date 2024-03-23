

template<typename Idx, typename Val1, typename Val2, typename Val3>
__global__
__launch_bounds__(BLOCK_SIZE, 1)
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

    // no work left for this thread
    if (begin >= end)
        return;

    float value = 0;
    Idx row = row_indices[ begin ];
    for (int i = begin; i < end; ++i)
    {
        Idx cur = row_indices[ i ];
        if (row != cur)
        {
            atomicAdd(&y[ row ], value);
            value = 0;
        }
        value += values[ i ] * x[ col_indices[ i ] ];
        row = cur;
    }
    auto* ptr = &y[ row ];
    atomicAdd(ptr, value);
}



template<typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void naive_multi_direct(const Idx* __restrict__ row_indices,
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

    // no work left for this thread
    if (begin >= end)
        return;

    float value = 0;
    Idx row = row_indices[ begin ];
    bool first = true;
    for (int i = begin; i < end; ++i)
    {
        Idx cur = row_indices[ i ];
        if (row != cur)
        {
            if (first)
                atomicAdd(&y[ row ], value);
            else
                y[ row ] = value;
            value = 0;
            first = false;
        }
        value += values[ i ] * x[ col_indices[ i ] ];
        row = cur;
    }
    auto* ptr = &y[ row ];
    atomicAdd(ptr, value);
}




template<typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void old_naive_multi_shared(const Idx* __restrict__ row_indices,
                        const Idx* __restrict__ col_indices,
                        const Val1* __restrict__ values,
                        const int num_entries,
                        const Val2* __restrict__ x,
                        Val3* __restrict__ y,
                        const int y_size)
{
    const unsigned idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const unsigned idx_in_blk = threadIdx.x;
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
void shared_multi(const Idx* __restrict__ row_indices,
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
#if VALUES_PER_THREAD == 1
    #if SHARED == 1
        shared_single(      row_indices, col_indices, values, num_entries, x, y, y_size     );
    #else
        naive_coo_kernel(   row_indices, col_indices, values, num_entries, x, y, y_size     );
    #endif
#else
    #if SHARED == 1
        shared_multi(       row_indices, col_indices, values, num_entries, x, y, y_size    );
    #elif SHARED == 0
        // naive_multi(        row_indices, col_indices, values, num_entries, x, y, y_size    );
        naive_multi_direct( row_indices, col_indices, values, num_entries, x, y, y_size);
    #else
        naive_multi(        row_indices, col_indices, values, num_entries, x, y, y_size    );
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

    __shared__ Idx  sh_rows[ BLOCK_SIZE + 2 ];
    __shared__ Val1 sh_vals[ BLOCK_SIZE + 2 ];

    // if (ti < num_entries)
    // {
        if (idx_in_blk == 0 || idx_in_blk == BLOCK_SIZE-1)
        {
            sh_rows[ idx_in_blk ] = -1;
            sh_vals[ idx_in_blk ] = 0;
            sh_rows[ idx_in_blk + 2 ] = -1;
            sh_vals[ idx_in_blk + 2 ] = 0;
        }

        if (ti < num_entries)
        {
            sh_rows[ idx_in_blk + 1 ] = row_indices[ ti ];
            sh_vals[ idx_in_blk + 1 ] = values[ ti ] * x[ col_indices[ ti ] ];
        }
        else
        {
            sh_rows[ idx_in_blk + 1 ] = -1;
            sh_vals[ idx_in_blk + 1 ] = 0;
        }

        __syncthreads();
        // __threadfence_block();

        if (ti >= num_entries)
            return;

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

            if (idx_in_blk == 0 || i == BLOCK_SIZE + 1 || sh_rows[ i ] == -1)
                atomicAdd( &y[ cur_row ], sum );
            else
                y[ cur_row ] = sum;
        }
    // }
}







template<typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void shared_multi(const Idx* __restrict__ row_indices,
                  const Idx* __restrict__ col_indices,
                  const Val1* __restrict__ values,
                  const int num_entries,
                  const Val2* __restrict__ x,
                  Val3* __restrict__ y,
                  const int y_size)
{
    const unsigned idx_in_blk = threadIdx.x;
    // s/threadIdx.x/idx_in_blk/
    const unsigned ti = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    __shared__ Idx  sh_rows[ BLOCK_SIZE * VALUES_PER_THREAD ];
    __shared__ Val1 sh_vals[ BLOCK_SIZE * VALUES_PER_THREAD ];

    // TODO: solve the remaining elements
    // TODO: probably should be <=
    if ( ( blockIdx.x + 1 ) * BLOCK_SIZE * VALUES_PER_THREAD <= num_entries )
    {
        for (int i = 0; i < VALUES_PER_THREAD; ++i)
        {
            // const int idx = ti * VALUES_PER_THREAD + BLOCK_SIZE * i;
            const int idx = VALUES_PER_THREAD * BLOCK_SIZE * blockIdx.x
                          + idx_in_blk + BLOCK_SIZE * i;

            auto row = row_indices[ idx ];
            auto value = values[ idx ] * x[ col_indices[ idx ] ];
            sh_rows[ idx_in_blk + i * BLOCK_SIZE ] = row;
            sh_vals[ idx_in_blk + i * BLOCK_SIZE ] = value;
        }

        __syncthreads();

        unsigned begin = idx_in_blk * VALUES_PER_THREAD;

        auto row = sh_rows[ begin ];
        Val1 value = 0;
        for (int i = 0; i < VALUES_PER_THREAD; ++i)
        {
            Idx cur = sh_rows[ begin + i ];
            if (row != cur)
            {
                if (i == 0) atomicAdd(&y[ row ], value);
                else        y[ row ] = value;
                value = 0;
            }
            value += sh_vals[ begin + i ];
            row = cur;
        }
        atomicAdd(&y[ row ], value);

        // const unsigned end = BLOCK_SIZE * VALUES_PER_THREAD - 1;

        // for (int j = 0; j < VALUES_PER_THREAD; ++j)
        // {
        //     const unsigned idx = idx_in_blk * VALUES_PER_THREAD + j;
        //     Idx cur_row = sh_rows[ idx ];
        //     Idx prv_row = idx == 0   ? -1 : sh_rows[ idx-1 ];
        //     Idx nxt_row = idx == end ? -1 : sh_rows[ idx+1 ];

        //     if (prv_row != cur_row && cur_row != nxt_row)
        //     {
        //         auto value = sh_vals[ idx ];
        //         if (idx == 0 || idx >= end)
        //             atomicAdd( &( y[ cur_row ] ), value );
        //         else
        //             y[ cur_row ] = value;
        //     }
        //     else if (prv_row != cur_row && cur_row == nxt_row)
        //     {
        //         Val1 sum = 0;
        //         int i = idx;
        //         for (; sh_rows[ i ] == cur_row && i < end + 1; ++i)
        //             sum += sh_vals[ i ];
        //         if (idx_in_blk == 0 || i >= end)
        //             atomicAdd( &y[ cur_row ], sum );
        //         else
        //             y[ cur_row ] = sum;
        //     }
        // }
    }

    // // const unsigned end = gridDim.x * BLOCK_SIZE * VALUES_PER_THREAD;
    // unsigned end = 0;
    // for (int i = 1; i <= gridDim.x; ++i)
    // {
    //     if ( i * BLOCK_SIZE * VALUES_PER_THREAD <= num_entries )
    //         end = i * BLOCK_SIZE * VALUES_PER_THREAD;
    //     else
    //         break;
    // }
    if (num_entries % (BLOCK_SIZE * VALUES_PER_THREAD) == 0)
        return;

    auto times = num_entries / (BLOCK_SIZE * VALUES_PER_THREAD);
    unsigned end = times * BLOCK_SIZE * VALUES_PER_THREAD;

    if (ti < num_entries - end)
    {
        // naive_coo_kernel( row_indices + end,
        //                   col_indices + end,
        //                   values + end,
        //                   num_entries - end,
        //                   x,
        //                   y,
        //                   y_size );

        naive_multi( row_indices + end,
                          col_indices + end,
                          values + end,
                          num_entries - end,
                          x,
                          y,
                          y_size );

        // const int n = ti + end;
        // // if (n >= num_entries)
        // //     return;
        // Val1 value = values[ n ] * x[ col_indices[ n ] ];
        // atomicAdd( &( y[ row_indices[ n ] ] ), value );
    }
}








// template<typename Idx, typename Val1, typename Val2, typename Val3>
// __device__
// void shared_multi(const Idx* __restrict__ row_indices,
//                   const Idx* __restrict__ col_indices,
//                   const Val1* __restrict__ values,
//                   const int num_entries,
//                   const Val2* __restrict__ x,
//                   Val3* __restrict__ y,
//                   const int y_size)
// {
//     const unsigned ti = BLOCK_SIZE * blockIdx.x + threadIdx.x;
//     const unsigned idx_in_blk = threadIdx.x;

//     __shared__ Idx  sh_rows[ BLOCK_SIZE * VALUES_PER_THREAD + 2 ];
//     __shared__ Val1 sh_vals[ BLOCK_SIZE * VALUES_PER_THREAD + 2 ];

//     // last index in shared arrays
//     const unsigned end = BLOCK_SIZE * VALUES_PER_THREAD + 1;

//     if ( ( blockIdx.x + 1 ) * BLOCK_SIZE * VALUES_PER_THREAD < num_entries )
//     {
//         if (idx_in_blk == 0 || idx_in_blk == BLOCK_SIZE-1)
//         {
//             sh_rows[ idx_in_blk * VALUES_PER_THREAD ] = -1;
//             sh_vals[ idx_in_blk * VALUES_PER_THREAD ] = 0;
//             sh_rows[ idx_in_blk * VALUES_PER_THREAD + 2 ] = -1;
//             sh_vals[ idx_in_blk * VALUES_PER_THREAD + 2 ] = 0;
//         }

//         for (int i = 0; i < VALUES_PER_THREAD; ++i)
//         {
//             const int idx = ti + VALUES_PER_THREAD * i;

//             auto row = row_indices[ idx ];
//             auto value = values[ idx ] * x[ col_indices[ idx ] ];
//             sh_rows[ idx_in_blk + 1 + i * VALUES_PER_THREAD ] = row;
//             sh_vals[ idx_in_blk + 1 + i * VALUES_PER_THREAD ] = value;
//         }

//         __syncthreads();

//         for (int j = 0; j < VALUES_PER_THREAD; ++j)
//         {
//             const unsigned idx = idx_in_blk * VALUES_PER_THREAD + j;
//             Idx prv_row = sh_rows[ idx ];
//             Idx cur_row = sh_rows[ idx + 1 ];
//             Idx nxt_row = sh_rows[ idx + 2 ];

//             if (prv_row != cur_row && cur_row != nxt_row)
//             {
//                 auto value = sh_vals[ idx + 1 ];

//                 if (idx == 0 || idx >= end)
//                     atomicAdd( &( y[ cur_row ] ), value );
//                 else
//                     y[ cur_row ] = value;
//             }
//             else if (prv_row != cur_row && cur_row == nxt_row)
//             {
//                 Val1 sum = 0;
//                 int i = idx + 1;
//                 for (; sh_rows[ i ] == cur_row && i < end + 1; ++i)
//                     sum += sh_vals[ i ];

//                 if (idx_in_blk == 0 || i >= end)
//                     atomicAdd( &y[ cur_row ], sum );
//                 else
//                     y[ cur_row ] = sum;
//             }
//         }
//     }
// }






template<typename Idx, typename Val1, typename Val2, typename Val3>
__global__
__launch_bounds__(BLOCK_SIZE, 1)
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
