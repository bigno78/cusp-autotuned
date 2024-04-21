

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
#if AVOID_ATOMIC == 1
            if (first)
#endif
                atomicAdd(&y[ row ], value);
#if AVOID_ATOMIC == 1
            else
                y[ row ] = value;
#endif
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

#if AVOID_ATOMIC == 1
            if (idx_in_blk == 0 || idx_in_blk == BLOCK_SIZE-1)
#endif
                atomicAdd( &( y[ cur_row ] ), value );
#if AVOID_ATOMIC == 1
            else
                y[ cur_row ] = value;
#endif
        }
        else if (prv_row != cur_row && cur_row == nxt_row)
        {
            Val1 sum = 0;
            int i = idx_in_blk + 1;
            for (; sh_rows[ i ] == cur_row && i < BLOCK_SIZE + 2; ++i)
            {
                sum += sh_vals[ i ];
            }
#if AVOID_ATOMIC == 1
            if (idx_in_blk == 0 || i == BLOCK_SIZE + 1 || sh_rows[ i ] == -1)
#endif
                atomicAdd( &y[ cur_row ], sum );
#if AVOID_ATOMIC == 1
            else
                y[ cur_row ] = sum;
#endif
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

    __shared__ Idx  sh_rows[ BLOCK_SIZE * VALUES_PER_THREAD ];
    __shared__ Val1 sh_vals[ BLOCK_SIZE * VALUES_PER_THREAD ];


    // const int last = VALUES_PER_THREAD * BLOCK_SIZE * blockIdx.x
    //                 + BLOCK_SIZE * VALUES_PER_THREAD;
    // if (last <= num_entries)
    // {
    //     #pragma unroll
    //     for (int i = 0; i < VALUES_PER_THREAD; ++i)
    //     {
    //         const int idx = VALUES_PER_THREAD * BLOCK_SIZE * blockIdx.x
    //                         + idx_in_blk + BLOCK_SIZE * i;

    //         auto row = row_indices[ idx ];
    //         auto value = values[ idx ] * x[ col_indices[ idx ] ];
    //         sh_rows[ idx_in_blk + i * BLOCK_SIZE ] = row;
    //         sh_vals[ idx_in_blk + i * BLOCK_SIZE ] = value;
    //     }
    // }
    // else
    // {
        // #pragma unroll
        for (int i = 0; i < VALUES_PER_THREAD; ++i)
        {
            // const int idx = ti * VALUES_PER_THREAD + BLOCK_SIZE * i;
            const int idx = VALUES_PER_THREAD * BLOCK_SIZE * blockIdx.x
                        + idx_in_blk + BLOCK_SIZE * i;

            if (idx < num_entries)
            {
                auto row = row_indices[ idx ];
                auto value = values[ idx ] * x[ col_indices[ idx ] ];
                sh_rows[ idx_in_blk + i * BLOCK_SIZE ] = row;
                sh_vals[ idx_in_blk + i * BLOCK_SIZE ] = value;
            }
            else
            {
                sh_rows[ idx_in_blk + i * BLOCK_SIZE ] = -1;
                sh_vals[ idx_in_blk + i * BLOCK_SIZE ] = 0;
            }
        }
    // }

    __syncthreads();

    unsigned begin = idx_in_blk * VALUES_PER_THREAD;

    auto row = sh_rows[ begin ];
    Val1 value = 0;
    bool first = true;
    for (int i = 0; i < VALUES_PER_THREAD; ++i)
    {
        Idx cur = sh_rows[ begin + i ];
        if (row != cur)
        {
            if (row != -1)
            {
#if AVOID_ATOMIC == 1
                if (first)
#endif
                    atomicAdd(&y[ row ], value);
#if AVOID_ATOMIC == 1
                else
                    y[ row ] = value;
#endif
            }
            value = 0;
            first = false;
        }
        value += sh_vals[ begin + i ];
        row = cur;
    }
    if (row != -1) atomicAdd(&y[ row ], value);
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



// Modulo operation assuming mod is a value of this form: mod = 2^exp.
template<typename T, typename S>
__device__
inline auto coo_mod2exp(T v, S mod)
{
    return v & (mod - 1);
}


template<typename T, typename U>
__device__
auto coo_divide_into(T value, U chunk)
{
    auto div = value / chunk;
    if (value % chunk == 0)
        return div;

    return div + 1;
}


template<typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void coo_warp_reduce(const Idx* __restrict__ row_indices,
                     const Idx* __restrict__ col_indices,
                     const Val1* __restrict__ values,
                     const int num_entries,
                     const Val2* __restrict__ x,
                     Val3* __restrict__ y,
                     const int y_size)
{
    constexpr unsigned WARP_SIZE = 32;

    const unsigned ti = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const unsigned lane = coo_mod2exp(threadIdx.x, WARP_SIZE);
    const unsigned global_warp_id = ti / WARP_SIZE;

    const unsigned chunk = VALUES_PER_THREAD * WARP_SIZE;

    const unsigned begin = global_warp_id * chunk;
    const unsigned end = min( begin + chunk, num_entries );

    if (begin >= end)
        return;

#if USE_CARRY != 0
    Idx  carry_row = -1;
    Val3 carry_val =  0;
#endif

    for (unsigned idx = begin + lane; idx < end; idx += WARP_SIZE)
    {
        Idx row = row_indices[ idx ];
        Idx col = col_indices[ idx ];

        Val3 val = values[ idx ] * x[ col ];

#if USE_CARRY != 0
        if (lane == 0)
        {
            if (row == carry_row)
                val += carry_val;
            else if (carry_row != -1)
                atomicAdd(y + carry_row, carry_val);
        }
#endif

        constexpr unsigned mask = 0xffffffff;

        for (unsigned off = 1; off <= WARP_SIZE / 2; off *= 2)
        {
            auto next_row = __shfl_up_sync(mask, row, off);
            auto next_val = __shfl_up_sync(mask, val, off);

            if (lane >= off && next_row == row)
                val += next_val;
        }

        auto neigh_row = __shfl_down_sync(mask, row, 1);

        if (lane < WARP_SIZE - 1 && row != neigh_row)
        {
            atomicAdd(y + row, val);
        }

#if USE_CARRY != 0
        carry_row = __shfl_down_sync(mask, row, WARP_SIZE - 1);
        carry_val = __shfl_down_sync(mask, val, WARP_SIZE - 1);
#else
        if (lane == WARP_SIZE - 1)
        {
            atomicAdd(y + row, val);
        }
#endif
    }

#if USE_CARRY != 0
    if (lane == 0)
        if (carry_row != -1)
            atomicAdd(y + carry_row, carry_val);
#endif
}





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
