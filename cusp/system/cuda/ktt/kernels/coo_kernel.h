

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
        y[ i ] = 0;
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
void naive_multi(const Idx* __restrict__ row_indices,
                 const Idx* __restrict__ col_indices,
                 const Val1* __restrict__ values,
                 const int num_entries,
                 const Val2* __restrict__ x,
                 Val3* __restrict__ y,
                 const int y_size)
{
    const unsigned idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const unsigned begin = idx * VALUES_PER_THREAD;
    const unsigned end = min(num_entries, begin + VALUES_PER_THREAD);

    // no work left for this thread
    if (begin >= end)
        return;

    Val3 value = 0;
    Idx row = row_indices[ begin ];
#if AVOID_ATOMIC == 1
    bool first = true;
#endif
    for (int i = begin; i < end; ++i)
    {
        Idx cur = row_indices[ i ];
        if (row != cur)
        {
#if AVOID_ATOMIC == 1
            if (first) atomicAdd(&y[ row ], value);
            else       y[ row ] = value;
            first = false;
#else
            atomicAdd(&y[ row ], value);
#endif
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
void shared_single(const Idx* __restrict__ row_indices,
                   const Idx* __restrict__ col_indices,
                   const Val1* __restrict__ values,
                   const int num_entries,
                   const Val2* __restrict__ x,
                   Val3* __restrict__ y,
                   const int y_size)
{
    // const unsigned ti         = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const unsigned idx_in_blk = threadIdx.x;
    const unsigned begin      = blockIdx.x * BLOCK_SIZE * VALUES_PER_THREAD;
    // const unsigned end        = begin      + BLOCK_SIZE * VALUES_PER_THREAD;

    __shared__ Idx  sh_rows[ BLOCK_SIZE + 2 ];
    __shared__ Val1 sh_vals[ BLOCK_SIZE + 2 ];

    for (int j = begin + idx_in_blk, i = 0; i < VALUES_PER_THREAD; j += BLOCK_SIZE, ++i)
    {
#if VALUES_PER_THREAD > 1
        // Need for sync. Some threads might be still reading shmem
        // from the end of the previous iteration.
        __syncthreads();
#endif

        if (idx_in_blk == 0 || idx_in_blk == BLOCK_SIZE-1)
        {
            sh_rows[ idx_in_blk ] = -1;
            sh_vals[ idx_in_blk ] = 0;
            sh_rows[ idx_in_blk + 2 ] = -1;
            sh_vals[ idx_in_blk + 2 ] = 0;
        }

        if (j < num_entries)
        {
            sh_rows[ idx_in_blk + 1 ] = row_indices[ j ];
            sh_vals[ idx_in_blk + 1 ] = values[ j ] * x[ col_indices[ j ] ];
        }
        else
        {
            sh_rows[ idx_in_blk + 1 ] = -1;
            sh_vals[ idx_in_blk + 1 ] = 0;
        }

        __syncthreads();

        if (j >= num_entries)
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
    }
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

    // TODO: unroll tuning parameter
    // #pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; ++i)
    {
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

    __syncthreads();

    unsigned begin = idx_in_blk * VALUES_PER_THREAD;

    auto row = sh_rows[ begin ];
    Val1 value = 0;
    bool first = true;

    // TODO: unroll tuning parameter, too
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
#if IMPL == 0
    coo_warp_reduce(        row_indices, col_indices, values, num_entries, x, y, y_size    );
#elif IMPL == 1
    #if VALUES_PER_THREAD == 1
        naive_coo_kernel(   row_indices, col_indices, values, num_entries, x, y, y_size    );
    #else
        naive_multi(        row_indices, col_indices, values, num_entries, x, y, y_size    );
    #endif
#elif IMPL == 2
        shared_single(      row_indices, col_indices, values, num_entries, x, y, y_size    );
#else
        shared_multi(       row_indices, col_indices, values, num_entries, x, y, y_size    );
#endif
}
