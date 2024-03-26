
template<typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void csr_kernel_naive(const unsigned int num_rows,
                const Idx*   Ar,
                const Idx*   Ac,
                const Val1*  Ax,
                const Val2*  x,
                Val3*        y)
{
    const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    Val1 value = 0;
    if (idx < num_rows)
    {
        Idx row_start = Ar[idx];
        Idx row_end = Ar[idx + 1];
        for (int i = row_start; i < row_end; ++i)
        {
            // Val3 value = Ax[i] * x[Ac[i]];
            // y[idx] += value;
            Val3 val = Ax[i] * x[Ac[i]];
            value += val;
        }
        y[idx] = value;
    }
}



// template<typename Idx, typename Val1, typename Val2, typename Val3>
// __device__
// void csr_kernel(const unsigned int num_rows,
//                 const Idx*   Ar,
//                 const Idx*   Ac,
//                 const Val1*  Ax,
//                 const Val2*  x,
//                 Val3*        y)
// {
//     const int WARP_SIZE = 32;

//     const int lane = threadIdx.x % WARP_SIZE;
//     const int idx_in_blk = threadIdx.x;
//     const int blk_idx = blockIdx.x;
//     // const int ti = BLOCK_SIZE * blockIdx.x + idx_in_blk;

//     __shared__ Val1 sh_vals[ BLOCK_SIZE ];
//     __shared__ Val1 sh_sum;

//     if (idx_in_blk == 0)
//         sh_sum = 0;

//     if (blk_idx < num_rows)
//     {
//         // TODO: fetch using two threads like cusp does
//         Idx row_start = Ar[ blk_idx ];
//         Idx row_end = Ar[ blk_idx + 1 ];

//         for (int i = row_start; i < row_end; i += BLOCK_SIZE)
//         {
//             int global_idx = i + idx_in_blk;
//             if (global_idx < row_end)
//             {
//                 Val3 val = Ax[ global_idx ] * x[ Ac[ global_idx ] ];
//                 sh_vals[ idx_in_blk ] = val;
//             }
//             else
//             {
//                 sh_vals[ idx_in_blk ] = 0;
//             }

//             __syncthreads();

//             Val1 value = sh_vals[ idx_in_blk ];
//             const unsigned mask = 0xffffffff;
//             value += __shfl_down_sync(mask, value, 16);
//             value += __shfl_down_sync(mask, value, 8);
//             value += __shfl_down_sync(mask, value, 4);
//             value += __shfl_down_sync(mask, value, 2);
//             value += __shfl_down_sync(mask, value, 1);

//             if (lane == 0)
//                 sh_vals[ idx_in_blk ] = value;

//             __syncthreads();

//             if (idx_in_blk == 0)
//             {
//                 Val1 sum_warps = 0;
//                 for (int j = 0; j < BLOCK_SIZE; j += WARP_SIZE)
//                     sum_warps += sh_vals[ j ];
//                 sh_sum += sum_warps;
//             }
//         }
//     }

//     __syncthreads();

//     if (idx_in_blk == 0)
//         y[ blk_idx ] = sh_sum;
// }




// Modulo operation assuming mod is a value of this form: mod = 2^exp.
template<typename T>
__device__
inline T exp2mod(T v, T mod)
{
    return v & (mod - 1);
}


template<typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void csr_kernel_warp(const unsigned int num_rows,
                const Idx*   Ar,
                const Idx*   Ac,
                const Val1*  Ax,
                const Val2*  x,
                Val3*        y)
{
    const int WARP_SIZE = 32;

    const int lane = threadIdx.x % WARP_SIZE;
    const int idx_in_blk = threadIdx.x;
    const int blk_idx = blockIdx.x;

    const int row = blk_idx;

    // const int ti = BLOCK_SIZE * blockIdx.x + idx_in_blk;

    // __shared__ Val1 sh_sum;

    // if (idx_in_blk == 0)
    //     sh_sum = 0;

    __shared__ Idx sh_row_info[2];

    if (blk_idx >= num_rows)
        return;

    // TODO: fetch using two threads like cusp does
    // Idx row_start = Ar[ blk_idx ];
    // Idx row_end   = Ar[ blk_idx + 1 ];
    if (idx_in_blk < 2)
        sh_row_info[ idx_in_blk ] = Ar[ row + idx_in_blk ];
    __syncthreads();
    Idx row_start = sh_row_info[ 0 ];
    Idx row_end   = sh_row_info[ 1 ];


    Val3 value = 0;

    if (row_end - row_start > 32)
    {
        Idx aligned_start = row_start - exp2mod(row_start, 32) + lane;

        if (int i = aligned_start; i >= row_start && i < row_end)
            value += Ax[ i ] * x[ Ac[ i ] ];

        for (int i = aligned_start + BLOCK_SIZE; i < row_end; i += BLOCK_SIZE)
            value += Ax[ i ] * x[ Ac[ i ] ];
    }
    else
    {
        for (int i = row_start + lane; i < row_end; i += BLOCK_SIZE)
            value += Ax[ i ] * x[ Ac[ i ] ];
    }


    const unsigned mask = 0xffffffff;
    value += __shfl_down_sync(mask, value, 16);
    value += __shfl_down_sync(mask, value,  8);
    value += __shfl_down_sync(mask, value,  4);
    value += __shfl_down_sync(mask, value,  2);
    value += __shfl_down_sync(mask, value,  1);

    // TODO: later
    // if (lane == 0)
    //     sh_vals[ idx_in_blk ] = value;

    // TODO: solve reduction for block size > 32
    if (idx_in_blk == 0)
    {
        // Val1 sum_warps = 0;
        // for (int j = 0; j < BLOCK_SIZE; j += WARP_SIZE)
        //     sum_warps += sh_vals[ j ];
        // sh_sum += sum_warps;
        y[ blk_idx ] = value;
    }
}








template<typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void csr_kernel_block(const unsigned int num_rows,
                const Idx*   Ar,
                const Idx*   Ac,
                const Val1*  Ax,
                const Val2*  x,
                Val3*        y)
{
    const int WARP_SIZE = 32;

    const int lane = threadIdx.x % WARP_SIZE;
    const int blk_idx = blockIdx.x;
    const int idx_in_blk = threadIdx.x;
    const int warp_in_block = idx_in_blk / WARP_SIZE;

    const int row = blk_idx;

    __shared__ Val3 sh_sums[ BLOCK_SIZE / WARP_SIZE ];
    __shared__ Idx sh_row_info[2];

    // if (lane == 0)
    //     sh_sums[ warp_in_block ] = 0;

    if (blk_idx >= num_rows)
        return;

    // TODO: fetch using two threads like cusp does
    // Idx row_start = Ar[ blk_idx ];
    // Idx row_end = Ar[ blk_idx + 1 ];
    if (idx_in_blk < 2)
        sh_row_info[ idx_in_blk ] = Ar[ row + idx_in_blk ];
    __syncthreads();
    const Idx row_start = sh_row_info[ 0 ];
    const Idx row_end   = sh_row_info[ 1 ];

    Val3 value = 0;

    for (int i = row_start + idx_in_blk; i < row_end; i += BLOCK_SIZE)
        value += Ax[ i ] * x[ Ac[ i ] ];

    const unsigned mask = 0xffffffff;
    value += __shfl_down_sync(mask, value, 16);
    value += __shfl_down_sync(mask, value, 8);
    value += __shfl_down_sync(mask, value, 4);
    value += __shfl_down_sync(mask, value, 2);
    value += __shfl_down_sync(mask, value, 1);

    if (lane == 0)
        sh_sums[ warp_in_block ] = value;

    __syncthreads();

    // if (idx_in_blk == 0)
    // {
    //     Val1 total_sum = 0;
    //     #pragma unroll
    //     for (int j = 0; j < BLOCK_SIZE / WARP_SIZE; ++j)
    //         total_sum += sh_sums[ j ];
    //     y[ row ] = total_sum;
    // }

    const int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

    if (idx_in_blk < WARPS_PER_BLOCK / 2)
    {
        for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset /= 2)
        {
            auto my_val = sh_sums[ warp_in_block ];
            auto friend_val = sh_sums[ warp_in_block + offset ];
            sh_sums[ warp_in_block ] = my_val + friend_val;
            __syncthreads();
        }
    }

    if (idx_in_blk == 0)
        y[ row ] = sh_sums[ 0 ];
}



template<typename Idx, typename Val1, typename Val2, typename Val3>
__global__
void csr_spmv(const unsigned int num_rows,
              const Idx*   Ar,
              const Idx*   Ac,
              const Val1*  Ax,
              const Val2*  x,
              Val3*        y)
{
    // csr_kernel_block<Idx, Val1, Val2, Val3>(num_rows, Ar, Ac, Ax, x, y);
    csr_kernel_warp<Idx, Val1, Val2, Val3>(num_rows, Ar, Ac, Ax, x, y);
    // csr_kernel_naive<Idx, Val1, Val2, Val3>(num_rows, Ar, Ac, Ax, x, y);
}
