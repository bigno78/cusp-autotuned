
// template <typename Idx, typename Val1, typename Val2, typename Val3>
// __device__
// void csr_kernel(const unsigned int num_rows,
//                 const Idx*   Ar,
//                 const Idx*   Ac,
//                 const Val1*  Ax,
//                 const Val2*  x,
//                 Val3*        y)
// {
//     const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

//     Val1 value = 0;
//     if (idx < num_rows)
//     {
//         Idx row_start = Ar[idx];
//         Idx row_end = Ar[idx + 1];
//         for (int i = row_start; i < row_end; ++i)
//         {
//             // Val3 value = Ax[i] * x[Ac[i]];
//             // y[idx] += value;
//             Val3 val = Ax[i] * x[Ac[i]];
//             value += val;
//         }
//         y[idx] = value;
//     }
// }



// template <typename Idx, typename Val1, typename Val2, typename Val3>
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




template <typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void csr_kernel(const unsigned int num_rows,
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
    // const int ti = BLOCK_SIZE * blockIdx.x + idx_in_blk;

    __shared__ Val1 sh_sum;

    if (idx_in_blk == 0)
        sh_sum = 0;

    if (blk_idx < num_rows)
    {
        // TODO: fetch using two threads like cusp does
        Idx row_start = Ar[ blk_idx ];
        Idx row_end   = Ar[ blk_idx + 1 ];

        Val3 value = 0;

        for (int i = row_start; i < row_end; i += BLOCK_SIZE)
        {
            int global_idx = i + idx_in_blk;
            if (global_idx < row_end)
                value += Ax[ global_idx ] * x[ Ac[ global_idx ] ];
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
}






template <typename Idx, typename Val1, typename Val2, typename Val3>
__global__
void csr_spmv(const unsigned int num_rows,
              const Idx*   Ar,
              const Idx*   Ac,
              const Val1*  Ax,
              const Val2*  x,
              Val3*        y)
{
    csr_kernel<Idx, Val1, Val2, Val3>(num_rows, Ar, Ac, Ax, x, y);
}
