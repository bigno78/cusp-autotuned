#pragma once


template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3,
          unsigned int BLOCK_SIZE>
__device__ void
ktt_dia_vector_kernel_simple(
                const int num_rows,
                const int num_cols,
                const int num_diagonals,
                const int pitch,
                const IndexType* diagonal_offsets,
                const ValueType1* values,
                const ValueType2* x,
                ValueType3* y)
{
    using ValueType = ValueType3;

    const IndexType thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (thread_id < num_rows) {
        ValueType sum = ValueType(0);
        for (IndexType i = 0; i < num_diagonals; ++i) {
            IndexType col = diagonal_offsets[i] + thread_id;
            if (col >= 0 && col < num_cols) {
                auto val = values[i*pitch + thread_id];
                auto a = x[col];
                sum +=val*a;
            }
        }
        y[thread_id] = sum;
    }
}

template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3,
          unsigned int BLOCK_SIZE>
__device__ void
ktt_dia_vector_kernel_blocked(
                const int num_rows,
                const int num_cols,
                const int num_diagonals,
                const int pitch,
                const IndexType* diagonal_offsets,
                const ValueType1* values,
                const ValueType2* x,
                ValueType3* y)
{
    using ValueType = ValueType3;

    const IndexType thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    __shared__ IndexType offsets[BLOCK_SIZE];
    ValueType sum = ValueType(0);

    for (int offset_base = 0; offset_base < num_diagonals; offset_base += BLOCK_SIZE) {
        if (offset_base + threadIdx.x < num_diagonals) {
            offsets[threadIdx.x] = diagonal_offsets[offset_base + threadIdx.x];
        }

        __syncthreads();

        int batch_size = BLOCK_SIZE > num_diagonals - offset_base ? num_diagonals - offset_base : BLOCK_SIZE;

        if (thread_id < num_rows) {
            for (IndexType i = 0; i < batch_size; ++i) {
                IndexType col = offsets[i] + thread_id;
                if (col >= 0 && col < num_cols) {
                    sum += values[(offset_base + i)*pitch + thread_id]*x[col];
                }
            }
            y[thread_id] = sum;
        }

        __syncthreads();
    }
}

#define ITERATION(i, offset) \
    col = offset + thread_id;                        \
    if (col >= 0 && col < num_cols) {                              \
        sum += values[i*pitch + thread_id]*x[col]; \
    }

template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3,
          unsigned int BLOCK_SIZE>
__device__ void
ktt_dia_vector_kernel_crazy(
                const int num_rows,
                const int num_cols,
                const int num_diagonals,
                const int pitch,
                const IndexType* diagonal_offsets,
                const ValueType1* values,
                const ValueType2* x,
                ValueType3* y)
{
    using ValueType = ValueType3;

    const IndexType thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    ValueType sum = ValueType(0);

    if (thread_id < num_rows) {
        IndexType col;
        ITERATION(0, -4);
        ITERATION(1, -3);
        ITERATION(2, -2);
        ITERATION(3, -1);
        ITERATION(4, -0);
        ITERATION(5, -1);
        ITERATION(6, -2);
        ITERATION(7, -3);
        ITERATION(8, -4);
        ITERATION(9, -5);
        y[thread_id] = sum;
    }

}

template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3,
          unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE,1) __global__ void
ktt_dia_vector_kernel(
                const int num_rows,
                const int num_cols,
                const int num_diagonals,
                const int pitch,
                const IndexType* diagonal_offsets,
                const ValueType1* values,
                const ValueType2* x,
                ValueType3* y)
{
#if KERNEL_TYPE == 0
    ktt_dia_vector_kernel_simple<IndexType, ValueType1, ValueType2, ValueType3, BLOCK_SIZE>
        (num_rows, num_cols, num_diagonals, pitch, diagonal_offsets, values, x, y);
#elif KERNEL_TYPE == 1
    ktt_dia_vector_kernel_blocked<IndexType, ValueType1, ValueType2, ValueType3, BLOCK_SIZE>
        (num_rows, num_cols, num_diagonals, pitch, diagonal_offsets, values, x, y);
#elif KERNEL_TYPE == 2
    ktt_dia_vector_kernel_crazy<IndexType, ValueType1, ValueType2, ValueType3, BLOCK_SIZE>
        (num_rows, num_cols, num_diagonals, pitch, diagonal_offsets, values, x, y);
#endif
}

// template <typename IndexType,
//           typename ValueType1,
//           typename ValueType2,
//           typename ValueType3,
//           unsigned int BLOCK_SIZE>
// __launch_bounds__(BLOCK_SIZE,1) __global__ void
// ktt_dia_vector_kernel(
//                 const int num_rows,
//                 const int num_cols,
//                 const int num_diagonals,
//                 const int pitch,
//                 const IndexType* diagonal_offsets,
//                 const ValueType1* values,
//                 const ValueType2* x,
//                 ValueType3* y)
// {
//     typedef ValueType1 ValueType;

//     __shared__ IndexType offsets[BLOCK_SIZE];

//     const IndexType thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x;
//     const IndexType grid_size = BLOCK_SIZE * gridDim.x;

//     for(IndexType base = 0; base < num_diagonals; base += BLOCK_SIZE)
//     {
//         // read a chunk of the diagonal offsets into shared memory
//         const IndexType chunk_size = IndexType(BLOCK_SIZE) < num_diagonals - base ? IndexType(BLOCK_SIZE) : num_diagonals - base;

//         if(threadIdx.x < chunk_size)
//             offsets[threadIdx.x] = diagonal_offsets[base + threadIdx.x];

//         __syncthreads();

//         // process chunk
//         for(IndexType row = thread_id; row < num_rows; row += grid_size)
//         {
//             ValueType sum = (base == 0) ? ValueType(0) : ValueType(0);

//             // index into values array
//             IndexType idx = row + pitch * base;

//             for(IndexType n = 0; n < chunk_size; n++)
//             {
//                 const IndexType col = row + offsets[n];

//                 if(col >= 0 && col < num_cols)
//                 {
//                     const ValueType A_ij = values[idx];
//                     sum = sum + (A_ij * x[col]);
//                 }

//                 idx += pitch;
//             }

//             y[row] = sum;
//         }

//         // wait until all threads are done reading offsets
//         __syncthreads();
//     }
// }