#pragma once

template <typename IndexType, typename ValueType1, typename ValueType2, typename ValueType3, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE,1)
__global__ void
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
    typedef ValueType1 ValueType;

    __shared__ IndexType offsets[BLOCK_SIZE];

    const IndexType thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const IndexType grid_size = BLOCK_SIZE * gridDim.x;

    for(IndexType base = 0; base < num_diagonals; base += BLOCK_SIZE)
    {
        // read a chunk of the diagonal offsets into shared memory
        const IndexType chunk_size = IndexType(BLOCK_SIZE) < num_diagonals - base ? IndexType(BLOCK_SIZE) : num_diagonals - base;

        if(threadIdx.x < chunk_size)
            offsets[threadIdx.x] = diagonal_offsets[base + threadIdx.x];

        __syncthreads();

        // process chunk
        for(IndexType row = thread_id; row < num_rows; row += grid_size)
        {
            ValueType sum = (base == 0) ? ValueType(0) : ValueType(0);

            // index into values array
            IndexType idx = row + pitch * base;

            for(IndexType n = 0; n < chunk_size; n++)
            {
                const IndexType col = row + offsets[n];

                if(col >= 0 && col < num_cols)
                {
                    const ValueType A_ij = values[idx];
                    sum = sum + (A_ij * x[col]);
                }

                idx += pitch;
            }

            y[row] = sum;
        }

        // wait until all threads are done reading offsets
        __syncthreads();
    }
}