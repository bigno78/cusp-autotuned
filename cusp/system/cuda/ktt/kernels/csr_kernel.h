
inline __device__ int fib(int n) {
    if (n < 2) {
        return n;
    }
    return fib(n-1) + fib(n-2);
}

template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3,
          unsigned int VECTORS_PER_BLOCK,
          unsigned int THREADS_PER_VECTOR>
__global__ void
ktt_csr_vector_kernel(const unsigned int num_rows,
                       const IndexType*   Ap,
                       const IndexType*   Aj,
                       const ValueType1*  Ax,
                       const ValueType2*  x,
                       ValueType3*        y)
{

#if TEST_PARAM == 0
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Test param is 0.\n");
    }
#else
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Fib of 20 is %d.\n", fib(20));
    }
#endif

    typedef ValueType1 ValueType;

    __shared__ volatile ValueType sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals
    __shared__ volatile IndexType ptrs[VECTORS_PER_BLOCK][2];

    const IndexType THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    const IndexType thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const IndexType thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const IndexType vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const IndexType vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const IndexType num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    for(IndexType row = vector_id; row < num_rows; row += num_vectors)
    {
        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the straightforward version
        if(thread_lane < 2)
            ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];

        const IndexType row_start = ptrs[vector_lane][0];                   //same as: row_start = Ap[row];
        const IndexType row_end   = ptrs[vector_lane][1];                   //same as: row_end   = Ap[row+1];

        // initialize local sum
        ValueType sum = (thread_lane == 0) ? ValueType(0) : ValueType(0);

        if (THREADS_PER_VECTOR == 32 && row_end - row_start > 32)
        {
            // ensure aligned memory access to Aj and Ax

            IndexType jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;

            // accumulate local sums
            if(jj >= row_start && jj < row_end)
                sum = sum + Ax[jj]*x[Aj[jj]];

            // accumulate local sums
            for(jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR)
                sum = sum + Ax[jj]*x[Aj[jj]];
        }
        else
        {
            // accumulate local sums
            for(IndexType jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR)
                sum = sum + Ax[jj]*x[Aj[jj]];
        }

        // store local sum in shared memory
        sdata[threadIdx.x] = sum;

        // TODO: remove temp var WAR for MSVC
        ValueType temp;

        // reduce local sums to row sum
        if (THREADS_PER_VECTOR > 16) {
            temp = sdata[threadIdx.x + 16];
            sdata[threadIdx.x] = sum = sum + temp;
        }
        if (THREADS_PER_VECTOR >  8) {
            temp = sdata[threadIdx.x +  8];
            sdata[threadIdx.x] = sum = sum+ temp;
        }
        if (THREADS_PER_VECTOR >  4) {
            temp = sdata[threadIdx.x +  4];
            sdata[threadIdx.x] = sum = sum+ temp;
        }
        if (THREADS_PER_VECTOR >  2) {
            temp = sdata[threadIdx.x +  2];
            sdata[threadIdx.x] = sum = sum+temp;
        }
        if (THREADS_PER_VECTOR >  1) {
            temp = sdata[threadIdx.x +  1];
            sdata[threadIdx.x] = sum = sum+temp;
        }

        // first thread writes the result
        if (thread_lane == 0)
            y[row] = ValueType(sdata[threadIdx.x]);
    }
}
