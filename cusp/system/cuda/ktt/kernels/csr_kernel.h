
template <typename Idx, typename Val1, typename Val2, typename Val3>
__device__
void csr_kernel(const unsigned int num_rows,
                const Idx*   Ar,
                const Idx*   Ac,
                const Val1*  Ax,
                const Val2*  x,
                Val3*        y)
{
    const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    Val1 value;
    if (idx < num_rows)
    {
        Idx row_start = Ar[idx];
        Idx row_end = Ar[idx + 1];
        for (int i = row_start; i < row_end; ++i)
        {
            Val3 value = Ax[i] * x[Ac[i]];
            y[idx] += value;
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
