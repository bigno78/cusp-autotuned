#include <cusp/array2d.h>
#include <cusp/convert.h>
#include <cusp/detail/utils.h>

#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h> // raw_pointer_cast

namespace cusp
{

namespace ktt
{


template<typename IndexType>
struct ell_row_length
{
    const IndexType* column_indices;
    size_t num_cols_per_row;
    size_t pitch;

    __host__ __device__ IndexType operator()(IndexType row_idx)
    {
        IndexType len = 0;

        while (len < num_cols_per_row
                && column_indices[row_idx + len*pitch] >= 0)
        {
            len++;
        }

        return len;
    }
};

template<typename IndexType, typename ValueType, typename MemorySpace>
void
compute_row_lengths(cusp::ktt::ellr_matrix<IndexType, ValueType, MemorySpace>& A)
{
    thrust::counting_iterator<IndexType> row_idx_it(0);

    thrust::transform(
        row_idx_it,
        row_idx_it + A.num_rows,
        A.row_lengths.begin(),
        ell_row_length<IndexType>{
            thrust::raw_pointer_cast(&A.column_indices(0, 0)),
            A.column_indices.num_cols,
            A.column_indices.pitch }
    );
}


//////////////////
// Constructors //
//////////////////

template <typename IndexType, typename ValueType, class MemorySpace>
ellr_matrix<IndexType, ValueType, MemorySpace>
::ellr_matrix(const size_t num_rows, const size_t num_cols, const size_t num_entries,
             const size_t num_entries_per_row, const size_t alignment)
    : Parent(num_rows, num_cols, num_entries, num_entries_per_row, alignment)
{
    row_lengths.resize(num_rows);
}

// construct from a different matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
ellr_matrix<IndexType, ValueType, MemorySpace>
::ellr_matrix(const MatrixType& matrix)
{
    cusp::convert(matrix, *this);
    update_row_lengths();
}

//////////////////////
// Member Functions //
//////////////////////

template <typename IndexType, typename ValueType, class MemorySpace>
void
ellr_matrix<IndexType, ValueType, MemorySpace>
::swap(ellr_matrix& matrix)
{
    Parent::swap(matrix);
    row_lengths.swap(matrix.row_lengths);
}

template <typename IndexType, typename ValueType, class MemorySpace>
void
ellr_matrix<IndexType, ValueType, MemorySpace>
::resize(const size_t num_rows, const size_t num_cols, const size_t num_entries,
         const size_t num_entries_per_row)
{
    Parent::resize(num_rows, num_cols, num_entries, num_entries_per_row);
    row_lengths.resize(num_rows);
}

template <typename IndexType, typename ValueType, class MemorySpace>
void
ellr_matrix<IndexType, ValueType, MemorySpace>
::resize(const size_t num_rows, const size_t num_cols, const size_t num_entries,
         const size_t num_entries_per_row, const size_t alignment)
{
    Parent::resize(num_rows, num_cols, num_entries,
                   num_entries_per_row, alignment);
    row_lengths.resize(num_rows);
}

// assignment from another matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
ellr_matrix<IndexType, ValueType, MemorySpace>&
ellr_matrix<IndexType, ValueType, MemorySpace>
::operator=(const MatrixType& matrix)
{
    cusp::convert(matrix, *this);
    update_row_lengths();

    return *this;
}

template <typename IndexType, typename ValueType, class MemorySpace>
void ellr_matrix<IndexType, ValueType, MemorySpace>::update_row_lengths()
{
    if (row_lengths.size() != this->num_rows)
        row_lengths.resize(this->num_rows);

    compute_row_lengths(*this);
}


} // namespace ktt

} // namespace cusp
