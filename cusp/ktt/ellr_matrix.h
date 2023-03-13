#pragma once

#include <cusp/detail/config.h>

#include <cusp/ell_matrix.h>

#include <type_traits>


namespace cusp
{

namespace ktt
{


template <typename IndexType, typename ValueType, typename MemorySpace>
class ellr_matrix : public cusp::ell_matrix<IndexType, ValueType, MemorySpace>
{

    using Parent = cusp::ell_matrix<IndexType, ValueType, MemorySpace>;

public:

    using row_lengths_array_type = cusp::array1d<size_t, MemorySpace>;

    row_lengths_array_type row_lengths;

    /*! Construct an empty \p ellr_matrix.
     */
    ellr_matrix(void) {}

    /*! Construct an \p ell_matrix with a specific shape, number of nonzero entries,
     *  and maximum number of nonzero entries per row.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     *  \param num_entries_per_row Maximum number of nonzeros per row.
     *  \param alignment Amount of padding used to align the data structure (default 32).
     */
    ellr_matrix(const size_t num_rows, const size_t num_cols,
                const size_t num_entries, const size_t num_entries_per_row,
                const size_t alignment = 32);

    /*! Construct an \p ell_matrix from another matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    ellr_matrix(const MatrixType& matrix);

    /*! Resize matrix dimensions and underlying storage
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     *  \param num_entries_per_row Maximum number of nonzeros per row.
     */
    void resize(const size_t num_rows, const size_t num_cols, const size_t num_entries,
                const size_t num_entries_per_row);

    /*! Resize matrix dimensions and underlying storage
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     *  \param num_entries_per_row Maximum number of nonzeros per row.
     *  \param alignment Amount of padding used to align the data structure (default 32).
     */
    void resize(const size_t num_rows, const size_t num_cols, const size_t num_entries,
                const size_t num_entries_per_row, const size_t alignment);

    /*! Swap the contents of two \p ell_matrix objects.
     *
     *  \param matrix Another \p ell_matrix with the same IndexType and ValueType.
     */
    void swap(ellr_matrix& matrix);

    /*! Assignment from another matrix.
     *
     *  \tparam MatrixType Format type of input matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    ellr_matrix& operator=(const MatrixType& matrix);

    void update_row_lengths();
};


} // namespace ktt

} // namespace cusp

#include <cusp/ktt/detail/ellr_matrix.inl>
