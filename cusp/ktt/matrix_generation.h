#pragma once

#include <cusp/dia_matrix.h>

#include <algorithm> // min

namespace cusp {

namespace ktt {

/**
 * @brief Create a symetric matrix with given number of diagonals filled with ones.
 *
 * The main diagonal is always filled and then equal number of diagonals is placed
 * on both sides of it. That means the matrix is actually symetric only for
 * odd numbers of diagonals.
 *
 * If the number of diagonals is greater then would fit into the matrix,
 * the extra diagonals are just discarded.
 *
 * @param rows            The number of rows of the resulting matrix.
 * @param cols            The number of columns of the resulting matrix.
 * @param offset_step     Distance between two diagonals.
 * @param diagonal_count  The number of non-zero diagonals in the matrix.
 *
 * @return The matrix.
 */
cusp::dia_matrix<int, float, cusp::host_memory>
make_diagonal_symmetric_matrix(int rows,
                               int cols,
                               int offset_step,
                               int diagonal_count)
{
    using DiaMatrix = cusp::dia_matrix<int, float, cusp::host_memory>;

    DiaMatrix::diagonal_offsets_array_type offsets;
    offsets.reserve(diagonal_count);

    DiaMatrix::values_array_type values(rows, diagonal_count, 0);

    int starting_offset = -offset_step * diagonal_count/2;
    size_t num_entries = 0;

    for (int i = 0; i < diagonal_count; ++i) {
        int offset = starting_offset + offset_step*i;

        int starting_row = offset < 0 ? -offset : 0;
        int starting_col = offset < 0 ? 0 : offset;

        if (starting_row >= rows || starting_col >= cols)
            continue;

        offsets.push_back(offset);

        int ending_row = starting_row + std::min(rows - starting_row, cols - starting_col);

        for (int row = starting_row; row < ending_row; ++row) {
            num_entries++;
            values(row, i) = 1;
        }
    }

    DiaMatrix res;

    res.num_rows = rows;
    res.num_cols = cols;
    res.num_entries = num_entries;

    res.diagonal_offsets.swap(offsets);
    res.values.swap(values);

    return res;
}

} // namespace ktt

} // namespace cusp
