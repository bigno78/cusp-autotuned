#pragma once

#include <Ktt.h>

#include <cusp/array1d.h>

namespace cusp {

namespace ktt {

/**
 * @brief Perform the multiplication y = A*x using an autotuned kernel.
 * 
 * @tparam Matrix      The type of the sparse matrix to use.
 * @tparam ValueType1  The value type of the x vector.
 * @tparam ValueType2  The value type of the y vector.
 * 
 * @param A The matrix.
 * @param x The input vector.
 * @param y The output vector.
 */
template <typename Matrix,
          typename ValueType1,
          typename ValueType2>
void multiply(const Matrix& A,
              const cusp::array1d<ValueType1, cusp::device_memory>& x,
              cusp::array1d<ValueType2, cusp::device_memory>& y);

// NOTE: Should we have this?
// /**
//  * @brief Perform the multiplication y = A*x without autotuning,
//  * instead just run the kernel with the given configuration.
//  * 
//  * @tparam Matrix      The type of the sparse matrix to use.
//  * @tparam ValueType1  The value type of the x vector.
//  * @tparam ValueType2  The value type of the y vector.
//  * 
//  * @param A The matrix.
//  * @param x The input vector.
//  * @param y The output vector.
//  */
// template <typename Matrix,
//           typename ValueType1,
//           typename ValueType2>
// void multiply(const Matrix& A,
//               const cusp::array1d<ValueType1, cusp::device_memory>& x,
//               cusp::array1d<ValueType2, cusp::device_memory>& y,
//               const ::ktt::KernelConfiguration& configuration);

/**
 * TODO: Add some documentation.
 * 
 * @brief
 * 
 * @tparam IndexType 
 * @tparam ValueType1 
 * @tparam ValueType2 
 * @tparam ValueType3 
 * @tparam Format 
 */
template <typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3,
          typename Format>
void reset_tuning();

/**
 * TODO: Add some documentation.
 * 
 * @brief 
 * 
 * @tparam MatrixType 
 * @tparam ValueType1 
 * @tparam ValueType2 
 * 
 * @param A 
 * @param x 
 * @param y 
 */
template <typename MatrixType,
          typename ValueType1,
          typename ValueType2>
void reset_tuning(const MatrixType& A,
                  const cusp::array1d<ValueType1, cusp::device_memory>& x,
                  cusp::array1d<ValueType2, cusp::device_memory>& y);

} // namespace ktt

} // namespace cusp

#include <cusp/ktt/detail/ktt.inl>
