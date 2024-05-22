#pragma once

#include <cusp/detail/config.h>
#include <cusp/array1d.h>

#include <optional>

#include <Ktt.h>

namespace cusp {

namespace ktt {

inline void disable();

inline void enable();

inline ::ktt::Tuner& get_tuner();


/**
 * @brief Perform the multiplication y = A*x using an autotuned kernel.
 * This function performs a single step of dynamic autotuning.
 *
 * @tparam Matrix      The type of the sparse matrix to use.
 * @tparam IndexType   The type used for indices.
 * @tparam ValueType1  The value type of the matrix.
 * @tparam ValueType2  The value type of the x vector.
 * @tparam ValueType3  The value type of the y vector.
 *
 * @param A The matrix.
 * @param x The input vector.
 * @param y The output vector.
 */
template <template<typename, typename, typename> typename Matrix,
          typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3>
::ktt::KernelResult multiply(
     const Matrix<IndexType, ValueType1, cusp::device_memory>& A,
     const cusp::array1d<ValueType2, cusp::device_memory>& x,
     cusp::array1d<ValueType3, cusp::device_memory>& y);


/**
 * @brief Perform the multiplication y = A*x without autotuning,
 * instead just run the kernel in the given configuration.
 *
 * @tparam Matrix      The type of the sparse matrix to use.
 * @tparam IndexType   The type used for indices.
 * @tparam ValueType1  The value type of the matrix.
 * @tparam ValueType2  The value type of the x vector.
 * @tparam ValueType3  The value type of the y vector.
 *
 * @param A                   The matrix.
 * @param x                   The input vector.
 * @param y                   The output vector.
 * @param configuration       The configuration of the kernel.
 * @param run_with_profiling  Flag used to toggle profiling.
 */
template <template<typename, typename, typename> typename Matrix,
          typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3>
::ktt::KernelResult multiply(
     const Matrix<IndexType, ValueType1, cusp::device_memory>& A,
     const cusp::array1d<ValueType2, cusp::device_memory>& x,
     cusp::array1d<ValueType3, cusp::device_memory>& y,
     const ::ktt::KernelConfiguration& configuration,
     bool run_with_profiling = false);


/**
 * @brief Perform offline autotuning on the specified matrix and vectors.
 *
 * @tparam Matrix      The type of the sparse matrix to use.
 * @tparam IndexType   The type used for indices.
 * @tparam ValueType1  The value type of the matrix.
 * @tparam ValueType2  The value type of the x vector.
 * @tparam ValueType3  The value type of the y vector.
 *
 * @param A                      The matrix.
 * @param x                      The input vector.
 * @param y                      The output vector.
 * @param reference_computation  The reference computation that computes
                                 the correct result.
 */
template <template<typename, typename, typename> typename Matrix,
          typename IndexType,
          typename ValueType1,
          typename ValueType2,
          typename ValueType3>
std::vector<::ktt::KernelResult>
tune(const Matrix<IndexType, ValueType1, cusp::device_memory>& A,
     const cusp::array1d<ValueType2, cusp::device_memory>& x,
     cusp::array1d<ValueType3, cusp::device_memory>& y,
     std::optional<::ktt::ReferenceComputation> reference_computation = std::nullopt,
     std::unique_ptr<::ktt::StopCondition> stop_condition = nullptr,
     std::unique_ptr<::ktt::Searcher> searcher = nullptr);


/**
 * @brief Resets the tuning for the kernel instantiation used for the given arguments.
 *
 * @tparam MatrixType    Type of the sparse matrix.
 * @tparam ValueType1    The value type of the x vector
 * @tparam ValueType2    The value type of the y vector.
 * @tparam MemorySpace1  The memory space of the x vector.
 * @tparam MemorySpace2  The memory space of the y vector.
 *
 * @param A The matrix.
 * @param x The input vector.
 * @param y The output vector.
 */
template <typename MatrixType,
          typename ValueType1,
          typename ValueType2,
          typename MemorySpace1,
          typename MemorySpace2>
void reset_tuning(const MatrixType& A,
                  const cusp::array1d<ValueType1, MemorySpace1>& x,
                  cusp::array1d<ValueType2, MemorySpace2>& y);


} // namespace ktt

} // namespace cusp

#include <cusp/ktt/detail/ktt.inl>
