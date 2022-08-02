#pragma once 

#include <cusp/system/cuda/ktt/kernel.h>
//#include <cusp/system/cuda/ktt/csr_multiply.h>
#include <cusp/system/cuda/ktt/dia_multiply.h>

namespace cusp {

namespace system {

namespace cuda {

namespace ktt {

// TODO: should this be here??
// Helpful function to get a kernel used for a given multiplication
// instead of having to manually specify the types.
template <typename MatrixType,
          typename ValueType1,
          typename ValueType2>
kernel_context get_kernel(::ktt::Tuner& tuner,
                          const MatrixType& A,
                          const cusp::array1d<ValueType1, cusp::device_memory>& x,
                          cusp::array1d<ValueType2, cusp::device_memory>& y)
{
    using IndexType = typename MatrixType::index_type;
    using ValueType = typename MatrixType::value_type;
    using FormatType = typename MatrixType::format;

    return get_kernel<IndexType, ValueType, ValueType1, ValueType2>(tuner, FormatType{});
}

} // namespace ktt

} // namespace cuda

} // namespace system

} // namespace cusp
