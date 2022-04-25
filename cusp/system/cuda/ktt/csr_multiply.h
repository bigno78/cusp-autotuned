#pragma once 

#include <iostream>

#include <cusp/ktt/detail/external/nameof.hpp>
#include <cusp/ktt/detail/ktt.inl>
#include <cusp/system/cuda/arch.h> // max_active_blocks

#include <cusp/system/cuda/ktt/kernels/csr_kernel.h>

namespace cusp::system::cuda::ktt {
    ::ktt::KernelId get_kernel_id(::ktt::KernelDefinitionId id,  std::string kernel_name) {
        static ::ktt::KernelId kernel = cusp::ktt::detail::tuner->CreateSimpleKernel(kernel_name, id);   
        return kernel;
    }
}

template<>
::ktt::KernelId cusp::ktt::detail::get_kernel_id<cusp::csr_format>() {
    return cusp::system::cuda::ktt::get_kernel_id(0, ""); 
}

namespace cusp {

namespace system {

namespace cuda {

namespace ktt {

#define STR(str) #str
#define STRING(str) STR(str)

template <typename DerivedPolicy,
          typename MatrixType,
          typename VectorType1,
          typename VectorType2,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void multiply(cuda::execution_policy<DerivedPolicy>& exec,
              const MatrixType& A,
              const VectorType1& x,
              VectorType2& y,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce,
              cusp::csr_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    static_assert( std::is_same_v<cusp::device_memory, typename MatrixType::memory_space>
                   && std::is_same_v<cusp::device_memory, typename VectorType1::memory_space>
                   && std::is_same_v<cusp::device_memory, typename VectorType2::memory_space>, 
                   "All arguments must be in device memory." );

    std::cout << "Hello from out thing\n";

    // copied from csr_vector_spmv.h
    typedef typename MatrixType::row_offsets_array_type::const_iterator     RowIterator;
    typedef typename MatrixType::column_indices_array_type::const_iterator  ColumnIterator;
    typedef typename MatrixType::values_array_type::const_iterator          ValueIterator1;

    typedef typename VectorType1::const_iterator                            ValueIterator2;
    typedef typename VectorType2::iterator                                  ValueIterator3;

    auto& tuner = *cusp::ktt::detail::tuner;

    std::string path = STRING(CUSP_PATH) "/cusp/system/cuda/ktt/kernels/csr_kernel.h";

    const size_t THREADS_PER_BLOCK  = 128;
    const size_t THREADS_PER_VECTOR = 32;
    const size_t VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
    
    std::vector< std::string > type_names {
        std::string(NAMEOF_TYPE(typename MatrixType::index_type)),
        std::string(NAMEOF_TYPE(typename MatrixType::value_type)),
        std::string(NAMEOF_TYPE(typename VectorType1::value_type)),
        std::string(NAMEOF_TYPE(typename VectorType2::value_type)),
        std::to_string(VECTORS_PER_BLOCK),
        std::to_string(THREADS_PER_VECTOR)
    };

    

    // TODO: we probably wanna change this later
    const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(
                                  ktt_csr_vector_kernel<typename MatrixType::index_type, typename MatrixType::value_type,
                                  typename VectorType1::value_type, typename VectorType2::value_type,
                                  VECTORS_PER_BLOCK, THREADS_PER_VECTOR>, THREADS_PER_BLOCK, (size_t) 0);
    const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, DIVIDE_INTO(A.num_rows, VECTORS_PER_BLOCK));

    const ::ktt::DimensionVector blockDimensions(THREADS_PER_BLOCK);
    const ::ktt::DimensionVector gridDimensions(NUM_BLOCKS);

    ::ktt::KernelDefinitionId definition = tuner.GetKernelDefinitionId("ktt_csr_vector_kernel", type_names);

    if (definition == ::ktt::InvalidKernelDefinitionId) {
        definition = tuner.AddKernelDefinitionFromFile(
            "ktt_csr_vector_kernel", 
            path, 
            gridDimensions,
            blockDimensions,
            type_names);
    }

    ::ktt::KernelId kernel = get_kernel_id(definition, "CsrKernel");

    auto cast = [](auto* ptr) { return const_cast<void*>(static_cast<const void*>(ptr)); };

    auto add_arg = [&] (auto& array, ::ktt::ArgumentAccessType access = ::ktt::ArgumentAccessType::ReadOnly) {
        return tuner.AddArgumentVector(cast(array.data().get()),
                                       array.size(),
                                       sizeof(decltype(array.front())),
                                       access,
                                       ::ktt::ArgumentMemoryLocation::Device);
    };

    auto num_rows_id = tuner.AddArgumentScalar(A.num_rows);
    auto Ap_id = add_arg(A.row_offsets); 
    auto Aj_id = add_arg(A.column_indices);
    auto Ax_id = add_arg(A.values);
    auto x_id = add_arg(x);
    auto y_id = add_arg(y, ::ktt::ArgumentAccessType::ReadWrite);

    tuner.SetArguments(definition, { num_rows_id, Ap_id, Aj_id, Ax_id, x_id, y_id });

    tuner.Run(kernel, {}, {});
}

}

}

}

}
