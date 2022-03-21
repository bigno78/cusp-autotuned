#pragma once 

#include <iostream>

namespace cusp {

namespace system {

namespace cuda {

namespace ktt {

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
    auto& tuner = *cusp::ktt::detail::tuner;

    // const ::ktt::DimensionVector blockDimensions(32);
    // const ::ktt::DimensionVector gridDimensions(1);

    // std::vector<int> result(50, 0);

    // std::string path = "/home/bigno/school/cusp-autotuned/cusp/system/cuda/ktt/kernels/csr_kernel.h";

    // const ::ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("test_kernel", path, gridDimensions,
    //     blockDimensions);
    
    // const ::ktt::KernelId kernel = tuner.CreateSimpleKernel("Addition", definition);

    // auto arguemntId = tuner.AddArgumentScalar(5);
    // auto outId = tuner.AddArgumentVector(result, ::ktt::ArgumentAccessType::WriteOnly);

    // tuner.SetArguments(definition, { arguemntId, outId });

    // tuner.Run(kernel, {}, { ::ktt::BufferOutputDescriptor(outId, result.data()) });

    // // std::cout << result[0] << "\n";

    // Initialize device index and path to kernel.
    ::ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = "/home/bigno/school/cusp-autotuned/cusp/system/cuda/ktt/kernels/csr_kernel.h";

    // Declare kernel parameters and data variables.
    const size_t numberOfElements = 1024 * 1024;
    // Dimensions of block and grid are specified with DimensionVector. Only single dimension is utilized in this tutorial.
    // In general, DimensionVector supports up to three dimensions.
    const ::ktt::DimensionVector blockDimensions(256);
    const ::ktt::DimensionVector gridDimensions(numberOfElements / blockDimensions.GetSizeX());

    std::vector<float> a(numberOfElements);
    std::vector<float> b(numberOfElements);
    std::vector<float> result(numberOfElements, 0.0f);

    // Initialize data
    for (size_t i = 0; i < numberOfElements; ++i)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i + 1);
    }

    // Add new kernel definition. Specify kernel function name, path to source file, default grid dimensions and block dimensions.
    // ::ktt returns handle to the newly added definition, which can be used to reference it in other API methods.
    const ::ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("vectorAddition", kernelFile, gridDimensions,
        blockDimensions);

    // Add new kernel arguments to tuner. Argument data is copied from std::vector containers. Specify whether the arguments are
    // used as input or output. ::ktt returns handle to the newly added argument, which can be used to reference it in other API
    // methods. 
    const ::ktt::ArgumentId aId = tuner.AddArgumentVector(a, ::ktt::ArgumentAccessType::ReadOnly);
    const ::ktt::ArgumentId bId = tuner.AddArgumentVector(b, ::ktt::ArgumentAccessType::ReadOnly);
    const ::ktt::ArgumentId resultId = tuner.AddArgumentVector(result, ::ktt::ArgumentAccessType::WriteOnly);

    // Set arguments for the kernel definition. The order of argument ids must match the order of arguments inside corresponding
    // CUDA kernel function.
    tuner.SetArguments(definition, {aId, bId, resultId});

    // Create simple kernel from the specified definition. Specify name which will be used during logging and output operations.
    // In more complex scenarios, kernels can have multiple definitions. Definitions can be shared between multiple kernels.
    const ::ktt::KernelId kernel = tuner.CreateSimpleKernel("Addition", definition);

    // Set time unit used during printing of kernel duration. The default time unit is milliseconds, but since computation in
    // this tutorial is very short, microseconds are used instead.
    tuner.SetTimeUnit(::ktt::TimeUnit::Microseconds);

    // Run the specified kernel. The second argument is related to kernel tuning and will be described in further tutorials.
    // In this case, it remains empty. The third argument is used to retrieve the kernel output. For each kernel argument that
    // is retrieved, one BufferOutputDescriptor must be specified. Each of these descriptors contains id of the retrieved argument
    // and memory location where the argument data will be stored. Optionally, it can also include number of bytes to be retrieved,
    // if only a part of the argument is needed. Here, the data is stored back into result buffer which was created earlier. Note
    // that the memory location size needs to be equal or greater than the retrieved argument size.
    tuner.Run(kernel, {}, {::ktt::BufferOutputDescriptor(resultId, result.data())});

    // Print first ten elements from the result to check they were computed correctly.
    std::cout << "Printing the first 10 elements from result: ";

    for (size_t i = 0; i < 10; ++i)
    {
        std::cout << result[i] << " ";
    }

    std::cout << std::endl;

    cusp::ktt::detail::tuner.reset();
}

}

}

}

}
