/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>
#include <cusp/detail/format.h>

#include <cusp/functional.h>

#include <cusp/system/detail/generic/multiply/generalized_spmv.h>
#include <cusp/system/detail/generic/multiply/generalized_spgemm.h>
#include <cusp/system/detail/generic/multiply/permute.h>
#include <cusp/system/detail/generic/multiply/spgemm.h>
#include <cusp/system/detail/generic/multiply/spmv.h>

#include <thrust/functional.h>

#include <cusp/ktt/ktt.h>
#include <cusp/system/cuda/ktt/multiply.h>

#include <type_traits>


namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

using namespace thrust::detail;
__THRUST_DEFINE_IS_CALL_POSSIBLE(has_member_operator_exec_impl, operator())

template <typename DerivedPolicy,
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
struct has_member_operator_exec
: has_member_operator_exec_impl<LinearOperator, void(thrust::execution_policy<DerivedPolicy>&,const MatrixOrVector1&,MatrixOrVector2&)>
{};

template <typename DerivedPolicy,
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
typename enable_if<
thrust::detail::and_<
  has_member_operator_exec<DerivedPolicy,LinearOperator,MatrixOrVector1,MatrixOrVector2>,
  thrust::detail::is_convertible<typename LinearOperator::format,cusp::unknown_format>
  >::value
>::type
multiply(thrust::execution_policy<DerivedPolicy> &exec,
         const LinearOperator&  A,
         const MatrixOrVector1& B,
               MatrixOrVector2& C)
{
    const_cast<LinearOperator&>(A)(exec, B, C);
}

template <typename DerivedPolicy,
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
typename enable_if<
thrust::detail::and_<
  thrust::detail::not_<has_member_operator_exec<DerivedPolicy,LinearOperator,MatrixOrVector1,MatrixOrVector2> >,
  thrust::detail::is_convertible<typename LinearOperator::format,cusp::unknown_format>
  >::value
>::type
multiply(thrust::execution_policy<DerivedPolicy> &exec,
         const LinearOperator&  A,
         const MatrixOrVector1& B,
               MatrixOrVector2& C)
{
    const_cast<LinearOperator&>(A)(B, C);
}

template <typename DerivedPolicy,
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
typename thrust::detail::disable_if_convertible<typename LinearOperator::format,cusp::unknown_format>::type
multiply(thrust::execution_policy<DerivedPolicy> &exec,
         const LinearOperator&  A,
         const MatrixOrVector1& B,
               MatrixOrVector2& C)
{
    typedef typename LinearOperator::value_type ValueType;

    cusp::constant_functor<ValueType> initialize(0);
    thrust::multiplies<ValueType> combine;
    thrust::plus<ValueType> reduce;

    cusp::multiply(exec, A, B, C, initialize, combine, reduce);
}


// The has_rebind type traits is used to check whether matrix is a view
template <typename T, typename /*U*/ = void>
struct has_rebind : std::false_type {};

template <typename T>
struct has_rebind<T, std::void_t<typename T::rebind<int>>> : std::true_type {};

template<typename T>
constexpr bool has_rebind_v = has_rebind<T>::value;


template <typename DerivedPolicy,
         typename Matrix,
         typename ValueType1,
         typename ValueType2>
typename thrust::detail::disable_if_convertible<
    typename Matrix::format,
    cusp::unknown_format
>::type
multiply(cusp::system::cuda::detail::execution_policy<DerivedPolicy> &exec,
         const Matrix& A,
         const cusp::array1d<ValueType1, cusp::device_memory>& x,
         cusp::array1d<ValueType2, cusp::device_memory>& y)
{
    using IndexType = typename Matrix::index_type;
    using ValueType = typename Matrix::value_type;

    if constexpr ((cusp::detail::is_ell<Matrix>::value
                    || cusp::detail::is_dia<Matrix>::value
                    && std::is_fundamental_v<IndexType>
                    && std::is_fundamental_v<ValueType>
                    && std::is_fundamental_v<ValueType1>
                    && std::is_fundamental_v<ValueType2>)
                    && has_rebind_v<Matrix>)
    {
        if (cusp::ktt::detail::is_enabled)
        {
            cusp::system::cuda::ktt::multiply(cusp::ktt::get_tuner(), A, x, y);
            return;
        }
    }

    // The call is not meant for KTT. Pass it on...

    cusp::constant_functor<ValueType> initialize(0);
    thrust::multiplies<ValueType> combine;
    thrust::plus<ValueType> reduce;

    cusp::multiply(exec, A, x, y, initialize, combine, reduce);
}


template <typename DerivedPolicy,
         typename LinearOperator,
         typename MatrixOrVector1,
         typename MatrixOrVector2,
         typename UnaryFunction,
         typename BinaryFunction1,
         typename BinaryFunction2>
typename thrust::detail::disable_if_convertible<UnaryFunction,cusp::known_format>::type
multiply(thrust::execution_policy<DerivedPolicy> &exec,
         const LinearOperator&  A,
         const MatrixOrVector1& B,
         MatrixOrVector2& C,
         UnaryFunction   initialize,
         BinaryFunction1 combine,
         BinaryFunction2 reduce)
{
    typedef typename LinearOperator::format  Format1;
    typedef typename MatrixOrVector1::format Format2;
    typedef typename MatrixOrVector2::format Format3;

    Format1 format1;
    Format2 format2;
    Format3 format3;

    multiply(thrust::detail::derived_cast(exec), A, B, C, initialize, combine, reduce, format1, format2, format3);
}

template <typename DerivedPolicy,
         typename LinearOperator,
         typename MatrixOrVector1,
         typename MatrixOrVector2,
         typename UnaryFunction,
         typename BinaryFunction1,
         typename BinaryFunction2>
void generalized_spgemm(thrust::execution_policy<DerivedPolicy> &exec,
                        const LinearOperator&  A,
                        const MatrixOrVector1& B,
                        MatrixOrVector2& C,
                        UnaryFunction   initialize,
                        BinaryFunction1 combine,
                        BinaryFunction2 reduce)
{
    typedef typename LinearOperator::format  Format1;
    typedef typename MatrixOrVector1::format Format2;
    typedef typename MatrixOrVector2::format Format3;

    Format1 format1;
    Format2 format2;
    Format3 format3;

    generalized_spgemm(exec, A, B, C, initialize, combine, reduce, format1, format2, format3);
}

template <typename DerivedPolicy,
         typename LinearOperator,
         typename Vector1,
         typename Vector2,
         typename Vector3,
         typename BinaryFunction1,
         typename BinaryFunction2>
void generalized_spmv(thrust::execution_policy<DerivedPolicy> &exec,
                      const LinearOperator&  A,
                      const Vector1& x,
                      const Vector2& y,
                      Vector3& z,
                      BinaryFunction1 combine,
                      BinaryFunction2 reduce)
{
    typedef typename LinearOperator::format  Format1;
    typedef typename Vector1::format         Format2;
    typedef typename Vector2::format         Format3;
    typedef typename Vector3::format         Format4;

    Format1 format1;
    Format2 format2;
    Format3 format3;
    Format4 format4;

    generalized_spmv(exec, A, x, y, z, combine, reduce, format1, format2, format3, format4);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp

