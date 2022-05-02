#pragma once

namespace cusp {

namespace ktt {

template<typename Matrix>
void reset_tuning(const Matrix& matrix);

} // namespace ktt

} // namespace cusp

#include <cusp/ktt/detail/ktt.inl>
