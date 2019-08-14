#pragma once

#include "types.h"

#include <stk/image/types.h>

#include <cmath>
#include <numeric>
#include <vector>

namespace stk {


// Arbitrary epsilon for floating-point comparisons
constexpr float eps()
{
    return 0.0001f;
}


// Convert a Matrix3x3f to a row-major std::vector<float> representation
std::vector<float> to_vector(const Matrix3x3f& o);


// Convert vector types to std::vector
template<typename T, int n = stk::type_id<T>::num_comp(), typename TBase = typename stk::type_id<T>::Base>
std::vector<TBase> to_vector(const T& o)
{
    const auto ptr = reinterpret_cast<const TBase*>(&o);
    return std::vector<TBase>(ptr, ptr + n);
}


// Check if any element is non-zero
template<typename T, typename TBase = typename stk::type_id<T>::Base>
bool nonzero(const T& o)
{
    const auto fold_op = [](TBase l, TBase r) { return l || (std::fabs(r) > eps()); };
    const auto v = to_vector(o);
    return std::accumulate(v.begin(), v.end(), false, fold_op);
}


} // namespace stk
