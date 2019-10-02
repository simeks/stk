#include "math.h"

Matrix3x3f Matrix3x3f::Identity {
    float3{1, 0, 0},
    float3{0, 1, 0},
    float3{0, 0, 1}
};

namespace stk {


// Convert a Matrix3x3f to a row-major std::vector<float> representation
std::vector<float> to_vector(const Matrix3x3f& o)
{
    const auto ptr = reinterpret_cast<const float*>(&o._rows);
    return std::vector<float>(ptr, ptr + 9);
}

} // namespace stk
