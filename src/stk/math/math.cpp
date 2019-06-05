#include "math.h"


namespace stk {


// Convert a Matrix3x3f to a row-major std::vector<float> representation
std::vector<float> to_vector(const Matrix3x3f& o)
{
    const auto ptr = reinterpret_cast<const float*>(&o._rows);
    return std::vector<float>(ptr, ptr + 9);
}

} // namespace stk
