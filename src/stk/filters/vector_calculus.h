#pragma once

#include <stk/image/volume.h>

namespace stk {

Volume nabla(const Volume& displacement);

Volume divergence(const Volume& displacement);

Volume rotor(const Volume& displacement);

Volume circulation_density(const Volume& displacement);

} // namespace stk
