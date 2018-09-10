#pragma once

#include <stk/image/volume.h>

namespace stk {

Volume read_itk_image(const std::string& file_name);

void write_itk_image(const Volume& volume, const std::string& file_name);

} // namespace stk
