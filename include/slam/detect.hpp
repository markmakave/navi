#pragma once

#include <base/matrix.hpp>
#include <base/color.hpp>

namespace lm {
namespace slam {

void
detect(const matrix<lm::gray>& input, matrix<bool>& output);

}
}
