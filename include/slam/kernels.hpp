#pragma once

#include "base/matrix.hpp"
#include "base/array.hpp"
#include "base/color.hpp"
#include "slam/brief.hpp"

namespace lm {
namespace slam {

void
detect(
    const matrix<lm::gray>& input, 
          matrix<bool>&     output
);

void
descript(
    const matrix<gray>&                 frame,
    const brief<256>&                   engine,
          array<feature>&               features
);

}
}
