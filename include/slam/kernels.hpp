#pragma once

#include "base/matrix.hpp"
#include "base/array.hpp"
#include "base/color.hpp"
#include "slam/brief.hpp"

namespace lumina {
namespace slam {

void
detect(
    const matrix<lumina::gray>& input,
    const int                   threshold,
          matrix<bool>&         output
);

void
descript(
    const matrix<gray>&                 frame,
    const brief<256>&                   engine,
          array<feature>&               features
);

}
}
