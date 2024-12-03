#pragma once

#include "base/matrix.hpp"
#include "base/color.hpp"
#include "slam/brief.hpp"

namespace lumina::slam {

void
detect(
    const matrix<gray>& input,
    const int           threshold,
          matrix<bool>& output
);

void descript(
    const matrix<gray>&         frame,
    const brief<256>&           engine,
          tensor<1, feature>&   features
);

}
