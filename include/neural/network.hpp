/*

    Copyright (c) 2023 Mokhov Mark

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.

*/

#pragma once

#include "base/matrix.hpp"
#include "base/array.hpp"
#include "base/blas.hpp"

#include "neural/node.hpp"

#include <cassert>
#include <random>

namespace lumina::neural {

class network
{
public:

    network() {}
    ~network() {
        for (node* n : _nodes) {
            delete n;
        }
    }

    void
    forward() {
        for (node* n : _nodes) {
            n->forward();
        }
    }

    void
    backward() {
        for (node* n : _nodes) {
            n->backward();
        }
    }

protected:

    array<node*> _nodes;
};

} // namespace lumina::neural
