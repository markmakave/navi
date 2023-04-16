#pragma once

namespace lm {
namespace cuda {

class error
{
public:

    error(cudaError_t error);

    const char*
    describe() const;

    operator bool() const;

    operator cudaError_t() const;

protected:

    cudaError_t _handle;
};

}
}
