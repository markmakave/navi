#pragma once

namespace lumina {
namespace cuda {

class error
{
public:

    error(cudaError_t error);

    const char*
    message() const;

    operator bool() const;

    operator cudaError_t() const;

protected:

    cudaError_t _handle;
};

}
}
