#pragma once

namespace lm {
namespace cuda {

class stream
{
public:

    static const stream main;

public:

    stream();
    stream(const stream&)      = delete;
    stream(stream&&)           = delete;
    ~stream();

    void
    operator = (const stream&) = delete;
    void
    operator = (stream&&)      = delete;

    bool
    operator == (const stream& s);
    bool
    operator != (const stream& s);

    operator cudaStream_t() const;
    
    void
    synchronize() const;

private:

    stream(cudaStream_t handle);

private:

    cudaStream_t _handle;
};

}
}
