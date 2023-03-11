#pragma once

namespace lm {

template <typename T>
class heap_allocator {

    static
    void*
    allocate(size_t size)
    {
        void* ptr = operator new[](size * sizeof(T));

        if (ptr == nullptr)
            lm::log::error("memory allocation failed");

        return ptr;
    }

    static
    void
    deallocate(void* ptr)
    {
        operator delete[](ptr);
    }

};

template <typename T>
class stack_allocator {

    static
    void*
    allocate(size_t size)
    {
        void* ptr = alloca(size * sizeof(T));

        if (ptr == nullptr)
            lm::log::error("memory allocation failed");

        return ptr;
    }

    static
    void
    deallocate(void* ptr)
    {
        // nop
    }

}

}
