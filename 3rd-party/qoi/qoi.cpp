
#define QOI_MALLOC(sz) operator new(sz)
#define QOI_FREE(p) operator delete(p)

#define QOI_IMPLEMENTATION
#include "qoi.hpp"
#undef QOI_IMPLEMENTATION
