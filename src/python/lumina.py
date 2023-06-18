import os
os.environ["EXTRA_CLING_ARGS"] = "-O3"

import cppyy

cppyy.add_include_path(os.path.abspath('/home/mark/dev/lumina/include'))
cppyy.add_include_path(os.path.abspath('/usr/local/cuda/include'))
cppyy.include('lumina.hpp')

from cppyy.gbl import lumina

class tensor:
    def __init__(self, shape, dtype = float):
        self.data = lumina.tensor[len(shape), dtype, lumina.heap_allocator[dtype]](*shape)

t = tensor((10, 10, 10, 10))
print(t.data.size())