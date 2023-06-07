import pybind11
from distutils.core import setup, Extension

ext_modules = [
    Extension(
        'lumina', # название нашей либы
        ['pylib.cpp'], # файлики которые компилируем
        include_dirs=[pybind11.get_include(), '../../include/', '/usr/local/cuda/include/'],  # не забываем добавить инклюды pybind11
        language='c++',
        extra_compile_args=['-std=c++17'],  # используем с++11
    ),
]

setup(
    name='lumina',
    version='0.0.1',
    author='markmakave',
    author_email='markmakave@gmail.com',
    description='lumina python wrapper',
    ext_modules=ext_modules,
    requires=['pybind11']  # не забываем указать зависимость от pybind11
)