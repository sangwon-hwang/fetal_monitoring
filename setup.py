import os
import re
import shutil
import torch

from distutils.core import Command
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import (BuildExtension, 
                                       CppExtension,
                                       CUDAExtension)


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


version_file = 'fetal_monitor/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def make_cuda_ext(name, module, sources, sources_cuda=[]):

    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtensions
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)

# TODO: fillup requirements
install_requires = []
setup_requires = []
tests_require = []
install_requires = []
extras_require = {}


if __name__ == '__main__':
    setup(
        name='fetal_monitor',
        version=get_version(),
        description='fetal_monitor',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='Cha_Mirae',
        keywords='',
        url='https://github.com/sangwon-hwang/conv',
        packages=find_packages(
            exclude=('configs', 'tools', 'demo', 'results', 'outputs', 'exps', 'docs')),
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        # license='Apache License 2.0',
        python_requires=">=3.7.0",
        install_requires=install_requires,
        ext_modules=[],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
