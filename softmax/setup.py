import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Helper to find the correct include path in Conda
conda_prefix = os.environ.get('CONDA_PREFIX')
include_dirs = []
library_dirs = []

if conda_prefix:
    # Primary conda include
    include_dirs.append(os.path.join(conda_prefix, 'include'))
    
    # Standard NVIDIA target directory
    target_include = os.path.join(conda_prefix, 'targets', 'x86_64-linux', 'include')
    if os.path.isdir(target_include):
        include_dirs.append(target_include)
        
        # CCCL include path
        cccl_include = os.path.join(target_include, 'cccl')
        if os.path.isdir(cccl_include):
            include_dirs.append(cccl_include)
    
    # Library directory
    target_lib = os.path.join(conda_prefix, 'targets', 'x86_64-linux', 'lib')
    if os.path.isdir(target_lib):
        library_dirs.append(target_lib)

setup(
    name='softmax_cpp',
    ext_modules=[
        CUDAExtension(
            'softmax_cpp',
            ['softmax.cu'],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_compile_args={
                'nvcc': [
                    '-std=c++17',
                    # Prevent system CUDA from interfering
                    '--no-host-device-initializer-list',
                    # Explicitly disable CUDA 13+ features
                    '-D__CUDA_NO_FP4_CONVERSIONS__',
                    '-D__CUDA_NO_FP6_CONVERSIONS__',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
