from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the directory containing this script
this_dir = os.path.dirname(os.path.abspath(__file__))

# Define the source files for MXFP8
source_cuda = [os.path.join(this_dir, filename)
               for filename in ['mxfp8_quant.cpp',  # MXFP8 C++ binding
                                'mxfp8_quant_kernel.cu'  # MXFP8 CUDA implementation
                                ]]

# Setup configuration
setup(
    name='mxfp8Quant',  # Name of the package
    packages=find_packages(),
    cmdclass={'build_ext': BuildExtension},
    ext_modules=[
        CUDAExtension(
            'mxfp8Quant._C',  # Extension name
            source_cuda,  # Source files
            extra_compile_args={
                # Compiler flags
                'cxx': ['/std:c++latest'],  # For Windows: Use /std:c++latest for C++17 support
                'nvcc': ['-O2']  # Optimize CUDA code
            }
        )
    ]
)
