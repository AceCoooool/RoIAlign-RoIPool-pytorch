from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(
    name='roi_pool_cpp',
    ext_modules=[
        CppExtension('roi_pool_cpu', [
            'roi_pool_binding.cpp'
        ]),
        CUDAExtension('roi_pool_cuda', [
            'roi_pool_cuda.cpp',
            'roi_pool_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
