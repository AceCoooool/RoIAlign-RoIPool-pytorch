from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(
    name='roi_align_cpp',
    ext_modules=[
        CppExtension('roi_align_cpu', [
            'roi_align_binding.cpp'
        ]),
        CUDAExtension('roi_align_cuda', [
            'roi_align_cuda.cpp',
            'roi_align_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
