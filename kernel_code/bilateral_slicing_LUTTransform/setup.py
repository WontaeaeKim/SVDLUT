import os
import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.chdir(osp.dirname(osp.abspath(__file__)))

setup(
    name='bilateral2D_slicing_LUTTransform',
    packages=find_packages(),
    include_package_data=False,
    ext_modules=[
        CUDAExtension('bilateral2D_slicing_LUTTransform', [
            'src/bilateral2D_slicing_LUTTransform.cpp',
            'src/trilinear2D_slice_LUTTransform_cpu.cpp',
            'src/trilinear2D_slice_LUTTransform_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })