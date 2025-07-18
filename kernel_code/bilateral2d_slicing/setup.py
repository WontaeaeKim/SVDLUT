import os
import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.chdir(osp.dirname(osp.abspath(__file__)))

setup(
    name='bilateral2D_slicing_wo_conv',
    packages=find_packages(),
    include_package_data=False,
    ext_modules=[
        CUDAExtension('bilateral2D_slicing_wo_conv', [
            'src/bilateral2D_slicing.cpp',
            'src/trilinear2D_slice_cpu.cpp',
            'src/trilinear2D_slice_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })