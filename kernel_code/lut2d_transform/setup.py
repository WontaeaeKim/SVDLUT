import os
import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.chdir(osp.dirname(osp.abspath(__file__)))

setup(
    name='lut2D_transform',
    packages=find_packages(),
    include_package_data=False,
    ext_modules=[
        CUDAExtension('lut2D_transform', [
            'src/lut2D_transform.cpp',
            'src/trilinear2D_cpu.cpp',
            'src/trilinear2D_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })