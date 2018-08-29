from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='Parser',
    ext_modules=[
         CUDAExtension(
        	name='Parser', 
        	sources=['Parser.cpp'],
        	#extra_compile_args={'cxx': ['-IC:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\include', 
        	#							'/LIBPATH:C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\lib\\x64',
        	#							'/link cuda.lib',
        	#							'/link python36.lib',
        	#							'/link _C.cp36-win_amd64.lib',
        	#							'/link _nvrtc.cp36-win_amd64.lib']}
        								)
    ],
    cmdclass={
        'build_ext': BuildExtension
    })