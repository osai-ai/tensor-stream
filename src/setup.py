from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

include_path = []
include_path = torch.utils.cpp_extension.include_paths(cuda=True)
include_path += ["C:\\Users\\Home\\Desktop\\VideoReader\\external\\ffmpeg\\include"]

library_path = torch.utils.cpp_extension.library_paths(cuda=True)
library_path += ["C:\\Users\\Home\\Desktop\\VideoReader\\external\\ffmpeg\\bin\\"]
library = ["cudart"]
library += ["cuda"]
library += ["cudadevrt"]
library += ["cudart_static"]
library += ["caffe2"]
library += ["torch"]
library += ["caffe2_gpu"]
library += ["_C"]
library +=  ["avcodec"]
library += ["avdevice"]
library += ["avfilter"]
library += ["avformat"]
library += ["avutil"]
library += ["swresample"]
library += ["swscale"]

setup(
    name='Source',
    ext_modules=[
    	Extension(
   			name='Source',
   			sources=["Source.cpp", "VPP.cu"],
   			include_dirs=include_path,
   			library_dirs=library_path,
   			libraries=library,
   			language='c++')
        #CUDAExtension(
        #	name='Parser', 
        #	sources=['Parser.cpp'],
        #			  ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })