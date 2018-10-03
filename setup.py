from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)

include_path = torch.utils.cpp_extension.include_paths(cuda=True)
include_path += ["include/"]

library_path = torch.utils.cpp_extension.library_paths(cuda=True)
library = ["cudart"]
library += ["cuda"]
library += ["cudadevrt"]
library += ["cudart_static"]
library += ["avcodec"]
library += ["avdevice"]
library += ["avfilter"]
library += ["avformat"]
library += ["avutil"]
library += ["swresample"]
library += ["swscale"]

app_src_path = []
app_src_path += ["src/Decoder.cpp"]
app_src_path += ["src/General.cpp"]
app_src_path += ["src/Kernels.cu"]
app_src_path += ["src/Parser.cpp"]
app_src_path += ["src/Source.cpp"]
app_src_path += ["src/VideoProcessor.cpp"]


setup(
    name='VideoReader',
    ext_modules=[
        Extension(
            name='VideoReader',
            sources=app_src_path,
            include_dirs=include_path,
            library_dirs=library_path,
            libraries=library,
            language='c++')
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
