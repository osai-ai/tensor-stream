import os
import io
import re
import platform
from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = read('README.md')

VERSION = find_version('video_reader', '__init__.py')

include_path = torch.utils.cpp_extension.include_paths(cuda=True)
include_path += ["include/"]
include_path += ["include/Wrappers/"]
if (platform.system() == 'Windows'):
    if (not os.getenv('FFMPEG_PATH')):
        raise RuntimeError("Please set FFmpeg root folder path to FFMPEG_PATH variable.")
    include_path += [os.getenv('FFMPEG_PATH') + "include"]

library_path = torch.utils.cpp_extension.library_paths(cuda=True)
if (platform.system() == 'Windows'):
    library_path += [os.getenv('FFMPEG_PATH') + "bin"]

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
if (platform.system() == 'Windows'):
    library += ["caffe2"]
    library += ["torch"]
    library += ["caffe2_gpu"]
    library += ["_C"]

app_src_path = []
app_src_path += ["src/Decoder.cpp"]
app_src_path += ["src/General.cpp"]
app_src_path += ["src/Kernels.cu"]
app_src_path += ["src/Parser.cpp"]
app_src_path += ["src/VideoProcessor.cpp"]
app_src_path += ["src/Wrappers/WrapperPython.cpp"]

setup(
    name='video_reader',
    version=VERSION,
    author='Bykadorov Roman',
    description='Stream video reader',
    long_description=readme,
    ext_modules=[
        Extension(
            name='VideoReader',
            sources=app_src_path,
            include_dirs=include_path,
            library_dirs=library_path,
            libraries=library,
            extra_compile_args=['-g'],
            language='c++')
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=find_packages(),
    zip_safe=True,
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=['torch>=0.4.0']
)
