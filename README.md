# VideoReader README
VideoReader is a C++ library for video stream (e.g. RTMP) decoding to CUDA memory which support some additional features:
* CUDA memory conversion to ATen Tensor for using it via Python
* Detecting basic video stream issues related to frames reordering/loss
* VPP operations: downscaling/upscaling, color conversion from NV12 to RGB24/BGR24/Y800

The whole pipeline works on GPU.
## Installation
### Install dependencies
* [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) 9.0 or above
* [FFmpeg](https://github.com/FFmpeg/FFmpeg) and FFmpeg version of headers required to interface with Nvidias codec APIs
[nv-codec-headers](https://github.com/FFmpeg/nv-codec-headers)
* [Pytorch](https://github.com/pytorch/pytorch) 1.0 to build C++ extension for Python

To build VideoReader on Windows, Visual Studio 2017 14.11 toolset is needed
### VideoReader source code

```
git clone https://github.com/Fonbet/argus-video-reader.git
cd argus-video-reader
```
### Install VideoReader
#### C++ extenssion for Python

On Linux:
```
python setup.py install
```
On Windows:
```
set FFMPEG_PATH="Path to FFmpeg install folder"
set path=%path%;%FFMPEG_PATH%\bin
set VS150COMNTOOLS="Path to Visual Studio vcvarsall.bat folder"
call "%VS150COMNTOOLS%\vcvarsall.bat" x64 -vcvars_ver=14.11
python setup.py install
```
#### C++ library:

On Linux:
```
mkdir build
cd build
cmake ..
```
On Windows:
```
set FFMPEG_PATH="Path to FFmpeg install folder"
mkdir build
cd build
cmake -G "Visual Studio 15 2017 Win64" -T v141,version=14.11 ..
```
### Docker image
Dockerfile can be found at the top level of repository. Build as usual:
```
docker build -t videoreader .
```
### Building examples and tests
Examples for Python and C++ can be found in ```c_examples``` and ```python_exapmles``` folders.  Tests for C++ can found in ```tests ``` folder.
#### Python example 
Can be executed via Python after VideoReader [C++ extenssion for Python](#c-extenssion-for-python) installation.
```
python python_examples/sample.py
```
#### C++ example and unit tests
On Linux
```
cd c_examples or tests
mkdir build
cd build
cmake ..
```
On Windows
```
set FFMPEG_PATH="Path to FFmpeg install folder"
cd c_examples or tests
mkdir build
cd build
cmake -G "Visual Studio 15 2017 Win64" -T v141,version=14.11 ..
```
## Usage
Python example demonstrates RTMP to Pytorch tensor conversion. Let's consider some usage scenarios:
> **Note:** You can pass **--help** to get list of all available options, their description and default values
*  Convert RTMP bitstream to RGB24 Pytorch tensor and dump result to dump.yuv file: 
```
python sample.py -i rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4 -fc RGB24 -o dump.yuv
```
> **Warning:** Dumps significantly influent to performance
* The same scenario with downscaling:
```
python sample.py -i rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4 -fc RGB24 -n 100 -w 720 -h 480 -o dump.yuv
```
* Number of frames can be limited by -n option:
```
python sample.py -i rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4 -fc RGB24 -n 100 -w 720 -h 480 -o dump.yuv -n 100
```
