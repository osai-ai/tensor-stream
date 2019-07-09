
# TensorStream
TensorStream is a C++ library for real-time video stream (e.g., RTMP) decoding to CUDA memory which supports some additional features:
* CUDA memory conversion to ATen Tensor for using it via Python in [PyTorch Deep Learning models](#pytorch-example)
* Detecting basic video stream issues related to frames reordering/loss
* Video Post Processing (VPP) operations: downscaling/upscaling, color conversion from NV12 to RGB24/BGR24/Y800  
* Support Linux and Windows  

Simple example how to use TensorStream for deep learning tasks:

```python
from tensor_stream import TensorStreamConverter, FourCC, Planes

reader = TensorStreamConverter("rtmp://127.0.0.1/live", cuda_device=0)
reader.initialize()
reader.start()

while need_predictions:
    # read latest available frame from the stream 
    tensor = reader.read(pixel_format=FourCC.BGR24,
                         width=256,
                         height=256,
                         normalization=True,
                         planes_pos=Planes.PLANAR)
                         
    # tensor dtype is torch.float32, device is 'cuda:0', shape is (3, 256, 256)
    prediction = model(tensor)
```

* Initialize tensor stream with a video (e.g., a local file or a network video stream) and start reading it in a separate process.

* Get latest available frame from the stream and make a prediction.

> **Note:** All tasks inside TensorStream processed on a GPU, so output tensor also located on the GPU.


## Table of Contents
 - [Installation](#install-tensorstream)
 - [Usage](#usage)
 - [Docker](#docker-image)
 - [Documentation](#documentation)
 - [License](#license)

## Install TensorStream

### Dependencies
* [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) 9.0 or above
* [FFmpeg](https://github.com/FFmpeg/FFmpeg) and FFmpeg version of headers required to interface with Nvidias codec APIs
[nv-codec-headers](https://github.com/FFmpeg/nv-codec-headers)
* [PyTorch](https://github.com/pytorch/pytorch) 1.1.0 or above to build C++ extension for Python
* [Python](https://www.python.org/) 3.6 or above to build C++ extension for Python

It is convenient to use TensorStream in Docker containers. The provided [Dockerfiles](#docker-image) is supplied to create an image with all the necessary dependencies.

### Installation from source

#### TensorStream source code

```
git clone -b master --single-branch https://github.com/Fonbet/argus-tensor-stream.git
cd argus-tensor-stream
```

#### C++ extension for Python

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
To build TensorStream on Windows, Visual Studio 2017 14.11 toolset is required

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

### Binaries (Linux only)
Extension for Python can be installed via pip:

- **CUDA 9:**
```
pip install https://tensorstream.argus-ai.com/wheel/cu9/linux/tensor_stream-0.1.8-cp36-cp36m-linux_x86_64.whl
```
- **CUDA 10:**
```
pip install https://tensorstream.argus-ai.com/wheel/cu10/linux/tensor_stream-0.1.8-cp36-cp36m-linux_x86_64.whl
```

#### Building examples and tests
Examples for Python and C++ can be found in [c_examples](c_examples) and [python_examples](python_examples) folders.  Tests for C++ can be found in [tests](tests) folder.
#### Python example 
Can be executed via Python after TensorStream [C++ extension for Python](#c-extension-for-python) installation.
```
cd python_examples
python simple.py
```
#### C++ example and unit tests
On Linux
```
cd c_examples  # tests
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$PWD/../../cmake ..
```
On Windows
```
set FFMPEG_PATH="Path to FFmpeg install folder"
cd c_examples or tests
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=%cd%\..\..\cmake -G "Visual Studio 15 2017 Win64" -T v141,version=14.11 ..
```

## Docker image
Dockerfiles can be found in [docker](docker) folder. Please note that for different CUDAs different Dockerfiles are required. To distinguish them name suffix is used, i.e. for **CUDA 9** Dockerfile name is Dockerfile_**cu9**, for **CUDA 10** Dockerfile_**cu10** and so on. 
```
docker build -t tensorstream -f docker/Dockerfile_cu10 .
```
Run with bash command line and follow [installation guide](#install-tensorstream)
```
nvidia-docker run -ti tensorstream bash
```

## Usage

### Samples

1. Simple [example](python_examples/simple.py) demonstrates RTMP to PyTorch tensor conversion. Let's consider some usage scenarios:
> **Note:** You can pass **--help** to get the list of all available options, their description and default values

* Convert an RTMP bitstream to RGB24 PyTorch tensors and dump the result to a dump.yuv file: 
```
python simple.py -i rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4 -fc RGB24 -o dump.yuv
```
> **Warning:** Dumps significantly affect performance

* The same scenario with downscaling with nearest resize algorithm:
```
python simple.py -i rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4 -fc RGB24 -w 720 -h 480 --resize_type NEAREST -o dump.yuv
```
* Number of frames to process can be limited by -n option:
```
python simple.py -i rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4 -fc RGB24 -w 720 -h 480 -o dump.yuv -n 100
```
* Output pixels format can be either torch.float32 or torch.uint8 depending on normalization flag is set or not:
```
python simple.py -i rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4 -fc RGB24 -w 720 -h 480 -o dump.yuv -n 100 --normalize
```
* Color planes in case of RGB can be either planar or merged and can be set via --planes option:
```
python simple.py -i rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4 -fc RGB24 -w 720 -h 480 -o dump.yuv -n 100 --planes MERGED
```
* Buffer size of processed frames via -bs or --buffer_size option:
```
python simple.py -i rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4 -fc RGB24 -w 720 -h 480 -o dump.yuv -n 100 --planes MERGED --buffer_size 5
```
> **Warning:** Buffer size should be less or equal to decoded picture buffer (DPB)
* GPU used for execution can be set via --cuda_device option:
```
python simple.py -i rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4 -fc RGB24 -w 720 -h 480 -o dump.yuv -n 100 --planes MERGED --cuda_device 0
```
* Logs types and levels can be configured with -v, -vd and --nvtx options. Check help to find available values and description:
```
python simple.py -i rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4 -fc RGB24 -w 720 -h 480 -o dump.yuv -n 100 --planes MERGED -v HIGH -vd CONSOLE --nvtx
```
2. [Example](python_examples/many_consumers.py) demonstrates how to use TensorStream in case of several stream consumers:
```
python many_consumers.py -i rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4 -n 100
```
3. [Example](python_examples/different_streams.py) demonstrates how to use TensorStream if several streams should be handled simultaneously:
```
python different_streams.py -i1 <path-to-first-stream> -i2 <path-to-second-stream> -n1 100 -n2 50 -v1 LOW -v2 HIGH --cuda_device1 0 --cuda_device2 1
```
> **Warning:** Default path to second stream is relative, so need to run different_streams.py from parent folder if no arguments are passing
### PyTorch example

Real-time video style transfer example: [fast-neural-style](python_examples/fast_neural_style).


## Documentation

Documentation for Python and C++ API can be found on the [site](https://tensorstream.argus-ai.com/).

## License

TensorStream is LGPL-2.1 licensed, see the [LICENSE](LICENSE) file for details.

### Used materials in samples

[Big Buck Bunny](https://peach.blender.org/)  is licensed under the  [Creative Commons Attribution 3.0 license](http://creativecommons.org/licenses/by/3.0/).
(c) copyright 2008, Blender Foundation /  [www.bigbuckbunny.org](http://www.bigbuckbunny.org/)