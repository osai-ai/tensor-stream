
# TensorStream
TensorStream is a C++ library for real-time video stream (e.g. RTMP) decoding to CUDA memory which support some additional features:
* CUDA memory conversion to ATen Tensor for using it via Python in [PyTorch Deep Learning models](#pytorch-example)
* Detecting basic video stream issues related to frames reordering/loss
* Video Post Processing (VPP) operations: downscaling/upscaling, color conversion from NV12 to RGB24/BGR24/Y800

The whole pipeline works on GPU.

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
* [PyTorch](https://github.com/pytorch/pytorch) to build C++ extension for Python
    * Stable version (1.0.0 or above) to build with CUDA 9
    * Latest stable version (1.0.1.post2 or above) to build with CUDA 10
* [Python](https://www.python.org/) 3.6 or above to build C++ extension for Python

### Binaries
Extension for Python can be installed via pip (**Linux only**):
 - **CUDA 9:**
```
pip install https://tensorstream.argus-ai.com/wheel/cu9/linux/tensor_stream-0.1.6-cp36-cp36m-linux_x86_64.whl
```
- **CUDA 10:**
```
pip install https://tensorstream.argus-ai.com/wheel/cu10/linux/tensor_stream-0.1.6-cp36-cp36m-linux_x86_64.whl
```

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
To build TensorStream on Windows, Visual Studio 2017 14.11 toolset is needed

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

## Docker image
Dockerfiles can be found in [docker](docker) folder. Please note that for different CUDAs different Dockerfiles are required. To distinguish them name suffix is used, i.e. for **CUDA 9** Dockerfile name  is Dockerfile_**cu9**, for **CUDA 10** Dockerfile_**cu10** and so on. 
```
docker build -t tensorstream -f docker/Dockerfile_cu10 .
```
Run with bash command line and follow C++ extension for Python [installation guide](#install-tensorstream)
```
nvidia-docker run -ti tensorstream bash
```

## Usage

### Samples

 1. Simple [example](python_examples/simple.py) demonstrates RTMP to PyTorch tensor conversion. Let's consider some usage scenarios:
 > **Note:** You can pass **--help** to get list of all available options, their description and default values

* Convert RTMP bitstream to RGB24 PyTorch tensor and dump result to dump.yuv file: 
```
python simple.py -i rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4 -fc RGB24 -o dump.yuv
```
> **Warning:** Dumps significantly affect performance

* The same scenario with downscaling:
```
python simple.py -i rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4 -fc RGB24 -w 720 -h 480 -o dump.yuv
```
* Number of frames can be limited by -n option:
```
python simple.py -i rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4 -fc RGB24 -w 720 -h 480 -o dump.yuv -n 100
```

2. [Example](python_examples/many_consumers.py) that demonstrates how to use TensorStream in case of several stream consumers:
> **Note:** You can pass **--help** to get list of all available options, their description and default values
```
python many_consumers.py -i rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4 -n 100
```

3. Using TensorStream with existing [fast-neural-style](https://github.com/pytorch/examples/tree/master/fast_neural_style) model to augment input frames

### PyTorch example

Simple example how to use TensorStream for Deep learning tasks:

```
reader = TensorStreamConverter(stream_url)
reader.initialize()
reader.start()
parameters = {
    'name': "RGB_reader",
    'delay': 0,
    'pixel_format': FourCC.RGB24,
    'return_index': False,
    'width': width,
    'height': height,
}

while need_predictions:
    tensor = reader.read(**parameters)  # tensor dtype is torch.uint8, device is cuda
    prediction = model(tensor)
```
Initialize tensor stream with video file (e.g. local or network video) and start reading it in separate process. Get last frame from read part of stream and do prediction.
> **Note:** All tasks inside TensorStream processed on GPU, so output tensor also located on GPU.

## Documentation
Documentation for Python and C++ API can be found on the [site](https://tensorstream.argus-ai.com/).
## License

TensorStream is LGPL-2.1 licensed, see LICENSE file for details.

### Used materials in samples

[Big Buck Bunny](https://peach.blender.org/)  is licensed under the  [Creative Commons Attribution 3.0 license](http://creativecommons.org/licenses/by/3.0/).
(c) copyright 2008, Blender Foundation /  [www.bigbuckbunny.org](http://www.bigbuckbunny.org/)
