# TensorStream
TensorStream is a C++ library for real-time video streams (e.g., RTMP) decoding to CUDA memory which supports some additional features:
* CUDA memory conversion to ATen Tensor for using it via Python in [PyTorch Deep Learning models](#pytorch-example)
* Detecting basic video stream issues related to frames reordering/loss
* Video Post Processing (VPP) operations: downscaling/upscaling, crops, color conversions, etc.

The library supports both Linux and Windows.

Simple example how to use TensorStream for deep learning tasks:

```python
from tensor_stream import TensorStreamConverter, FourCC, Planes

reader = TensorStreamConverter("rtmp://127.0.0.1/live", cuda_device=0)
reader.initialize()
reader.start()

while need_predictions:
    # read the latest available frame from the stream
    tensor = reader.read(pixel_format=FourCC.BGR24,
                         width=256,                 # resize to 256x256 px
                         height=256,
                         normalization=True,        # normalize to range [0, 1]
                         planes_pos=Planes.PLANAR)  # dimension order [C, H, W]

    # tensor dtype is torch.float32, device is 'cuda:0', shape is (3, 256, 256)
    prediction = model(tensor.unsqueeze(0))
```

* Initialize tensor stream with a video (e.g., a local file or a network video stream) and start reading it in a separate process.

* Get the latest available frame from the stream and make a prediction.

> **Note:** All tasks inside TensorStream processed on a GPU, so the output tensor is also located on the GPU.


## Table of Contents
 - [Installation](#install-tensorstream)
 - [Usage](#usage)
 - [Docker](#docker-image)
 - [Documentation](#documentation)
 - [License](#license)

## Install TensorStream

### Dependencies
* [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) 11.8 or above
* [FFmpeg](https://github.com/FFmpeg/FFmpeg) 6.0 or above
* [nv-codec-headers](https://github.com/FFmpeg/nv-codec-headers) FFmpeg version of headers required to interface with Nvidias codec APIs
* [PyTorch](https://github.com/pytorch/pytorch) 2.0 or above to build C++ extension for Python
* [Python](https://www.python.org/) 3.6 or above to build C++ extension for Python

It is convenient to use TensorStream in Docker containers. The provided [Dockerfiles](#docker-image) is supplied to create an image with all the necessary dependencies.

### Installation from source

#### TensorStream source code

```
git clone -b master --single-branch https://github.com/osai-ai/tensor-stream.git
cd tensor-stream
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
cmake ..
```
### Building examples and tests
Examples for Python and C++ can be found in [c_examples](c_examples) and [python_examples](python_examples) folders.  Tests for C++ can be found in [tests](tests) folder.
#### Python example
Can be executed via Python after TensorStream [C++ extension for Python](#c-extension-for-python) installation.
```
cd python_examples
python simple.py
```
#### C++ example and unit tests
On Linux:
```
cd c_examples  # tests
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$PWD/../../cmake ..
```
On Windows:
```
set FFMPEG_PATH="Path to FFmpeg install folder"
cd c_examples or tests
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=%cd%\..\..\cmake ..
```

## Docker image
To build TensorStream need to pass Pytorch version via TORCH_VERSION argument:
```
docker build --build-arg TORCH_VERSION=2.0 -t tensorstream .
```
Run with a bash command line and follow the [installation guide](#install-tensorstream)
```
docker run --gpus=all -ti tensorstream bash
```

## Usage

### Python examples

1. Simple [example](python_examples/simple.py) demonstrates RTMP to PyTorch tensor conversion. Let's consider some usage scenarios:
> **Note:** You can pass **--help** to get the list of all available options, their description and default values

* Convert an RTMP bitstream to RGB24 PyTorch tensors and dump the result to a dump.yuv file:
```
python simple.py -i tests/resources/bunny.mp4 -fc RGB24 -o dump
```
> **Warning:** Dumps significantly affect performance. Suffix .yuv will be added to the output filename.

* The same scenario with downscaling with nearest resize algorithm:
```
python simple.py -i tests/resources/bunny.mp4 -fc RGB24 -w 720 -h 480 --resize_type NEAREST -o dump
```
> **Note:** Besides nearest resize algorithm, bilinear, bicubic and area (similar to OpenCV INTER_AREA) algorithms are available.

> **Warning:** Resize algorithms applied to NV12, so b2b with popular frameworks, which perform resize on other than NV12 format, aren't guaranteed.
* Number of frames to process can be limited by -n option:
```
python simple.py -i tests/resources/bunny.mp4 -fc RGB24 -w 720 -h 480 -o dump -n 100
```
* The result file can be cropped via --crop option which takes coordinates of left top and right bottom corners as parameters:
```
python simple.py -i tests/resources/bunny.mp4 -fc RGB24 -w 720 -h 480 --crop 0,0,320,240 -o dump -n 100
```
>**Warning:** Crop is applied before resize algorithm.
* Output pixels format can be either torch.float32 or torch.uint8 depending on normalization option which can be True, False or not set so TensorStream will decide which value should be used:
```
python simple.py -i tests/resources/bunny.mp4 -fc RGB24 -w 720 -h 480 -o dump -n 100 --normalize True
```
* Color planes in case of RGB can be either planar or merged and can be set via --planes option:
```
python simple.py -i tests/resources/bunny.mp4 -fc RGB24 -w 720 -h 480 -o dump -n 100 --planes MERGED
```
* Buffer size of processed frames via -bs or --buffer_size option:
```
python simple.py -i tests/resources/bunny.mp4 -fc RGB24 -w 720 -h 480 -o dump -n 100 --planes MERGED --buffer_size 5
```
> **Warning:** Buffer size should be less or equal to decoded picture buffer (DPB)
* GPU used for execution can be set via --cuda_device option:
```
python simple.py -i tests/resources/bunny.mp4 -fc RGB24 -w 720 -h 480 -o dump -n 100 --planes MERGED --cuda_device 0
```
* Input stream reading mode can be chosen with --framerate_mode option. Check help to find available values and description:
```
python simple.py -i tests/resources/bunny.mp4 -fc RGB24 -w 720 -h 480 -o dump -n 100 --planes MERGED --framerate_mode NATIVE
```
* Bitstream analyze stage can be skipped to decrease latency with --skip_analyze flag:
```
python simple.py -i tests/resources/bunny.mp4 -fc RGB24 -w 720 -h 480 -o dump -n 100 --planes MERGED --skip_analyze
```
* Timeout for input frame reading can be set via --timeout option (time in seconds):
```
python simple.py -i tests/resources/bunny.mp4 -fc RGB24 -w 720 -h 480 -o dump -n 100 --planes MERGED --timeout 2
```
* Logs types and levels can be configured with -v, -vd and --nvtx options. Check help to find available values and description:
```
python simple.py -i tests/resources/bunny.mp4 -fc RGB24 -w 720 -h 480 -o dump -n 100 --planes MERGED -v HIGH -vd CONSOLE --nvtx
```
2. [Example](python_examples/many_consumers.py) demonstrates how to use TensorStream in case of several stream consumers:
```
python many_consumers.py -i tests/resources/bunny.mp4 -n 100
```
3. [Example](python_examples/different_streams.py) demonstrates how to use TensorStream if several streams should be handled simultaneously:
```
python different_streams.py -i1 <path-to-first-stream> -i2 <path-to-second-stream> -n1 100 -n2 50 -v1 LOW -v2 HIGH --cuda_device1 0 --cuda_device2 1
```
> **Warning:** Default path to second stream is relative, so need to run different_streams.py from parent folder if no arguments are passing
### PyTorch example

Real-time video style transfer example: [fast-neural-style](python_examples/fast_neural_style).

## Documentation

Documentation is in Doxygen, can be build manually.

## License

TensorStream is LGPL-2.1 licensed, see the [LICENSE](LICENSE) file for details.

### Used materials in samples

[Big Buck Bunny](https://peach.blender.org/)  is licensed under the  [Creative Commons Attribution 3.0 license](http://creativecommons.org/licenses/by/3.0/).
(c) copyright 2008, Blender Foundation /  [www.bigbuckbunny.org](http://www.bigbuckbunny.org/)