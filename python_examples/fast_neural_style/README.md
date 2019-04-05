
# Real-time video style transfer with TensorStream

This example demonstrates how to use TensorStream with [fast-neural-style](https://github.com/pytorch/examples/tree/master/fast_neural_style) model.
Input frames taken from RTMP stream are converted to Tensor and passed to trained model so augmented video is produced. The whole pipeline works in realtime.

<p align="center">
  <img src="/python_examples/fast_neural_style/example.gif" width="1080" title="Real-time video style transfer with TensorStream">
</p>


## Requirements

* TensorStream and next external dependencies:
    * [FFmpeg](https://github.com/FFmpeg/FFmpeg)
    * [Python](https://www.python.org/) 3.6 or above
    * Latest stable version of [PyTorch](https://github.com/pytorch/pytorch) and NumPy

## Usage

 - Download saved models:
```
python download_saved_models.py
```
 - Run example with default parameters:
```
python neural_style.py -w 808 -h 456 
```
>**Note:** on Windows you have to add path to ffmpeg executable file to PATH
 - Run example with specific input/output video, different model weights and model input resolution:
>**Note:** You can pass **--help** to get list of all available options, their description and default values:

```
python neural_style.py -m </path/to/saved-model.pth> -i </path/to/input-file-or-stream> -o </path/to/output-file-or-stream> -t <sec-to-record> -h <height> -w <width>
```

 - You can stream result video to local address and open in VLC or something else:
```
python neural_style.py -i rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4 -o udp://127.0.0.1:1234 -m ./saved_models/candy.pth -w 808 -h 456
```
for example open in MPlayer:
```
mplayer -demuxer +mpegts -framedrop -benchmark udp://127.0.0.1:1234?buffer_size=1000000
```
##
Example tested at 24 fps on:
* RTMP stream - 1080x608, 24 fps 
* Model input resolution - 808x456 
* CPU - i7 7700k
* GPU - Nvidia GTX 1080ti
