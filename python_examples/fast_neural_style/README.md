# Real-time video style transfer with TensorStream

Description + credentials + gif

## Requirements

### Software

* TensorStream and all dependencies
    * FFmpeg
    * Python 
    * NumPy 
    * PyTorch   

### Hardware

Example tested on RTMP stream (1080x608, 24 fps) on with the following hardware:
* i7 7700k
* Nvidia GTX 1060


## Usage

Download saved models:
```
python download_saved_models.py
```

Run example with default parameters:
```
python neural_style.py
```

Run example on with specific input, output video and different model. All arguments you can see with `python neural_style.py --help`.
```
python neural_style.py -m </path/to/saved-model.pth> -i </path/to/input-file-or-stream> -o </path/to/output-file-or-stream> -t <sec-to-record> 
```

You can stream result video to local address and open with VLC or something else.
```
python neural_style.py -i rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4 -o udp://127.0.0.1:1234 -m ./saved_models/candy.pth
```

For example open in mplayer:
```
mplayer -demuxer +mpegts -framedrop -benchmark udp://127.0.0.1:1234?buffer_size=1000000
```
