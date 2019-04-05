# Real-time video style transfer with TensorStream

Description + credentials + gif

## Requirements

* TensorStream and all dependencies
    * FFmpeg
    * Python 
    * NumPy 
    * PyTorch   

## Usage

Download saved models:
```
python download_saved_models.py
```

Run example with default parameters:
```
python neural_style.py -w 808 -h 456 
```

Run example with specific input, output video, different model weights and model input resolution. All arguments you can see with `python neural_style.py --help`.
```
python neural_style.py -m </path/to/saved-model.pth> -i </path/to/input-file-or-stream> -o </path/to/output-file-or-stream> -t <sec-to-record> -h <height> -w <width>
```

You can stream result video to local address and open in VLC or something else.
```
python neural_style.py -i rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4 -o udp://127.0.0.1:1234 -m ./saved_models/candy.pth -w 808 -h 456
```

For example open in mplayer:
```
mplayer -demuxer +mpegts -framedrop -benchmark udp://127.0.0.1:1234?buffer_size=1000000
```

Example tested at 24 fps on:
* RTMP stream - 1080x608, 24 fps 
* Model input resolution - 808x456 
* CPU - i7 7700k
* GPU - Nvidia GTX 1080ti
