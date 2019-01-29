import time
from video_reader import StreamVideoReader, LogsLevel, LogsType, FourCC
import argparse
import os

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--help', action='help', help='show this help message and exit')
parser.add_argument("-i", "--input", default="rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4", help="Path to bitstream: RTMP, local file")
parser.add_argument("-o", "--output", default="dump.yuv", help="Name of output raw stream (default: dump.yuv)")
parser.add_argument("-w", "--width", default=720, help="Output width (default: 720)", type=int)
parser.add_argument("-h", "--height", default=480, help="Output height (default: 480)", type=int)
parser.add_argument("-fc", "--fourcc", default="RGB24", choices=["RGB24","BGR24", "Y800"], help="Decoded stream' FourCC (default: RGB24)")
parser.add_argument("-v", "--verbose", default="LOW", choices=["LOW", "MEDIUM", "HIGH"], help="Set output level from library (default: LOW)")
parser.add_argument("-n", "--number", default=100, help="Number of frame to parse (default: 100)", type=int)
args = parser.parse_args()

class DeltaTimeProfiler:
    def __init__(self):
        self.mean = 0.0
        self.count = 0
        self.prev_time = time.time()

    def start(self):
        self.prev_time = time.time()

    def end(self):
        self.count += 1
        now_time = time.time()
        delta = now_time - self.prev_time
        self.mean += (delta - self.mean) / self.count
        self.prev_time = now_time

    def mean_delta(self):
        return self.mean

    def reset(self):
        self.mean = 0.0
        self.count = 0


if __name__ == '__main__':
    reader = StreamVideoReader(args.input, repeat_number=20)
    reader.enable_logs(LogsLevel[args.verbose], LogsType.CONSOLE)
    reader.initialize()

    reader.start()
    parameters = {
        'name': "first",
        'delay': 0,
        'pixel_format': FourCC[args.fourcc],
        'return_index': False,
        'width': args.width,
        'height': args.height,
    }

    if os.path.exists(args.output):
        os.remove(args.output)

    try:
        profiler = DeltaTimeProfiler()        
        for i in range(args.number):
            profiler.start()
            tensor = reader.read(**parameters)
            profiler.end()
            reader.dump(tensor, args.output)

        print("Frame size: ", reader.frame_size)
        print("FPS: ", reader.fps)
        print("Mean latancy:", profiler.mean)
        print("Tensor shape:", tensor.shape)

        reader.stop()
    except RuntimeError as e:
           print("Bad things happened: " + str(e) + "\n")
           reader.stop()