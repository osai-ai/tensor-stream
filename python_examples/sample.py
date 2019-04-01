import time
from tensor_stream import TensorStreamConverter, LogsLevel, LogsType, FourCC
import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--help', action='help',
                        help='show this help message and exit')
    parser.add_argument("-i", "--input",
                        default="rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4",
                        help="Path to bitstream: RTMP, local file")
    parser.add_argument("-o", "--output",
                        help="Name of output raw stream", default="")
    parser.add_argument("-w", "--width",
                        help="Output width (default: input bitstream width)",
                        type=int, default=0)
    parser.add_argument("-h", "--height",
                        help="Output height (default: input bitstream height)",
                        type=int, default=0)
    parser.add_argument("-fc", "--fourcc", default="RGB24",
                        choices=["RGB24","BGR24", "Y800"],
                        help="Decoded stream' FourCC (default: RGB24)")
    parser.add_argument("-v", "--verbose", default="LOW",
                        choices=["LOW", "MEDIUM", "HIGH"],
                        help="Set output level from library (default: LOW)")
    parser.add_argument("-n", "--number",
                        help="Number of frame to parse (default: unlimited)", type=int)
    return parser.parse_args()


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
    args = parse_arguments()

    reader = TensorStreamConverter(args.input, repeat_number=20)
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
    print("parameters: ", parameters)

    if args.output:
        if os.path.exists(args.output):
            os.remove(args.output)

    profiler = DeltaTimeProfiler()
    tensor = None
    try:
        while True:
            profiler.start()
            tensor = reader.read(**parameters)
            profiler.end()
            if args.output:
                reader.dump(tensor, args.output)
    except RuntimeError as e:
        print(f"Bad things happened: {e}")
    finally:
        print("Frame size: ", reader.frame_size)
        print("FPS: ", reader.fps)
        print("Mean latency:", profiler.mean)
        if tensor is not None:
            print("Tensor shape:", tensor.shape)
        reader.stop()
