import time
from video_reader import StreamVideoReader, LogsLevel, LogsType, FourCC

parser = argparse.ArgumentParser()
parser.add_argument("-url", help="Path to bitstream (RTMP, local file)")
parser.add_argument("-width", help="Bitstream width", type=int)
parser.add_argument("-height", help="Bitstream height", type=int)
parser.add_argument("-FourCC", choices=["RGB24","BGR24", "Y800"], help="Decoded stream' FourCC")
parser.add_argument("-v", choices=["LOW", "MEDIUM", "HIGH"],
                    help="Set output level from library")
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
    url = "rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4"
    reader = StreamVideoReader(url, repeat_number=20)
    reader.enable_logs(LogsLevel.MEDIUM, LogsType.CONSOLE)

    reader.start()
    parameters = {
        'name': "first",
        'delay': 0,
        'pixel_format': FourCC.RGB24,
        'return_index': False,
        'width': 720,
        'height': 480
    }
    try:
    # Warm up
        for i in range(100):
            tensor = reader.read(**parameters)

        profiler = DeltaTimeProfiler()        
        for i in range(100):
            profiler.start()
            tensor = reader.read(**parameters)
            profiler.end()
            #reader.dump(tensor, "dump")

        print("Frame size: ", reader.frame_size)
        print("FPS: ", reader.fps)
        print("Mean latancy:", profiler.mean)
        print("Tensor shape:", tensor.shape)

        reader.stop()
    except RuntimeError as e:
           print("Bad things happened: " + str(e) + "\n")
           reader.stop()