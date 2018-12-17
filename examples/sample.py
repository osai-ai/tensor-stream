import time
from video_reader import StreamVideoReader, LogsLevel, LogsType, FourCC


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
    url = "rtmp://b.sportlevel.com/relay/pooltop"
    reader = StreamVideoReader(url, repeat_number=20)
    reader.enable_logs(LogsLevel.LOW, LogsType.CONSOLE)

    reader.start()
    parameters = {
        'name': "first",
        'delay': 0,
        'pixel_format': FourCC.RGB24,
        'return_index': False,
        'width': 720,
        'height': 420
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
            time.sleep(0.016)  # Simulate consumer work

    except RuntimeError:
           print("Bad things happened\n")

    print("Frame size: ", reader.frame_size)
    print("FPS: ", reader.fps)
    print("Mean latancy:", profiler.mean)
    print("Tensor shape:", tensor.shape)

    reader.stop()
