import time
from video_reader import StreamVideoReader, LogsLevel, LogsType


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

    # Warm up
    for i in range(100):
        tensor = reader.read("first")

    profiler = DeltaTimeProfiler()
    for i in range(1000):
        profiler.start()
        tensor = reader.read("first")
        profiler.end()
        time.sleep(0.016)  # Simulate consumer work

    print("Frame size: ", reader.frame_size)
    print("FPS: ", reader.fps)
    print("Mean latancy:", profiler.mean)
    print("Tensor shape:", tensor.shape)

    reader.stop()
