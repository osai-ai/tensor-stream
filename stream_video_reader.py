import torch
import VideoReader
import threading
import logging


class StatusLevel:
    OK = 0
    REPEAT = 1
    ERROR = 2


class LogsLevel:
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class LogsType:
    FILE = 1
    CONSOLE = 2


class CloseLevel:
    HARD = 1
    SOFT = 2


class StreamVideoReader:
    def __init__(self, stream_url, repeat_number=1):
        self.log = logging.getLogger()
        self.log.info("Create VideoStream")
        self.thread = None

        self.stream_url = stream_url
        self.repeat_number = repeat_number
        self.initialize()

    def initialize(self):
        self.log.info("Initialize VideoStream")
        status = StatusLevel.REPEAT
        repeat = self.repeat_number
        while status != StatusLevel.OK and repeat > 0:
            status = VideoReader.init(self.stream_url)
            if status != StatusLevel.OK:
                # Mode 1 - full close, mode 2 - soft close (for reset)
                self.stop(CloseLevel.SOFT)
            repeat = repeat - 1

        if repeat == 0:
            raise RuntimeError("Can't initialize VideoReader")

    def enable_logs(self, level, log_type):
        if log_type == LogsType.FILE:
            VideoReader.enableLogs(level)
        else:
            VideoReader.enableLogs(-level)

    def read(self, name: str, delay_index=0):
        return VideoReader.get(name, delay_index)

    def _start(self):
        VideoReader.start()

    def start(self):
        self.thread = threading.Thread(target=self._start)
        self.thread.start()

    def stop(self, level=CloseLevel.HARD):
        self.log.info("Stop VideoStream")
        VideoReader.close(level)
        if self.thread is not None:
            self.thread.join()

    def __del__(self):
        self.stop()


if __name__ == '__main__':
    import time

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


    url = "rtmp://b.sportlevel.com/relay/pooltop"
    video_reader = StreamVideoReader(url, repeat_number=20)
    video_reader.enable_logs(LogsLevel.LOW, LogsType.CONSOLE)

    video_reader.start()

    for i in range(100):
        tensor = video_reader.read("first")

    profiler = DeltaTimeProfiler()
    for i in range(1000):
        profiler.start()
        tensor = video_reader.read("first")
        profiler.end()
        time.sleep(0.016)  # Simulate consumer work

    print("Mean latancy:", profiler.mean)
    print("Tensor shape:", tensor.shape)

    video_reader.stop()
