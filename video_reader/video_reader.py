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

class FourCC:
    Y800 = 0
    RGB24 = 1
    BGR24 = 2
    NV12 = 3

class StreamVideoReader:
    def __init__(self, stream_url, repeat_number=1):
        self.log = logging.getLogger(__name__)
        self.log.info("Create VideoStream")
        self.thread = None
        self.fps = None
        self.frame_size = None

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
        else:
            params = VideoReader.getPars()
            self.fps = params['framerate_num'] / params['framerate_den']
            self.frame_size = (params['width'], params['height'])

    def enable_logs(self, level, log_type):
        if log_type == LogsType.FILE:
            VideoReader.enableLogs(level)
        else:
            VideoReader.enableLogs(-level)

    def read(self, **parameters):
        tensor, index = VideoReader.get(parameters)
        if int(parameters["return_index"]):
            return tensor, index
        else:
            return tensor

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
