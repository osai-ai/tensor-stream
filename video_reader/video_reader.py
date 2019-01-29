import torch
import VideoReader
import threading
import logging
from enum import Enum

class StatusLevel(Enum):
    OK = 0
    REPEAT = 1
    ERROR = 2


class LogsLevel(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class LogsType(Enum):
    FILE = 1
    CONSOLE = 2


class CloseLevel(Enum):
    HARD = 1
    SOFT = 2


class FourCC(Enum):
    Y800 = 0
    RGB24 = 1
    BGR24 = 2


class StreamVideoReader:
    def __init__(self, stream_url, repeat_number=1):
        self.log = logging.getLogger(__name__)
        self.log.info("Create VideoStream")
        self.thread = None
        self.fps = None
        self.frame_size = None

        self.stream_url = stream_url
        self.repeat_number = repeat_number

    def initialize(self):
        self.log.info("Initialize VideoStream")
        status = StatusLevel.REPEAT.value
        repeat = self.repeat_number
        while status != StatusLevel.OK.value and repeat > 0:
            status = VideoReader.init(self.stream_url)
            if status != StatusLevel.OK.value:
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
            VideoReader.enableLogs(level.value)
        else:
            VideoReader.enableLogs(-level.value)

    def read(self,
             name: str,
             delay=0,
             pixel_format=FourCC.RGB24,
             return_index=False,
             width=0,
             height=0):
        tensor, index = VideoReader.get(name, delay, pixel_format.value, width, height)
        if return_index:
            return tensor, index
        else:
            return tensor

    def dump(self, tensor, name):
        VideoReader.dump(tensor, name)

    def _start(self):
        VideoReader.start()

    def start(self):
        self.thread = threading.Thread(target=self._start)
        self.thread.start()

    def stop(self, level=CloseLevel.HARD):
        self.log.info("Stop VideoStream")
        VideoReader.close(level.value)
        if self.thread is not None:
            self.thread.join()

    def __del__(self):
        self.stop()
