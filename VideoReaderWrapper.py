import torch
import VideoReader
import threading


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


def EnableLogs(level, log_type):
    if log_type == LogsType.FILE:
        VideoReader.enableLogs(level)
    else:
        VideoReader.enableLogs(-level)


def Close(level):
    VideoReader.close(level)


def Initialize(url, repeat_number=20):
    status = StatusLevel.REPEAT
    repeat = repeat_number
    while status != StatusLevel.OK and repeat > 0:
        status = VideoReader.init(url)
        if status != StatusLevel.OK:
            # Mode 1 - full close, mode 2 - soft close (for reset)
            Close(CloseLevel.SOFT)
        repeat = repeat - 1

    if repeat == 0:
        raise RuntimeError("Can't initialize VideoReader")


def start():
    VideoReader.start()


def StartProcessing():
    pipeline = threading.Thread(target=start)
    pipeline.start()
    return pipeline


def GetFrame(consumer_name, frame_index=0):
    return VideoReader.get(consumer_name, frame_index)
