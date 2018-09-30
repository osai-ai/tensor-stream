import torch
import VideoReader
from enum import IntEnum
import os
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

def EnableLogs(level, type):
	if (type == LogsType.FILE):
		VideoReader.enableLogs(level)
	else:
		VideoReader.enableLogs(-level)

def Close(level):
	VideoReader.close(level)

def Initialize(fileName, repeatNumber = 20):
	status = StatusLevel.REPEAT
	repeat = repeatNumber
	while (status != StatusLevel.OK and repeat > 0):
		status = VideoReader.init(fileName)
		if (status != StatusLevel.OK):
			#Mode 1 - full close, mode 2 - soft close (for reset)
			Close(CloseLevel.SOFT)
		repeat = repeat - 1

	if (repeat == 0):
		return StatusLevel.ERROR

def start():
	VideoReader.start()

def StartProcessing():
	pipeline = threading.Thread(target=start)
	pipeline.start()
	return pipeline

def GetFrame(consumerID, frameIndex):
	return VideoReader.get(consumerID, frameIndex)