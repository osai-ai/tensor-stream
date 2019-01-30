import torch
import VideoReader
import threading
import logging
from enum import Enum

## Class with list of possible error statuses can be returned from VideoReader extension
# @warning These statuses are used only in Python wrapper that communicates with VideoReader C++ extension 
class StatusLevel(Enum):
    ## No errors
    OK = 0
    ## Need to call VideoReader API one more time
    REPEAT = 1
    ## Some issue in VideoReader component occured
    ERROR = 2

## Class with list of modes for logs output
# @details Used in StreamVideoReader.enable_logs() function
class LogsLevel(Enum):
    ## No logs are needed
    NONE = 0
    ## Print the indexes of processed frames
    LOW = 1
    ## Print also frame processing duration
    MEDIUM = 2
    ## Print also the detailed information about functions in callstack
    HIGH = 3

## Class with list of places the log file has to be written to
# @details Used in StreamVideoReader.enable_logs() function
class LogsType(Enum):
    ## Print all logs to file
    FILE = 1
    ## Print all logs to console
    CONSOLE = 2


## Class with possible C++ extension module close options
# @details Used in StreamVideoReader.stop() function
class CloseLevel(Enum):
    ## Close all opened handlers, free resources
    HARD = 1
    ## Close all opened handlers except logs file handler, free resources
    SOFT = 2

## Class with supported frame output color formats
# @details Used in StreamVideoReader.read() function
class FourCC(Enum):
    ## Monochrome format, 8 bit for pixel
    Y800 = 0
    ## RGB format, 24 bit for pixel, color plane order: R, G, B
    RGB24 = 1
    ## RGB format, 24 bit for pixel, color plane order: B, G, R
    BGR24 = 2


## Class which allow start decoding process and get Pytorch tensors with post-processed frame data
class StreamVideoReader:
    ## Constructor of StreamVideoReader class
    # @param[in] stream_url Path to stream should be decoded
    # @param[in] repeat_number Set how many times initialize() function will try to initialize pipeline in case of any issues
    def __init__(self, stream_url, repeat_number=1):
        self.log = logging.getLogger(__name__)
        self.log.info("Create VideoStream")
        self.thread = None
        self.fps = None
        self.frame_size = None

        self.stream_url = stream_url
        self.repeat_number = repeat_number

    ## Initialization of C++ extension 
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

    ## Enable logs from VideoReader C++ extension
    # @param[in] level Specify output level of logs, see LogsLevel for supported values
    # @param[in] log_type Specify where the logs should be printed, see LogsType for supported values
    def enable_logs(self, level, log_type):
        if log_type == LogsType.FILE:
            VideoReader.enableLogs(level.value)
        else:
            VideoReader.enableLogs(-level.value)

    ## Read the next decoded frame, should be invoked only after start() call
    # @param[in] name The unique ID of consumer. Needed mostly in case of several consumers work in different threads
    # @param[in] delay Specify which frame shoould be read from decoded buffer. Can take values in range [-10, 0]
    # @param[in] pixel_format Output FourCC of frame stored in tensor, see FourCC for supported values
    # @param[in] return_index Specify whether need return index of decoded frame or not
    # @param[in] width Specify the width of decoded frame
    # @param[in] height Specify the height of decoded frame
    # @return Decoded frame wrapped to Pytorch tensor and index of decoded frame if return_index option set
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

    ## Dump the tensor to hard driver
    # @param[in] tensor Tensor which should be dumped
    # @param[in] name The name of file with dumps
    def dump(self, tensor, name):
        VideoReader.dump(tensor, name)

    def _start(self):
        VideoReader.start()

    ## Start decoding in separate thread
    def start(self):
        self.thread = threading.Thread(target=self._start)
        self.thread.start()

    ## Close decoding session
    def stop(self, level=CloseLevel.HARD):
        self.log.info("Stop VideoStream")
        VideoReader.close(level.value)
        if self.thread is not None:
            self.thread.join()

    def __del__(self):
        self.stop()

    ## @var fps Amount of frames per second obtained from input bitstream, set by initialize() function
    ## @var frame_size Size (width and height) of frames in input bitstream, set by initialize() function

