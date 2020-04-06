import torch
import TensorStream
import threading
import logging
from enum import Enum


## @defgroup pythonAPI Python API
# @brief The list of TensorStream components can be used via Python interface
# @details Here are all the classes, enums, functions described which can be used via Python to do RTMP/local stream converting to Pytorch Tensor with additional post-processing conversions
# @{

## Class with list of possible error statuses can be returned from TensorStream extension
# @warning These statuses are used only in Python wrapper that communicates with TensorStream C++ extension
class StatusLevel(Enum):
    ## No errors
    OK = 0
    ## Need to call %TensorStream API one more time
    REPEAT = 1
    ## Some issue in %TensorStream component occured
    ERROR = 2


## Class with list of modes for logs output
# @details Used in @ref TensorStreamConverter.enable_logs() function
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
# @details Used in @ref TensorStreamConverter.enable_logs() function
class LogsType(Enum):
    ## Print all logs to file
    FILE = 1
    ## Print all logs to console
    CONSOLE = 2


## Class with supported frame output color formats
# @details Used in @ref TensorStreamConverter.read() function
class FourCC(Enum):
    ## Monochrome format, 8 bit for pixel
    Y800 = 0
    ## RGB format, 24 bit for pixel, color plane order: R, G, B
    RGB24 = 1
    ## RGB format, 24 bit for pixel, color plane order: B, G, R
    BGR24 = 2
    ## YUV semi-planar format, 12 bit for pixel
    NV12 = 3
    ## YUV merged format, 16 bit for pixel
    UYVY = 4
    ## YUV merged format, 24 bit for pixel
    YUV444 = 5
    ## HSV format, 24 bit for pixel
    HSV = 6


## Algorithm used to do resize
# @details Resize algorithms are applied to NV12 so b2b with another frameworks isn't guaranteed
class ResizeType(Enum):
    ## Simple algorithm without any interpolation
    NEAREST = 0
    ## Algorithm that does simple linear interpolation
    BILINEAR = 1
    ## Algorithm that does spline bicubic interpolation
    BICUBIC = 2
    ## Algorithm that does INTER_AREA OpenCV interpolation
    AREA = 3


## Possible planes order in RGB format
class Planes(Enum):
    ## Color components R, G, B are stored in memory separately like RRRRR, GGGGG, BBBBB
    PLANAR = 0
    ## Color components R, G, B are stored in memory one by one like RGBRGBRGB
    MERGED = 1


## Enum with possible stream reading modes
class FrameRate(Enum):
    ## Read at native stream frame rate
    NATIVE = 0
    ## Read at fixed stream frame rate
    NATIVE_SIMPLE = 1
    ## Read frames as fast as possible
    FAST = 2
    ## Read frame by frame without skipping (only local files)
    BLOCKING = 3


## Class that stores frame parameters
class FrameParameters:
    ## Constructor of FrameParameters class
    # @param[in] width Specify the width of decoded frame
    # @param[in] height Specify the height of decoded frame
    # @param[in] crop_coords Left top and right bottom coordinates of crop
    # @param[in] resize_type Algorithm used to do resize, see @ref ResizeType for supported values
    # @param[in] pixel_format Output FourCC of frame stored in tensor, see @ref FourCC for supported values
    # @param[in] planes_pos Possible planes order in RGB format, see @ref Planes for supported values
    # @param[in] normalization Should final colors be normalized or not
    def __init__(self,
                 width=0,
                 height=0,
                 crop_coords=(0, 0, 0, 0),
                 resize_type=ResizeType.NEAREST,
                 pixel_format=FourCC.RGB24,
                 planes_pos=Planes.MERGED,
                 normalization=None):
        parameters = TensorStream.FrameParameters()
        color_options = TensorStream.ColorOptions(TensorStream.FourCC(pixel_format.value))
        if normalization is not None:
            color_options.normalization = normalization
        color_options.planesPos = TensorStream.Planes(planes_pos.value)

        resize_options = TensorStream.ResizeOptions()
        resize_options.width = width
        resize_options.height = height
        resize_options.resizeType = TensorStream.ResizeType(resize_type.value)

        crop_options = TensorStream.CropOptions()
        crop_options.leftTopCorner = crop_coords[0:2]
        crop_options.rightBottomCorner = crop_coords[2:4]

        parameters.color = color_options
        parameters.resize = resize_options
        parameters.crop = crop_options
        self.parameters = parameters

    def __repr__(self):
        string = (f"FrameParameters(\n"
                  f"    width={self.parameters.resize.height},\n"
                  f"    height={self.parameters.resize.width},\n"
                  f"    crop_left_top={self.parameters.crop.leftTopCorner},\n"
                  f"    crop_right_bottom={self.parameters.crop.rightBottomCorner},\n"
                  f"    resize_type={self.parameters.resize.resizeType},\n"
                  f"    pixel_format={self.parameters.color.dstFourCC},\n"
                  f"    planes_pos={self.parameters.color.planesPos},\n"
                  f"    normalization={self.parameters.color.normalization}\n"
                  ")")
        return string


## Class which allow start decoding process and get Pytorch tensors with post-processed frame data
class TensorStreamConverter:
    ## Constructor of TensorStreamConverter class
    # @param[in] stream_url Path to stream should be decoded
    # @param[in] max_consumers Allowed number of simultaneously working consumers
    # @param[in] cuda_device GPU used for execution
    # @param[in] buffer_size Set how many processed frames can be stored in internal buffer
    # @warning Size of buffer should be less or equal to DPB
    # @param[in] timeout How many seconds to wait for the new frame
    def __init__(self,
                 stream_url,
                 max_consumers=5,
                 cuda_device=torch.cuda.current_device(),
                 buffer_size=5,
                 framerate_mode=FrameRate.NATIVE,
                 timeout=None):
        self.log = logging.getLogger(__name__)
        self.log.info("Create TensorStream")
        self.tensor_stream = TensorStream.TensorStream()
        self.thread = None
        ## Amount of frames per second obtained from input bitstream, set by @ref initialize() function
        self.fps = None
        ## Size (width and height) of frames in input bitstream, set by @ref initialize() function
        self.frame_size = None

        self.max_consumers = max_consumers
        self.cuda_device = cuda_device
        self.buffer_size = buffer_size
        self.stream_url = stream_url
        self.framerate_mode = TensorStream.FrameRateMode(framerate_mode.value)
        self.set_timeout(timeout=timeout)

    ## Initialization of C++ extension
    # @param[in] repeat_number Set how many times try to initialize pipeline in case of any issues
    # @warning if initialization attempts exceeded @ref repeat_number, RuntimeError is being thrown
    def initialize(self, repeat_number=1):
        self.log.info("Initialize TensorStream")
        status = StatusLevel.REPEAT.value
        repeat = repeat_number
        while status != StatusLevel.OK.value and repeat > 0:
            status = self.tensor_stream.init(self.stream_url,
                                             self.max_consumers,
                                             self.cuda_device,
                                             self.buffer_size,
                                             self.framerate_mode)
            if status != StatusLevel.OK.value:
                self.stop()
                repeat = repeat - 1

        if repeat == 0:
            raise RuntimeError("Can't initialize TensorStream")
        else:
            params = self.tensor_stream.getPars()
            self.fps = params['framerate_num'] / params['framerate_den']
            self.frame_size = (params['width'], params['height'])

    ## Enable logs from TensorStream C++ extension
    # @param[in] level Specify output level of logs, see @ref LogsLevel for supported values
    # @param[in] log_type Specify where the logs should be printed, see @ref LogsType for supported values
    def enable_logs(self, level, log_type):
        if level != LogsLevel.NONE:
            if log_type == LogsType.FILE:
                self.tensor_stream.enableLogs(level.value)
            else:
                self.tensor_stream.enableLogs(-level.value)

    ## Enable NVTX from TensorStream C++ extension
    def enable_nvtx(self):
        self.tensor_stream.enableNVTX()

    ## Pass timeout for reading input frame
    # @param[in] timeout How many seconds to wait for the new frame
    def set_timeout(self, timeout):
        if timeout is None:
            self.tensor_stream.setTimeout(-1)
        else:
            ms_timeout = int(timeout * 1000)
            self.tensor_stream.setTimeout(ms_timeout)

    ## Skip bitstream frames reordering / loss analyze stage
    def skip_analyze(self):
        self.tensor_stream.skipAnalyze()

    def read_absolute(self,
             batch,
             name="default",
             width=0,
             height=0,
             resize_type=ResizeType.NEAREST,
             crop_coords=(0,0,0,0),
             pixel_format=FourCC.RGB24,
             planes_pos=Planes.MERGED,
             normalization=None):

        frame_parameters = FrameParameters(
            width=width,
            height=height,
            crop_coords=crop_coords,
            resize_type=resize_type,
            pixel_format=pixel_format,
            planes_pos=planes_pos,
            normalization=normalization
        )
        result = self.param_read_absolute(batch=batch,
                                          frame_parameters=frame_parameters,
                                          name=name)
        return result

    def param_read_absolute(self,
                            batch,
                            frame_parameters: FrameParameters,
                            name="default"):

        tensor = self.tensor_stream.getAbsolute(name, batch, frame_parameters.parameters)
        return tensor


    ## Read the next decoded frame, should be invoked only after @ref start() call
    # @param[in] name The unique ID of consumer. Needed mostly in case of several consumers work in different threads
    # @param[in] width Specify the width of decoded frame
    # @param[in] height Specify the height of decoded frame
    # @param[in] crop_coords Left top and right bottom coordinates of crop
    # @param[in] resize_type Algorithm used to do resize, see @ref ResizeType for supported values
    # @param[in] pixel_format Output FourCC of frame stored in tensor, see @ref FourCC for supported values
    # @param[in] planes_pos Possible planes order in RGB format, see @ref Planes for supported values
    # @param[in] normalization Should final colors be normalized or not
    # @param[in] delay Specify which frame should be read from decoded buffer. Can take values in range [-buffer_size, 0]
    # @param[in] return_index Specify whether need return index of decoded frame or not

    # @return Decoded frame in CUDA memory wrapped to Pytorch tensor and index of decoded frame if @ref return_index option set
    def read(self,
             name="default",
             width=0,
             height=0,
             resize_type=ResizeType.NEAREST,
             crop_coords=(0,0,0,0),
             pixel_format=FourCC.RGB24,
             planes_pos=Planes.MERGED,
             normalization=None,
             delay=0,
             return_index=False):

        frame_parameters = FrameParameters(
            width=width,
            height=height,
            crop_coords=crop_coords,
            resize_type=resize_type,
            pixel_format=pixel_format,
            planes_pos=planes_pos,
            normalization=normalization
        )
        result = self.param_read(frame_parameters,
                                 name=name,
                                 delay=delay,
                                 return_index=return_index)
        return result

    ## Read the next decoded frame, should be invoked only after @ref start() call
    # @param[in] name The unique ID of consumer. Needed mostly in case of several consumers work in different threads
    # @param[in] frame_parameters Frame parameters
    # @param[in] delay Specify which frame should be read from decoded buffer. Can take values in range [-buffer_size, 0]
    # @param[in] return_index Specify whether need return index of decoded frame or not

    # @return Decoded frame in CUDA memory wrapped to Pytorch tensor and index of decoded frame if @ref return_index option set
    def param_read(self,
                   frame_parameters: FrameParameters,
                   name="default",
                   delay=0,
                   return_index=False):
        tensor, index = self.tensor_stream.get(name, delay, frame_parameters.parameters)
        if return_index:
            return tensor, index
        else:
            return tensor

    ## Dump the tensor to hard driver
    # @param[in] tensor Tensor which should be dumped
    # @param[in] name The name of file with dumps
    # @param[in] width Specify the width of decoded frame
    # @param[in] height Specify the height of decoded frame
    # @param[in] crop_coords Left top and right bottom coordinates of crop
    # @param[in] resize_type Algorithm used to do resize, see @ref ResizeType for supported values
    # @param[in] pixel_format Output FourCC of frame stored in tensor, see @ref FourCC for supported values
    # @param[in] planes_pos Possible planes order in RGB format, see @ref Planes for supported values
    # @param[in] normalization Should final colors be normalized or not
    def dump(self,
             tensor,
             name="default",
             width=0,
             height=0,
             crop_coords=(0,0,0,0),
             resize_type=ResizeType.NEAREST,
             pixel_format=FourCC.RGB24,
             planes_pos=Planes.MERGED,
             normalization=None):
        frame_parameters = FrameParameters(
            width=width,
            height=height,
            crop_coords=crop_coords,
            resize_type=resize_type,
            pixel_format=pixel_format,
            planes_pos=planes_pos,
            normalization=normalization
        )
        self.tensor_stream.dump(tensor, name, frame_parameters.parameters)

    def _start(self):
        self.tensor_stream.start()

    ## Start processing with parameters set via @ref initialize() function
    # This functions is being executed in separate thread
    def start(self):
        self.thread = threading.Thread(target=self._start)
        self.thread.start()

    ## Close TensorStream session
    # @param[in] level Value from @ref CloseLevel
    def stop(self):
        self.log.info("Stop TensorStream")
        self.tensor_stream.close()
        if self.thread is not None:
            self.thread.join()

## @}