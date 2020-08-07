import torch
from tensor_stream import TensorStreamConverter, StreamPool
from tensor_stream import LogsLevel, LogsType, FourCC, Planes, FrameRate, ResizeType
from threading import Thread
import argparse
import os

def string_bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def batch_frames(s):
    try:
        frames = list(map(int, s.split(',')))
        return frames
    except:
        raise argparse.ArgumentTypeError("Frames must be x1,x2,..")

def crop_coords(s):
    try:
        x1, y1, x2, y2 = map(int, s.split(','))
        return x1, y1, x2, y2
    except:
        raise argparse.ArgumentTypeError("Coordinates must be x1,y1,x2,y2")

def parse_arguments():
    parser = argparse.ArgumentParser(add_help=False,
                                     description="Simple usage example")
    parser.add_argument('--help', action='help')
    parser.add_argument("-i", "--input",
                        default="rtmp://37.228.119.44:1935/vod/big_buck_bunny.mp4",
                        help="Path to bitstream: RTMP, local file")
    parser.add_argument("-o", "--output",
                        help="Name of output raw stream", default="")
    parser.add_argument("-b", "--batch",
                        help="Name of output raw stream", type=batch_frames)
    parser.add_argument("-w", "--width",
                        help="Output width (default: input bitstream width)",
                        type=int, default=0)
    parser.add_argument("-h", "--height",
                        help="Output height (default: input bitstream height)",
                        type=int, default=0)
    parser.add_argument("-fc", "--fourcc", default="RGB24",
                        choices=["RGB24","BGR24", "Y800", "NV12", "UYVY", "YUV444", "HSV"],
                        help="Decoded stream' FourCC (default: RGB24)")
    parser.add_argument("-v", "--verbose", default="LOW",
                        choices=["LOW", "MEDIUM", "HIGH"],
                        help="Set output level from library (default: LOW)")
    parser.add_argument("-vd", "--verbose_destination", default="CONSOLE",
                        choices=["CONSOLE", "FILE"],
                        help="Set destination of logs (default: CONSOLE)")
    parser.add_argument("--normalize",
                        help="Set if output pixel values should be normalized. Option takes True or False arguments. \
                              If not set TensorStream will define value automatically",
                        type=string_bool)
    parser.add_argument("--nvtx",
                        help="Enable NVTX logs",
                        action='store_true')
    parser.add_argument("--cuda_device",
                        help="Set GPU for processing (default: 0)",
                        type=int, default=0)
    parser.add_argument("--planes", default="MERGED",
                        choices=["PLANAR", "MERGED"],
                        help="Possible planes order in RGB format")
    parser.add_argument("--resize_type", default="NEAREST",
                        choices=["NEAREST", "BILINEAR", "BICUBIC", "AREA"],
                        help="Algorithm used to do resize")
    parser.add_argument("--crop",
                        help="set crop, left top corner and right bottom corner (default: disabled)",
                        type=crop_coords, default=(0,0,0,0))
    parser.add_argument("--sw",
                        help="Use SW decoding (default: 0)",
                        type=int, default=0)

    return parser.parse_args()

def consumer(reader, args):
    parameters = {'pixel_format': FourCC[args.fourcc],
                  'width': args.width,
                  'height': args.height,
                  'crop_coords' : args.crop,
                  'normalization': args.normalize,
                  'planes_pos': Planes[args.planes],
                  'resize_type': ResizeType[args.resize_type]}

    result = reader.read_absolute(batch=args.batch, **parameters)
    if args.output:
        for i in range(0, result.shape[0]):
            reader.dump(result[i], args.output, **parameters)

    reader.reset("D:/Work/argus-tensor-stream/tests/resources/basler_004.mp4")

    result = reader.read_absolute(batch=args.batch, **parameters)
    if args.output:
        for i in range(0, result.shape[0]):
            reader.dump(result[i], args.output, **parameters)


if __name__ == '__main__':
    args = parse_arguments()

    stream_pool = StreamPool()
    stream_pool.cache_stream("D:/Work/argus-tensor-stream/tests/resources/tennis_2s.mp4")
    stream_pool.cache_stream("D:/Work/argus-tensor-stream/tests/resources/basler_004.mp4")
    #Note: max_consumers and buffer_size should be zero, otherwise for each instance additional memory will be allocated
    reader = TensorStreamConverter("D:/Work/argus-tensor-stream/tests/resources/tennis_2s.mp4", 0, args.cuda_device, 0, cuda=not args.sw)
    # To log initialize stage, logs should be defined before initialize call
    reader.enable_logs(LogsLevel[args.verbose], LogsType[args.verbose_destination])

    if args.nvtx:
        reader.enable_nvtx()

    reader.initialize(repeat_number=20)
    reader.enable_batch_optimization()

    consumer(reader, args)

    reader.stop()