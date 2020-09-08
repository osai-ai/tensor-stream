from tensor_stream import TensorStreamConverter
from tensor_stream import LogsLevel, LogsType, FourCC, Planes, FrameRate, ResizeType

import argparse
import os

def string_bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

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
    parser.add_argument("-n", "--number",
                        help="Number of frame to parse (default: unlimited)",
                        type=int, default=0)
    parser.add_argument("-bs", "--buffer_size",
                        help="Size of internal buffer stores processed frames (default: 5)",
                        type=int, default=5)
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
    parser.add_argument("--framerate_mode", default="NATIVE",
                        choices=["NATIVE", "FAST", "BLOCKING", "NATIVE_LOW_DELAY"],
                        help="Stream reading mode")
    parser.add_argument("--skip_analyze",
                        help="Skip bitstream frames reordering / loss analyze stage",
                        action='store_true')
    parser.add_argument("--timeout",
                        help="Set timeout in seconds for input frame reading (default: None, means disabled)",
                        type=float, default=None)
    parser.add_argument("--crop",
                        help="set crop, left top corner and right bottom corner (default: disabled)",
                        type=crop_coords, default=(0,0,0,0))

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    reader = TensorStreamConverter(args.input,
                                   max_consumers=5,
                                   cuda_device=args.cuda_device,
                                   buffer_size=args.buffer_size,
                                   framerate_mode=FrameRate[args.framerate_mode],
                                   timeout=args.timeout)
    # To log initialize stage, logs should be defined before initialize call
    reader.enable_logs(LogsLevel[args.verbose], LogsType[args.verbose_destination])

    if args.nvtx:
        reader.enable_nvtx()

    reader.initialize(repeat_number=20)

    if args.skip_analyze:
        reader.skip_analyze()

    reader.start()

    if args.output:
        if os.path.exists(args.output + ".yuv"):
            os.remove(args.output + ".yuv")

    print(f"Normalize {args.crop}")
    tensor = None
    try:
        while True:
            parameters = {'pixel_format': FourCC[args.fourcc],
                          'width': args.width,
                          'height': args.height,
                          'crop_coords' : args.crop,
                          'normalization': args.normalize,
                          'planes_pos': Planes[args.planes],
                          'resize_type': ResizeType[args.resize_type]}

            tensor, index = reader.read(**parameters, return_index=True)

            if args.number:
                if index > args.number:
                    break

            if args.output:
                reader.dump(tensor, args.output, **parameters)
    except RuntimeError as e:
        print(f"Bad things happened: {e}")
    finally:
        print("Frame size: ", reader.frame_size)
        print("FPS: ", reader.fps)
        if tensor is not None:
            print("Tensor shape:", tensor.shape)
            print("Tensor dtype:", tensor.dtype)
            print("Tensor device:", tensor.device)
        reader.stop()
