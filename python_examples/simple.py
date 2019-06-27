from tensor_stream import TensorStreamConverter, LogsLevel, LogsType, FourCC, Planes, ResizeType

import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(add_help=False,
                                     description="Simple usage example")
    parser.add_argument('--help', action='help')
    parser.add_argument("-i", "--input",
                        default="rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4",
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
                        choices=["RGB24","BGR24", "Y800"],
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
                        help="Size of internal buffer stores processed frames (default: 10)",
                        type=int, default=10)
    parser.add_argument("--normalize",
                        help="Set if output pixel values should be normalized",
                        action='store_true')
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
                        choices=["NEAREST", "BILINEAR"],
                        help="Algorithm used to do resize")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    reader = TensorStreamConverter(args.input, max_consumers = 5, cuda_device = args.cuda_device, buffer_size = args.buffer_size, repeat_number=20)
    reader.enable_logs(LogsLevel[args.verbose], LogsType[args.verbose_destination])
    if (args.nvtx):
        reader.enable_nvtx()
    
    reader.initialize()

    reader.start()

    if args.output:
        if os.path.exists(args.output):
            os.remove(args.output)

    tensor = None
    try:
        while True:
            parameters = {'pixel_format' : FourCC[args.fourcc],
                          'width' : args.width,
                          'height' : args.height,
                          'normalization' : args.normalize,
                          'planes_pos' : Planes[args.planes],
                          'resize_type' : ResizeType[args.resize_type]}

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
