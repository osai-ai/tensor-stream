from tensor_stream import TensorStreamConverter, LogsLevel, LogsType, FourCC

import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--help', action='help',
                        help='show this help message and exit')
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
    parser.add_argument("-n", "--number",
                        help="Number of frame to parse (default: unlimited)",
                        type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    reader = TensorStreamConverter(args.input, repeat_number=20)
    reader.enable_logs(LogsLevel[args.verbose], LogsType.CONSOLE)
    reader.initialize()

    reader.start()
    parameters = {
        'name': "first",
        'delay': 0,
        'pixel_format': FourCC[args.fourcc],
        'return_index': True,
        'width': args.width,
        'height': args.height,
    }
    print("Read parameters: ", parameters)

    if args.output:
        if os.path.exists(args.output):
            os.remove(args.output)

    tensor = None
    try:
        while True:
            tensor, index = reader.read(**parameters)

            if args.number:
                if index > args.number:
                    break

            if args.output:
                reader.dump(tensor, args.output)
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
