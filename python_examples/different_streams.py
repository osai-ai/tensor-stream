import time
import argparse
from threading import Thread

from tensor_stream import TensorStreamConverter, FourCC, LogsLevel, LogsType


def parse_arguments():
    parser = argparse.ArgumentParser(add_help=False,
                                     description="Example with two consumers")
    parser.add_argument("-i1", "--input1",
                        default="rtmp://37.228.119.44:1935/vod/big_buck_bunny.mp4",
                        help="Path to bitstream: RTMP, local file")
    parser.add_argument("-i2", "--input2",
                        default="rtmp://37.228.119.44:1935/vod/big_buck_bunny.mp4",
                        help="Path to bitstream: RTMP, local file")
    parser.add_argument("-o1", "--output1",
                        help="Name of output raw stream", default="")
    parser.add_argument("-o2", "--output2",
                        help="Name of output raw stream", default="")
    parser.add_argument("-v1", "--verbose1", default="NONE",
                        choices=["LOW", "MEDIUM", "HIGH"],
                        help="Set output level from library (default: NONE)")
    parser.add_argument("-v2", "--verbose2", default="NONE",
                        choices=["LOW", "MEDIUM", "HIGH"],
                        help="Set output level from library (default: NONE)")
    parser.add_argument("-n1", "--number1",
                        help="Number of frame to parse (default: 50)",
                        type=int, default=50)
    parser.add_argument("-n2", "--number2",
                        help="Number of frame to parse (default: 50)",
                        type=int, default=50)
    parser.add_argument("--cuda_device1",
                        help="Set GPU for processing (default: 0)",
                        type=int, default=0)
    parser.add_argument("--cuda_device2",
                        help="Set GPU for processing (default: 0)",
                        type=int, default=0)
    return parser.parse_args()


def consumer1(reader, n_frames):
    for i in range(n_frames):
        parameters = {'pixel_format': FourCC.RGB24}
        tensor = reader.read(**parameters, name="consumer1")
        if args.output1:
            reader.dump(tensor, args.output1, **parameters)

    print()
    print("consumer1 shape:", tensor.shape)
    print("consumer1 dtype:", tensor.dtype, end='\n\n')
    reader.stop()


def consumer2(reader, n_frames):
    for i in range(n_frames):
        parameters = {'pixel_format': FourCC.BGR24,
                      'width': 720,
                      'height': 480}
        tensor, index = reader.read(**parameters,
                                    name="consumer2",
                                    return_index=True)
        if args.output2:
            reader.dump(tensor, args.output2, **parameters)

        if index % int(reader.fps) == 0:
            print("consumer2 frame index", index)

    reader.stop()
    time.sleep(1.0)  # prevent simultaneous print
    print("consumer2 shape:", tensor.shape)
    print("consumer2 dtype:", tensor.dtype)
    print("consumer2 last frame index:", index)


if __name__ == "__main__":
    args = parse_arguments()

    reader1 = TensorStreamConverter(args.input1,
                                    cuda_device=args.cuda_device1)
    reader1.enable_logs(LogsLevel[args.verbose1], LogsType.CONSOLE)
    reader1.initialize(repeat_number=20)

    reader2 = TensorStreamConverter(args.input2,
                                    cuda_device=args.cuda_device2)
    reader2.enable_logs(LogsLevel[args.verbose2], LogsType.CONSOLE)
    reader2.initialize(repeat_number=20)

    reader1.start()
    reader2.start()

    thread1 = Thread(target=consumer1, args=(reader1, args.number1))
    thread2 = Thread(target=consumer2, args=(reader2, args.number2))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()
