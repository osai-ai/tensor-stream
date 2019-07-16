import time
import argparse
from threading import Thread

from tensor_stream import TensorStreamConverter, FourCC


def parse_arguments():
    parser = argparse.ArgumentParser(add_help=False,
                                     description="Example with two consumers")
    parser.add_argument("-i", "--input",
                        default="rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4",
                        help="Path to bitstream: RTMP, local file")
    parser.add_argument("-n", "--number",
                        help="Number of frame to parse (default: 100)",
                        type=int, default=100)
    return parser.parse_args()


def consumer1(reader, n_frames):
    try:
        for i in range(n_frames):
            tensor = reader.read(name="consumer1",
                                pixel_format=FourCC.RGB24,
                                width=540,
                                height=304)

        print()
        print("consumer1 shape:", tensor.shape)
        print("consumer1 dtype:", tensor.dtype, end='\n\n')

    except RuntimeError as e:
        print(f"Bad things happened: {e}")

def consumer2(reader, n_frames):
    try:
        for i in range(n_frames):
            tensor, index = reader.read(name="consumer2",
                                        pixel_format=FourCC.BGR24,
                                        return_index=True)

            if index % int(reader.fps) == 0:
                print("consumer2 frame index", index)
  
        time.sleep(1.0)  # prevent simultaneous print
        print("consumer2 shape:", tensor.shape)
        print("consumer2 dtype:", tensor.dtype)
        print("consumer2 last frame index:", index)

    except RuntimeError as e:
        print(f"Bad things happened: {e}")
 
if __name__ == "__main__":
    args = parse_arguments()

    reader = TensorStreamConverter(args.input, repeat_number=20)
    reader.initialize()

    reader.start()

    thread1 = Thread(target=consumer1, args=(reader, args.number))
    thread2 = Thread(target=consumer2, args=(reader, args.number))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    reader.stop()
