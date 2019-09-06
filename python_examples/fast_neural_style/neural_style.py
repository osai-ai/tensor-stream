import re
import torch
import argparse
import numpy as np

from transfromer_net import TransformerNet
from ffmpeg_video_writer import FFmpegVideoWriter

from tensor_stream import TensorStreamConverter, Planes, FourCC


def parse_arguments():
    parser = argparse.ArgumentParser(add_help=False,
                                     description="Real-time video style transfer example")
    parser.add_argument('--help', action='help')
    parser.add_argument("-m", "--model",
                        default="saved_models/mosaic.pth",
                        help="Path to model weight")
    parser.add_argument("-i", "--input",
                        default="rtmp://37.228.119.44:1935/vod/big_buck_bunny.mp4",
                        help="Input stream (RTMP) or local video file")
    parser.add_argument("-o", "--output",
                        default="video.mp4",
                        help="Output stream or video file")
    parser.add_argument("--concat_orig", action='store_true',
                        help="Concatenate original frames to output video")
    parser.add_argument("-w", "--width",
                        help="Output width (default: input width)",
                        type=int, default=0)
    parser.add_argument("-h", "--height",
                        help="Output height (default: input height)",
                        type=int, default=0)
    parser.add_argument("-t", "--time",
                        help="Seconds to record, (default: unlimited)",
                        type=int, default=0)
    parser.add_argument('-c', '--codec', default="h264_nvenc",
                        help='Encoder codec for output video', type=str)
    parser.add_argument('-p', '--preset', default="slow",
                        help='Preset for output video', type=str)
    parser.add_argument('-b', '--bitrate', default=5000,
                        help='Bitrate (kb/s) for output video', type=int)
    return parser.parse_args()


def load_model(model_path, device='cuda'):
    model = TransformerNet()
    state_dict = torch.load(model_path)

    # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def tensor_to_image(tensor):
    image = tensor[0].to(torch.uint8)
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy()
    return image


if __name__ == "__main__":
    args = parse_arguments()

    style_model = load_model(args.model, device='cuda')

    reader = TensorStreamConverter(args.input, repeat_number=20)
    reader.initialize()
    print(f"Input video frame size: {reader.frame_size}, fps: {reader.fps}")

    width = args.width if args.width else reader.frame_size[0]
    height = args.height if args.height else reader.frame_size[1]
    print(f"Model input image width: {width}, height: {height}")

    writer = FFmpegVideoWriter(args.output,
                               out_size=(width * 2 if args.concat_orig else width,
                                         height),
                               out_fps=reader.fps,
                               bitrate=args.bitrate,
                               codec=args.codec,
                               preset=args.preset)

    reader.start()

    try:
        while True:
            tensor, index = reader.read(pixel_format=FourCC.RGB24,
                                        return_index=True,
                                        width=width,
                                        height=height,
                                        planes_pos=Planes.PLANAR,
                                        normalization=True)
            tensor = tensor.unsqueeze(0)
            with torch.no_grad():
                output = style_model(tensor)
                output = torch.clamp(output, 0, 255)

            style_frame = tensor_to_image(output)
            if args.concat_orig:
                orig_frame = tensor_to_image(tensor)
                write_frame = np.concatenate((orig_frame, style_frame), axis=1)
            else:
                write_frame = style_frame
            writer.write(write_frame)

            if args.time:
                if index > args.time * reader.fps:
                    break

    except RuntimeError as e:
        print(f"Bad things happened: {e}")
    finally:
        reader.stop()
        writer.stop()
