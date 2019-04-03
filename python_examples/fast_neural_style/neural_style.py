import re
import torch
import argparse
import numpy as np

from transfromer_net import TransformerNet
from ffmpeg_video_writer import FFmpegVideoWriter

from tensor_stream import TensorStreamConverter, FourCC


def parse_arguments():
    parser = argparse.ArgumentParser(add_help=False,
                                     description="Real-time style transfer on video")
    parser.add_argument('--help', action='help')
    parser.add_argument("-m", "--model",
                        default="saved_models/mosaic.pth",
                        help="Path to model weight")
    parser.add_argument("-i", "--input",
                        default="rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4",
                        help="Input stream (RTMP) or local video file")
    parser.add_argument("-o", "--output",
                         default="video.mp4",
                        help="Output stream or video file")
    parser.add_argument("-w", "--width",
                        help="Output width (default: input width)",
                        type=int, default=0)
    parser.add_argument("-h", "--height",
                        help="Output height (default: input height)",
                        type=int, default=0)
    parser.add_argument("-t", "--time",
                        help="Seconds to record, (default: unlimited)",
                        type=int, default=0)
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

    style_model = load_model(args.model)

    reader = TensorStreamConverter(args.input, repeat_number=20)
    reader.initialize()

    width = args.width if args.width else reader.frame_size[0]
    height = args.height if args.height else reader.frame_size[1]

    writer = FFmpegVideoWriter(args.output,
                               out_size=(width * 2, height),
                               out_fps=reader.fps,
                               bitrate=10000)

    reader.start()

    try:
        while True:
            tensor, index = reader.read("style",
                                        pixel_format=FourCC.RGB24,
                                        return_index=True,
                                        width=width,
                                        height=height)

            tensor = tensor.permute(2, 0, 1)
            tensor = tensor.unsqueeze(0)
            tensor = tensor.to(torch.float32)

            with torch.no_grad():
                output = style_model(tensor)
                output = torch.clamp(output, 0, 255)

            image = tensor_to_image(tensor)
            style_image = tensor_to_image(output)

            show_image = np.concatenate((image, style_image), axis=1)
            writer.write(show_image)

            if args.time:
                if index > args.time * reader.fps:
                    break

    except RuntimeError as e:
        print(f"Bad things happened: {e}")
    finally:
        reader.stop()
        writer.stop()
