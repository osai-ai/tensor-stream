import logging
import subprocess as sp


class FFmpegVideoWriter:
    def __init__(self,
                 url,
                 out_size,
                 out_fps=30,
                 bitrate=2000,
                 codec="h264_nvenc",
                 preset="medium",
                 loglevel="warning",
                 keyframe_freq=30):
        self.logger = logging.getLogger(__name__)

        self.url = url
        self.out_fps = out_fps
        self.out_size = out_size
        self.bitrate = bitrate
        self.codec = codec
        self.preset = preset
        self.loglevel = loglevel
        self.keyframe_freq = keyframe_freq

        self.in_size = None
        self.pipe = None
        self.stopped = True
        self.logger.info("Create VideoWriter")

    def _start(self):
        command = [
            'ffmpeg',
            '-y',
            '-loglevel', self.loglevel,
            # input
            '-r:v', str(self.out_fps),
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s:v', '%dx%d'%tuple(self.in_size), '-pix_fmt', 'rgb24',
            '-i', '-', '-an',
            # output
            '-vcodec', self.codec,
            '-s:v', '%dx%d'%tuple(self.out_size),
            '-r:v', str(self.out_fps),
            '-b:v', '%dk' % self.bitrate,
            '-preset', self.preset,
            '-g', str(int(self.keyframe_freq * self.out_fps)),
            '-f', 'mpegts', self.url
        ]
        self.logger.info(f"Start VideoWriter\nFFmpeg command: {command}")
        self.pipe = sp.Popen(
            command,
            stdin=sp.PIPE
        )
        self.stopped = False

    def write(self, frame):
        if self.stopped:
            self.in_size = tuple(frame.shape[:2][::-1])
            self._start()

        self.pipe.stdin.write(frame.tobytes())

    def stop(self):
        self.logger.info("Stop VideoWriter")
        if self.pipe is not None:
            self.pipe.stdin.close()
            self.pipe.kill()
            self.pipe.wait()
