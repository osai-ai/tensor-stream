import VideoReaderWrapper
import threading


def get():
    for i in range(500):
        tensor = VideoReaderWrapper.GetFrame("name")


# inputFile = "rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4"
inputFile = "rtmp://b.sportlevel.com/relay/pooltop"
VideoReaderWrapper.EnableLogs(VideoReaderWrapper.LogsLevel.MEDIUM, VideoReaderWrapper.LogsType.CONSOLE)

try:
    VideoReaderWrapper.Initialize(inputFile, 20)
except RuntimeError as error:
    print(error.args)

pipeline = VideoReaderWrapper.StartProcessing()
getThread = threading.Thread(target=get)
getThread.start()

pipeline.join()
getThread.join()

print("joined")

#Call close when need to complete pipeline thread and clear all reserved by VideoReader resources
VideoReaderWrapper.Close(VideoReaderWrapper.CloseLevel.HARD)
