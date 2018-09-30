import VideoReaderWrapper
import os
import threading

def get():
	for i in range(500):
		tensor = VideoReaderWrapper.GetFrame("first", 0)

#inputFile = "rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4"
inputFile = "rtmp://b.sportlevel.com/relay/pooltop"
VideoReaderWrapper.EnableLogs(VideoReaderWrapper.LogsLevel.LOW, VideoReaderWrapper.LogsType.FILE)
VideoReaderWrapper.Initialize(inputFile)
pipeline = VideoReaderWrapper.StartProcessing()
getThread = threading.Thread(target=get)
getThread.start()

pipeline.join()
getThread.join()

print("joined")

VideoReaderWrapper.Close(CloseLevel.HARD)


