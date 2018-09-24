import torch;
import VideoReader;
from multiprocessing import Process
import os
import threading

def get():
	for i in range(500):
		x = VideoReader.get("first", 0)
		print(x.data)
		VideoReader.free(x)
		torch.cuda.empty_cache()

def start():
	VideoReader.start()	

print("Init")
inputFile = "rtmp://b.sportlevel.com/relay/pooltop"
VideoReader.init(inputFile)	
VideoReader.enableLogs(3)

t1 = threading.Thread(target=start)
t2 = threading.Thread(target=get)

t1.start()
t2.start()

t1.join()
t2.join()

print("joined")

VideoReader.close()


