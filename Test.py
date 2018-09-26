import torch;
import VideoReader;
from multiprocessing import Process
import os
import threading

tensor = torch.tensor((), dtype=torch.uint8)
def get():
	for i in range(500):
		tensor = VideoReader.get("first", 0)
		print(tensor)
		del tensor
		#VideoReader.free(x)
		#torch.cuda.empty_cache()

def start():
	VideoReader.start()	

print("Init")
inputFile = "rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4"
#"rtmp://b.sportlevel.com/relay/pooltop"
VideoReader.enableLogs(0)
VideoReader.init(inputFile)	

t1 = threading.Thread(target=start)
t2 = threading.Thread(target=get)

t1.start()
t2.start()

t1.join()
t2.join()

print("joined")

VideoReader.close()


