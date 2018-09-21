import torch;
import VideoReader;
from multiprocessing import Process
import os
import threading

def get():
	for i in range(1000):
		x = VideoReader.get("first", 0)
		print(x[0])
		print(x[0][0])
		print(x)
		del x
		torch.cuda.empty_cache()

def start():
	VideoReader.start()	

VideoReader.init()	

print("Init")
#VideoReader.start()
t1 = threading.Thread(target=start)

t1.start()
print("here")
t2 = threading.Thread(target=get)

t2.start()
print("after")

t1.join()
t2.join()


