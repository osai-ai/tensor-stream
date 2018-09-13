import torch;
import VideoReader;
from multiprocessing import Process
import os
import threading

def get():
	for i in range(100):
		VideoReader.get("first", 0)

def start():
	VideoReader.start()	

VideoReader.init()	
t1 = threading.Thread(target=someFunc)
t1.start()
print("here")


