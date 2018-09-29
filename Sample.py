import torch;
import VideoReader;
from multiprocessing import Process
import os
import threading

def get():
	for i in range(500):
		tensor = VideoReader.get("first", 0)
		#print(tensor)

def start():
	VideoReader.start()	

print("Init")
inputFile = "rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4"
#"rtmp://b.sportlevel.com/relay/pooltop"

#3 levels of logs: Low = 1, Medium = 2, High = 3. 
#If use negative form of argument logs will be printed to console instead of file
VideoReader.enableLogs(-2)
status = 1
repeatNumber = 0
endCount = 20
while (status != 0 and repeatNumber < endCount):
	status = VideoReader.init(inputFile)
	if (status != 0):
		#Mode 1 - full close, mode 2 - soft close (for reset)
		VideoReader.close(2)
	repeatNumber = 	repeatNumber + 1

if (repeatNumber == endCount):
	exit()

print("Status")
print(status)

t1 = threading.Thread(target=start)
t2 = threading.Thread(target=get)

t1.start()
t2.start()

t1.join()
t2.join()

print("joined")

VideoReader.close(1)


