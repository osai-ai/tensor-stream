import torch
from tensor_stream import TensorStreamConverter
from tensor_stream import LogsLevel, LogsType, FourCC, Planes, FrameRate, ResizeType
from threading import Thread

index = 0

def consumer(reader):
    parameters = {'width': 1920,
                  'height': 1080}
    global index
    result = reader.read_absolute(batch=(125,126,127,128,129), **parameters)
    reader.dump(result, f"temp_{index}", **parameters)
    index = index + 1

if __name__ == '__main__':
    readers = []
    for i in range(0, 12):
        cuda = 0
        if i < 6:
            cuda = 1
        reader = TensorStreamConverter("tests/resources/tennis_2s.mp4", 0, 0, 0, cuda=cuda, threads=0)
        # To log initialize stage, logs should be defined before initialize call
        reader.enable_logs(LogsLevel.LOW, LogsType.CONSOLE)
        reader.enable_nvtx()

        reader.initialize(repeat_number=20)
        reader.enable_batch_optimization()
        readers.append(reader)

    consumer(readers[i])

    threads = []
    for i in range(0, len(readers)):
        threads.append(Thread(target=consumer, args=(readers[i],)))
        threads[i].start()

    for i in range(0, len(readers)):
        threads[i].join()
        readers[i].stop()