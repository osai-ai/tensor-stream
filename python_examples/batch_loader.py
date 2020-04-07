import torch
from tensor_stream import TensorStreamConverter
from tensor_stream import FourCC
import matplotlib.pyplot as plt

reader = TensorStreamConverter("D:\\Work\\argus-tensor-stream\\c_examples\\build\\basler_cr_train_005.mp4")
reader.initialize(repeat_number=20)

parameters = {'pixel_format': FourCC.RGB24,
              'width': 1920,
              'height': 1080}

for i in range(0, 10000):
    result1 = reader.read_absolute(batch=[310, 100, 1341, 5012], name="first", **parameters)
    result2 = reader.read_absolute(batch=[310, 100, 1341, 5012], name="second", **parameters)
    result3 = reader.read_absolute(batch=[310, 100, 1341, 5012], name="third", **parameters)
    result4 = reader.read_absolute(batch=[310, 100, 1341, 5012], name="fourth", **parameters)
