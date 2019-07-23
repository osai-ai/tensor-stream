import torch
from tensor_stream import TensorStreamConverter, LogsLevel, LogsType, FourCC, Planes, ResizeType
import time
import unittest
import os
import numbers
class TestTensorStream(unittest.TestCase):

    def test_constructor(self):
        path = os.path.dirname(os.path.abspath(__file__)) + "/../../tests/resources/billiard_1920x1080_420_100.h264"
        max_consumers = 5
        cuda_device = 0
        buffer_size = 10
        repeat_number = 5
        reader = TensorStreamConverter(path, max_consumers = max_consumers, cuda_device = cuda_device, buffer_size = buffer_size, repeat_number = repeat_number)
        self.assertEqual(reader.max_consumers, max_consumers)
        self.assertEqual(reader.cuda_device, cuda_device)
        self.assertEqual(reader.buffer_size, buffer_size)
        self.assertEqual(reader.repeat_number, repeat_number)
        self.assertEqual(reader.stream_url, path)

    def test_constructor_default(self):
        path = os.path.dirname(os.path.abspath(__file__)) + "/../../tests/resources/billiard_1920x1080_420_100.h264"
        reader = TensorStreamConverter(path)
        self.assertEqual(reader.max_consumers, 5)
        self.assertEqual(reader.cuda_device, torch.cuda.current_device())
        self.assertEqual(reader.buffer_size, 10)
        self.assertEqual(reader.repeat_number, 1)
        self.assertEqual(reader.stream_url, path)

    def test_initialize_wrong_path(self):
        reader = TensorStreamConverter("wrong.h264", max_consumers = 5, cuda_device = 0, buffer_size = 10, repeat_number=5)
        with self.assertRaises(RuntimeError):
            reader.initialize()

    def test_initialize_correct_path(self):
        path = os.path.dirname(os.path.abspath(__file__)) + "/../../tests/resources/billiard_1920x1080_420_100.h264"
        reader = TensorStreamConverter(path, max_consumers = 5, cuda_device = 0, buffer_size = 10, repeat_number=5)
        reader.initialize()
        self.assertEqual(reader.frame_size, (1920, 1080))
        self.assertEqual(reader.fps, 25)

    def test_stop_without_init(self):
        path = os.path.dirname(os.path.abspath(__file__)) + "/../../tests/resources/billiard_1920x1080_420_100.h264"
        reader = TensorStreamConverter(path)
        reader.stop()

    def test_logs_enabling(self):
        path = os.path.dirname(os.path.abspath(__file__)) + "/../../tests/resources/billiard_1920x1080_420_100.h264"
        reader = TensorStreamConverter(path)
        reader.enable_logs(LogsLevel.LOW, LogsType.CONSOLE)
        reader.enable_nvtx()

    def test_start_close(self):
        path = os.path.dirname(os.path.abspath(__file__)) + "/../../tests/resources/billiard_1920x1080_420_100.h264"
        reader = TensorStreamConverter(path)
        reader.initialize()
        reader.start()
        time.sleep(1.0)
        reader.stop()

    def test_close_start(self):
        path = os.path.dirname(os.path.abspath(__file__)) + "/../../tests/resources/billiard_1920x1080_420_100.h264"
        reader = TensorStreamConverter(path)
        reader.initialize()
        reader.stop()
        #won't work but at least no crush
        reader.start()
    
    def test_start_read_close(self):
        path = os.path.dirname(os.path.abspath(__file__)) + "/../../tests/resources/billiard_1920x1080_420_100.h264"
        reader = TensorStreamConverter(path)
        reader.initialize()
        reader.start()
        time.sleep(1.0)
        tensor = reader.read()
        self.assertEqual(tensor.shape[0], 1080)
        self.assertEqual(tensor.shape[1], 1920)
        self.assertEqual(tensor.shape[2], 3)
        reader.stop()

    def test_return_index(self):
        path = os.path.dirname(os.path.abspath(__file__)) + "/../../tests/resources/billiard_1920x1080_420_100.h264"
        reader = TensorStreamConverter(path)
        reader.initialize()
        reader.start()
        time.sleep(1.0)
        tensor, index = reader.read(return_index = True)
        self.assertTrue(index > 0 and index < 100)
        reader.stop()

    def test_normalization(self):
        path = os.path.dirname(os.path.abspath(__file__)) + "/../../tests/resources/billiard_1920x1080_420_100.h264"
        reader = TensorStreamConverter(path)
        reader.initialize()
        reader.start()
        time.sleep(1.0)
        tensor = reader.read(normalization = True)
        value = tensor[0][0][0].item()
        self.assertEqual(type(value), float)
        reader.stop()
    
    def test_read_without_init(self):
        path = os.path.dirname(os.path.abspath(__file__)) + "/../../tests/resources/billiard_1920x1080_420_100.h264"
        reader = TensorStreamConverter(path)
        reader.start()
        time.sleep(1.0)
        with self.assertRaises(RuntimeError):
            tensor, index = reader.read(return_index = True)

        reader.stop()
    
    def test_check_dump_size(self):
        path = os.path.dirname(os.path.abspath(__file__)) + "/../../tests/resources/billiard_1920x1080_420_100.h264"
        reader = TensorStreamConverter(path)
        reader.initialize()
        reader.start()
        time.sleep(1.0)
        expected_width = 1920
        expected_height = 1080
        expected_channels = 3
        tensor, index = reader.read(return_index = True)
        self.assertEqual(tensor.shape[0], expected_height)
        self.assertEqual(tensor.shape[1], expected_width)
        self.assertEqual(tensor.shape[2], expected_channels)
        #need to find dumped file and compare expected and real sizes
        reader.dump(tensor)

        dump_size = os.stat('default.yuv')
        os.remove("default.yuv")
        self.assertEqual(dump_size.st_size, expected_width * expected_height * expected_channels)
        reader.stop()

    def test_read_without_init_start(self):
        path = os.path.dirname(os.path.abspath(__file__)) + "/../../tests/resources/billiard_1920x1080_420_100.h264"
        reader = TensorStreamConverter(path)
        time.sleep(1.0)
        with self.assertRaises(RuntimeError):
            tensor, index = reader.read(return_index = True)

        reader.stop()
    
    def test_dump_name(self):
        path = os.path.dirname(os.path.abspath(__file__)) + "/../../tests/resources/billiard_1920x1080_420_100.h264"
        reader = TensorStreamConverter(path)
        reader.initialize()
        reader.start()
        time.sleep(1.0)
        tensor = reader.read()
        #need to find dumped file and compare expected and real sizes
        reader.dump(tensor, name = "dump")
        self.assertTrue(os.path.isfile("dump.yuv")) 
        os.remove("dump.yuv")
        reader.stop()
    
    def test_multiple_init(self):
        path = os.path.dirname(os.path.abspath(__file__)) + "/../../tests/resources/billiard_1920x1080_420_100.h264"
        reader = TensorStreamConverter(path)
        number_close_init = 10
        while number_close_init > 0:
            reader.initialize()
            reader.stop()
            number_close_init -= 1

    def test_read_after_stop(self):
        path = os.path.dirname(os.path.abspath(__file__)) + "/../../tests/resources/billiard_1920x1080_420_100.h264"
        reader = TensorStreamConverter(path)
        reader.initialize()
        reader.start()
        time.sleep(1.0)
        reader.stop()
        with self.assertRaises(RuntimeError):
            tensor = reader.read()
    
    def test_frame_number(self):
        path = os.path.dirname(os.path.abspath(__file__)) + "/../../tests/resources/billiard_1920x1080_420_100.h264"
        reader = TensorStreamConverter(path)
        reader.initialize()
        reader.start()
        time.sleep(1.0)
        frame_num = i = 10
        while i > 0:
            tensor = reader.read()
            reader.dump(tensor)
            i -= 1

        dump_size = os.stat('default.yuv')
        print(f"SIZE {dump_size}")
        os.remove("default.yuv")
        expected_width = 1920
        expected_height = 1080
        expected_channels = 3
        self.assertEqual(dump_size.st_size, expected_width * expected_height * expected_channels * frame_num)
        reader.stop()

if __name__ == '__main__':
    unittest.main()