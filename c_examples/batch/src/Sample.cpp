#include "WrapperC.h"
#include <experimental/filesystem> //C++17 
#include <chrono>
#include <thread>

void get_cycle_batch(TensorStream& reader, FrameParameters frameParameters, std::map<std::string, std::string> executionParameters, std::vector<int> frames) {
	for (int i = 0; i < 1; i++) {
		std::shared_ptr<FILE> dumpFile;
		std::string fileName = executionParameters["dumpName"];
		if (!fileName.empty()) {
			remove(fileName.c_str());
			dumpFile = std::shared_ptr<FILE>(fopen(fileName.c_str(), "ab"), std::fclose);
		}

		auto result = reader.getFrameAbsolute<unsigned char>(frames, frameParameters);
		if (!fileName.empty()) {
			for (auto frame : result) {
				int status = reader.dumpFrame<unsigned char>((unsigned char*)frame, frameParameters, dumpFile);
				if (status < 0)
					return;
				cudaFree(frame);
			}
		}

		result = reader.getFrameAbsolute<unsigned char>({10, 11, 20}, frameParameters);
	}

}

void processStream(std::shared_ptr<TensorStream> reader, std::vector<int> frames, std::string path, FrameParameters frameParameters, std::shared_ptr<FILE> dumpFile) {
	reader->resetPipeline(path);
	auto result = reader->getFrameAbsolute<unsigned char>(frames, frameParameters);
	for (auto frame : result) {
		int status = reader->dumpFrame<unsigned char>((unsigned char*)frame, frameParameters, dumpFile);
		if (status < 0)
			return;
	}
	for (auto frame : result) {
		cudaFree(frame);
	}
}

//TODO: if decoder was SW but became HW and vice versa

//TODO: StreamPool synchronization 
int main() {
	auto cpuNumber = std::thread::hardware_concurrency();
	std::vector<std::shared_ptr<TensorStream> > readers{ 2 };
	int index = 0;
	for (auto& reader : readers) {
		reader = std::make_shared<TensorStream>();
		/*
		std::shared_ptr<StreamPool> streamPool = std::make_shared<StreamPool>();
		streamPool->cacheStream("D:/Work/argus-tensor-stream/tests/resources/1.mp4");
		streamPool->cacheStream("D:/Work/argus-tensor-stream/tests/resources/2.mp4");
		streamPool->cacheStream("D:/Work/argus-tensor-stream/tests/resources/3.mp4");
		streamPool->cacheStream("D:/Work/argus-tensor-stream/tests/resources/4.mp4");
		streamPool->cacheStream("D:/Work/argus-tensor-stream/tests/resources/5.mp4");
		streamPool->cacheStream("D:/Work/argus-tensor-stream/tests/resources/6.mp4");
		streamPool->cacheStream("D:/Work/argus-tensor-stream/tests/resources/7.mp4");
		streamPool->cacheStream("D:/Work/argus-tensor-stream/tests/resources/8.mp4");
		streamPool->cacheStream("D:/Work/argus-tensor-stream/tests/resources/9.mp4");
		streamPool->cacheStream("D:/Work/argus-tensor-stream/tests/resources/10.mp4");
		streamPool->cacheStream("D:/Work/argus-tensor-stream/tests/resources/11.mp4");
		reader->addStreamPool(streamPool);
		*/
		reader->enableLogs(-LOW);
		reader->enableNVTX();
		int sts = VREADER_OK;
		int initNumber = 10;
		while (initNumber--) {
			sts = reader->initPipeline("D:/Work/argus-tensor-stream/tests/resources/1.mp4", 0, 0, 0, FrameRateMode::NATIVE, index % 2, 0);
			if (sts != VREADER_OK)
				reader->endProcessing();
			else
				break;
		}

		reader->enableBatchOptimization();
		reader->skipAnalyzeStage();
		CHECK_STATUS(sts);
		index++;
	}

	int dstWidth = 1920;
	int dstHeight = 1080;
	std::tuple<int, int> cropTopLeft = { 0, 0 };
	std::tuple<int, int> cropBotRight = { 0, 0 };
	ColorOptions colorOptions = { FourCC::NV12 };
	colorOptions.planesPos = Planes::PLANAR;
	colorOptions.normalization = false;
	ResizeOptions resizeOptions = { dstWidth, dstHeight };
	CropOptions cropOptions = { cropTopLeft, cropBotRight };
	FrameParameters frameParameters = { resizeOptions, colorOptions, cropOptions };
	std::vector<std::thread> threads{ 2 };
	for (int i = 0; i < readers.size(); i++) {
		threads[i] = std::thread([=]() {
			std::shared_ptr<FILE> dumpFile;
			std::string fileName = "test" + std::to_string(i) + ".yuv";
			if (!fileName.empty()) {
				remove(fileName.c_str());
				dumpFile = std::shared_ptr<FILE>(fopen(fileName.c_str(), "ab"), std::fclose);
			}

			std::vector<int> frames = { 0, 100, 200, 120, 100, 101 };
			processStream(readers[i], frames, "D:/Work/argus-tensor-stream/tests/resources/1.mp4", frameParameters, dumpFile);
			processStream(readers[i], frames, "D:/Work/argus-tensor-stream/tests/resources/2.mp4", frameParameters, dumpFile);
			processStream(readers[i], frames, "D:/Work/argus-tensor-stream/tests/resources/3.mp4", frameParameters, dumpFile);
			processStream(readers[i], frames, "D:/Work/argus-tensor-stream/tests/resources/4.mp4", frameParameters, dumpFile);
			processStream(readers[i], frames, "D:/Work/argus-tensor-stream/tests/resources/5.mp4", frameParameters, dumpFile);
			processStream(readers[i], frames, "D:/Work/argus-tensor-stream/tests/resources/6.mp4", frameParameters, dumpFile);
			processStream(readers[i], frames, "D:/Work/argus-tensor-stream/tests/resources/7.mp4", frameParameters, dumpFile);
			processStream(readers[i], frames, "D:/Work/argus-tensor-stream/tests/resources/8.mp4", frameParameters, dumpFile);
			processStream(readers[i], frames, "D:/Work/argus-tensor-stream/tests/resources/9.mp4", frameParameters, dumpFile);
			processStream(readers[i], frames, "D:/Work/argus-tensor-stream/tests/resources/10.mp4", frameParameters, dumpFile);
			processStream(readers[i], frames, "D:/Work/argus-tensor-stream/tests/resources/11.mp4", frameParameters, dumpFile);
		});
	}
	for (int i = 0; i < threads.size(); i++) {
		threads[i].join();
	}
	return 0;
}