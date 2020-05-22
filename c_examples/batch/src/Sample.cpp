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

void get_cycle(std::shared_ptr<TensorStream> reader, FrameParameters frameParameters, std::vector<int> indexes) {
	std::shared_ptr<Logger> logger = std::make_shared<Logger>();
	auto result = reader->getFrameAbsolute<unsigned char>(indexes, frameParameters);
}

int main() {
	auto cpuNumber = std::thread::hardware_concurrency();
	std::vector<std::shared_ptr<TensorStream> > readers{ 6 };
	for (auto& reader : readers) {
		reader = std::make_shared<TensorStream>();
		reader->enableLogs(-LOW);
		reader->enableNVTX();
		int sts = VREADER_OK;
		int initNumber = 10;

		while (initNumber--) {
			sts = reader->initPipeline("D:/Work/argus-tensor-stream/tests/resources/tennis_2s.mp4", 0, 0, 0);
			if (sts != VREADER_OK)
				reader->endProcessing();
			else
				break;
		}

		reader->enableBatchOptimization();
		reader->skipAnalyzeStage();
		CHECK_STATUS(sts);
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
	std::map<std::string, std::string> executionParameters = { {"dumpName", std::to_string(std::get<0>(cropBotRight) - std::get<0>(cropTopLeft)) + "x" + std::to_string(std::get<1>(cropBotRight) - std::get<1>(cropTopLeft)) + "1.yuv"} };
	std::vector<std::thread> threads{ 6 };
	for (int i = 0; i < readers.size(); i++) {
		std::vector<int> frames = { 125 - i * 10, 126 - i * 10, 127 - i * 10, 128 - i * 10, 129 - i * 10 };
		threads[i] = std::thread(get_cycle, readers[i], frameParameters, frames);
		//threads[i].join();
	}
	for (int i = 0; i < threads.size(); i++) {
		threads[i].join();
	}
	return 0;
}