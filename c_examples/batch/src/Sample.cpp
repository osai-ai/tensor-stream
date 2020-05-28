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

/*
1) Каждая "пачка" кадров должна обрабатываться в отдельном ИНСТАНСЕ TensorReader, паралеллить с помощью потоков не получится из-за seek
2) На каждый стрим нужно создавать пару инстансов с SW и HW, запрашивать кадры нужны из SW И из HW одновременно, т.е. готовить сразу два батча (неважно инстансы с разными стримами или одинаковыми)
3) Как сделать, чтобы они одновременно запускались и работали параллельно в Python? Запускать в разных тредах в питоне и откреплять GIL в С++
*/

int main() {
	auto cpuNumber = std::thread::hardware_concurrency();
	std::vector<std::shared_ptr<TensorStream> > readers{ 1 };
	int index = 0;
	for (auto& reader : readers) {
		reader = std::make_shared<TensorStream>();
		reader->enableLogs(-LOW);
		reader->enableNVTX();
		int sts = VREADER_OK;
		int initNumber = 10;

		while (initNumber--) {
			sts = reader->initPipeline("D:/Work/argus-tensor-stream/tests/resources/tennis_2s.mp4", 0, 0, 0, FrameRateMode::NATIVE, 0, 0);
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
	std::map<std::string, std::string> executionParameters = { {"dumpName", std::to_string(std::get<0>(cropBotRight) - std::get<0>(cropTopLeft)) + "x" + std::to_string(std::get<1>(cropBotRight) - std::get<1>(cropTopLeft)) + "1.yuv"} };
	std::vector<std::thread> threads{ 1 };
	for (int i = 0; i < readers.size(); i++) {
		std::vector<int> frames = { 125, 126, 127, 128, 129 };
		threads[i] = std::thread([=]() { readers[i]->getFrameAbsolute<unsigned char>(frames, frameParameters); });
	}
	for (int i = 0; i < threads.size(); i++) {
		threads[i].join();
	}
	return 0;
}