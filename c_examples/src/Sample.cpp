#include "WrapperC.h"
#include <filesystem> //C++17 

TensorStream reader;

void get_cycle(FrameParameters frameParameters, std::map<std::string, std::string> executionParameters) {
	try {
		int frames = std::atoi(executionParameters["frames"].c_str());
		if (!frames)
			return;

		std::shared_ptr<FILE> dumpFile;
		std::string fileName = executionParameters["dumpName"];
		if (!fileName.empty()) {
			remove(fileName.c_str());

			dumpFile = std::shared_ptr<FILE>(fopen(fileName.c_str(), "ab"), std::fclose);
		}
		for (int i = 0; i < frames; i++) {
			if (frameParameters.color.normalization) {
				auto result = reader.getFrame<float>(executionParameters["name"], std::atoi(executionParameters["delay"].c_str()), frameParameters);
				if (!fileName.empty()) {
					int status = reader.dumpFrame<float>((float*)std::get<0>(result), frameParameters, dumpFile);
					if (status < 0)
						return;
				}
				cudaFree(std::get<0>(result));
			}
			else {
				auto result = reader.getFrame<unsigned char>(executionParameters["name"], std::atoi(executionParameters["delay"].c_str()), frameParameters);
				if (!fileName.empty()) {
					int status = reader.dumpFrame<unsigned char>((unsigned char*)std::get<0>(result), frameParameters, dumpFile);
					if (status < 0)
						return;
				}
				cudaFree(std::get<0>(result));
			}
		}
	}
	catch (std::runtime_error e) {
		return;
	}

}

int main()
{
	reader.enableLogs(-LOW);
	reader.enableNVTX();
	int sts = VREADER_OK;
	int initNumber = 10;

	while (initNumber--) {
		sts = reader.initPipeline("BalanceTest-BalanceSingleDigitRecording.flv", 5, 0, 1, FrameRateMode::BLOCKING);
		if (sts != VREADER_OK)
			reader.endProcessing();
		else
			break;
	}

	reader.skipAnalyzeStage();
	CHECK_STATUS(sts);
	std::thread pipeline([] { reader.startProcessing(); reader.drain(); });
	int dstWidth = 720;
	int dstHeight = 480;
	std::tuple<int, int> cropTopLeft = { 0, 0 };
	std::tuple<int, int> cropBotRight = { 0, 0 };
	ColorOptions colorOptions = { FourCC::NV12 };
	colorOptions.planesPos = Planes::PLANAR;
	colorOptions.normalization = false;
	ResizeOptions resizeOptions = { dstWidth, dstHeight };
	CropOptions cropOptions = { cropTopLeft, cropBotRight };
	FrameParameters frameParameters = {resizeOptions, colorOptions, cropOptions};

	std::map<std::string, std::string> executionParameters = { {"name", "first"}, {"delay", "0"}, {"frames", "250"}, 
															   {"dumpName", "720x480_1.yuv"} };
	std::thread get(get_cycle, frameParameters, executionParameters);
	get.join();
	pipeline.join();
	reader.endProcessing();
	return 0;
}