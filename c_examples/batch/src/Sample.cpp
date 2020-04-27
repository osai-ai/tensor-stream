#include "WrapperC.h"
#include <experimental/filesystem> //C++17 

TensorStream reader;

void get_cycle_batch(FrameParameters frameParameters, std::map<std::string, std::string> executionParameters, std::vector<int> frames) {
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

		result = reader.getFrameAbsolute<unsigned char>({0, 100, 1000}, frameParameters);
	}

}

int main() {
	reader.enableLogs(-LOW);
	reader.enableNVTX();
	int sts = VREADER_OK;
	int initNumber = 10;

	while (initNumber--) {
		sts = reader.initPipeline("../../../basler_office_005.mp4", 0, 0, 0);
		if (sts != VREADER_OK)
			reader.endProcessing();
		else
			break;
	}

	reader.skipAnalyzeStage();
	CHECK_STATUS(sts);
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
	std::vector<int> frames = { 18715, 18716 };
	std::thread get(get_cycle_batch, frameParameters, executionParameters, frames);

	get.join();

	reader.endProcessing();
	return 0;
}