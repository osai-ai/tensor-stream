#include "WrapperC.h"
#include <experimental/filesystem> //C++17 

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
				auto result = reader.getFrame<float>(executionParameters["name"], { std::atoi(executionParameters["delay"].c_str()) }, frameParameters);
				if (!fileName.empty()) {
					int status = reader.dumpFrame<float>((float*)std::get<0>(result), frameParameters, dumpFile);
					if (status < 0)
						return;
				}
				cudaFree(std::get<0>(result));
			}
			else {
				auto result = reader.getFrame<unsigned char>(executionParameters["name"], { std::atoi(executionParameters["delay"].c_str()) }, frameParameters);
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

int startSimplePipeline() {
	reader.enableLogs(-MEDIUM);
	reader.enableNVTX();
	int sts = VREADER_OK;
	int initNumber = 10;

	while (initNumber--) {
		sts = reader.initPipeline("rtmp://37.228.119.44:1935/vod/big_buck_bunny.mp4", 5, 0, 5);
		if (sts != VREADER_OK)
			reader.endProcessing();
		else
			break;
	}

	reader.skipAnalyzeStage();
	CHECK_STATUS(sts);
	std::thread pipeline([] { reader.startProcessing(); });
	int dstWidth = 720;
	int dstHeight = 480;
	std::tuple<int, int> cropTopLeft = { 0, 0 };
	std::tuple<int, int> cropBotRight = { 0, 0 };
	ColorOptions colorOptions = { FourCC::NV12 };
	colorOptions.planesPos = Planes::PLANAR;
	colorOptions.normalization = false;
	ResizeOptions resizeOptions = { dstWidth, dstHeight };
	CropOptions cropOptions = { cropTopLeft, cropBotRight };
	FrameParameters frameParameters = { resizeOptions, colorOptions, cropOptions };

	std::map<std::string, std::string> executionParameters = { {"name", "first"}, {"delay", "0"}, {"frames", "50"},
															   {"dumpName", std::to_string(std::get<0>(cropBotRight) - std::get<0>(cropTopLeft)) + "x" + std::to_string(std::get<1>(cropBotRight) - std::get<1>(cropTopLeft)) + ".yuv"} };
	std::thread get(get_cycle, frameParameters, executionParameters);
	get.join();
	reader.endProcessing();
	pipeline.join();
	return 0;
}

size_t getGraphicDeviceVRamUsage()
{
	size_t l_free = 0;
	size_t l_Total = 0;
	cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);

	return (l_Total - l_free);
}

void get_cycle_batch(FrameParameters frameParameters, std::map<std::string, std::string> executionParameters, std::vector<int> frames) {
	for (int i = 0; i < 1; i++) {
		size_t before = getGraphicDeviceVRamUsage();

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

		size_t after = getGraphicDeviceVRamUsage();
		std::cout << i << " " << before << "->" << after << std::endl;
	}

}

int startBatchPipeline() {
	reader.enableLogs(-LOW);
	reader.enableNVTX();
	int sts = VREADER_OK;
	int initNumber = 10;

	while (initNumber--) {
		//sts = reader.initPipeline("rtmp://37.228.119.44:1935/vod/big_buck_bunny.mp4", 5, 0, 5);
		sts = reader.initPipeline("basler_cr_train_005.mp4", 0, 0, 0);
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
	//std::vector<int> frames = { 310, 100, 1341, 5012 };
	std::vector<int> frames = { 0, 100, 1341, 5012 };
	std::thread get(get_cycle_batch, frameParameters, executionParameters, frames);
	
	executionParameters = { {"dumpName", std::to_string(std::get<0>(cropBotRight) - std::get<0>(cropTopLeft)) + "x" + std::to_string(std::get<1>(cropBotRight) - std::get<1>(cropTopLeft)) + "2.yuv"} };
	frames = { 310, 100, 2341, 3012 };
	std::thread get1(get_cycle_batch, frameParameters, executionParameters, frames);

	get.join();
	get1.join();

	reader.endProcessing();
	return 0;
}

int main() {
	//return startSimplePipeline();
	return startBatchPipeline();
}