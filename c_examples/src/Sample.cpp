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
				auto result = reader.getFrame<float>(executionParameters["name"], std::atoi(executionParameters["delay"].c_str()), frameParameters);
				if (!fileName.empty()) {
					int status = reader.dumpFrame<float>((float*)std::get<0>(result), frameParameters, dumpFile);
					if (status < 0)
						return;
				}
			}
			else {
				auto result = reader.getFrame<unsigned char>(executionParameters["name"], std::atoi(executionParameters["delay"].c_str()), frameParameters);
				if (!fileName.empty()) {
					int status = reader.dumpFrame<unsigned char>((unsigned char*)std::get<0>(result), frameParameters, dumpFile);
					if (status < 0)
						return;
				}
			}
		}
	}
	catch (std::runtime_error e) {
		return;
	}

}

int main()
{
	//reader.enableLogs(-HIGH);
	reader.enableNVTX();
	int sts = VREADER_OK;
	int initNumber = 10;

	while (initNumber--) {
		sts = reader.initPipeline("rtmp://b.sportlevel.com/relay/pooltop");
 		//sts = reader.initPipeline("..\\..\\tests\\resources\\test_resize\\forest.jpg");
		if (sts != VREADER_OK)
			reader.endProcessing();
		else
			break;
	}

	reader.skipAnalyzeStage();
	CHECK_STATUS(sts);
	std::thread pipeline([] { reader.startProcessing(); });
	int dstWidth = 1280;
	int dstHeight = 720;
	ColorOptions colorOptions = { FourCC::NV12 };
	colorOptions.planesPos = Planes::PLANAR;
	colorOptions.normalization = false;
	ResizeOptions resizeOptions = { dstWidth, dstHeight };
	resizeOptions.type = ResizeType::BICUBIC;
	FrameParameters frameParameters = {resizeOptions, colorOptions};

	std::map<std::string, std::string> executionParameters = { {"name", "first"}, {"delay", "0"}, {"frames", "1"}, {"dumpName", "sample_output.yuv"} };
	std::thread get(get_cycle, frameParameters, executionParameters);
	get.join();
	reader.endProcessing();
	pipeline.join();
	return 0;
}