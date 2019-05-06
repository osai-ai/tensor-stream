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
			
			if (std::experimental::filesystem::exists(fileName))
				std::experimental::filesystem::remove_all(fileName);

			dumpFile = std::shared_ptr<FILE>(fopen(fileName.c_str(), "ab"), std::fclose);
		}
		for (int i = 0; i < frames; i++) {
			auto result = reader.getFrame(executionParameters["name"], std::atoi(executionParameters["delay"].c_str()), frameParameters);
			if (!fileName.empty()) {
				int status = reader.dumpFrame((float*)std::get<0>(result), frameParameters, dumpFile);
				if (status < 0)
					return;
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
	int sts = VREADER_OK;
	int initNumber = 10;

	while (initNumber--) {
		//sts = reader.initPipeline("rtmp://b.sportlevel.com/relay/pooltop");
		sts = reader.initPipeline("rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4");
		if (sts != VREADER_OK)
			reader.endProcessing(SOFT);
		else
			break;
	}
	CHECK_STATUS(sts);
	std::thread pipeline(&TensorStream::startProcessing, &reader);
	int dstWidth = 720;
	int dstHeight = 480;
	ColorOptions colorOptions = { false, Planes::MERGED, BGR24 };
	ResizeOptions resizeOptions = { dstWidth, dstHeight, ResizeType::NEAREST };
	FrameParameters frameParameters = {resizeOptions, colorOptions};

	std::map<std::string, std::string> executionParameters = { {"name", "first"}, {"delay", "0"}, {"frames", "100"}, {"dumpName", "sample_output.yuv"} };
	std::thread get(get_cycle, frameParameters, executionParameters);
	get.join();
	reader.endProcessing(HARD);
	pipeline.join();
	return 0;
}