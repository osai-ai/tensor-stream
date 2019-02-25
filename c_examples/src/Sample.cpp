#include "WrapperC.h"

TensorStream reader;

void get_cycle(std::map<std::string, std::string> parameters) {
	try {
		int width = std::atoi(parameters["width"].c_str());
		int height = std::atoi(parameters["height"].c_str());
		FourCC format = (FourCC) std::atoi(parameters["format"].c_str());
		int frames = std::atoi(parameters["frames"].c_str());
		std::shared_ptr<FILE> dumpFile(std::shared_ptr<FILE>(fopen(parameters["dumpName"].c_str(), "ab"), std::fclose));
		for (int i = 0; i < frames; i++) {
			auto result = reader.getFrame(parameters["name"], std::atoi(parameters["delay"].c_str()), format,
				width, height);
			int status = reader.dumpFrame(std::get<0>(result), width, height, format, dumpFile);
			if (status < 0)
				return;

		}
	}
	catch (std::runtime_error e) {
		return;
	}

}

int main()
{
	reader.enableLogs(-MEDIUM);
	int sts = reader.initPipeline("rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4");
	CHECK_STATUS(sts);
	std::thread pipeline(&TensorStream::startProcessing, &reader);
	std::map<std::string, std::string> parameters = { {"name", "first"}, {"delay", "0"}, {"format", std::to_string(RGB24)}, {"width", "720"}, {"height", "480"},
													  {"frames", "100"}, {"dumpName", "sample_output.yuv"} };
	std::thread get(get_cycle, parameters);
	get.join();
	reader.endProcessing(HARD);
	pipeline.join();
	return 0;
}