#include "WrapperC.h"

VideoReader reader;

void get_cycle(std::map<std::string, std::string> parameters) {
	try {
		int width = std::atoi(parameters["width"].c_str());
		int height = std::atoi(parameters["height"].c_str());
		FourCC format = (FourCC) std::atoi(parameters["format"].c_str());
		std::shared_ptr<FILE> dumpFile(std::shared_ptr<FILE>(fopen("output.yuv", "ab"), std::fclose));
		for (int i = 0; i < 1000; i++) {
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
	//int sts = reader.initPipeline("rtmp://b.sportlevel.com/relay/pooltop");
	int sts = reader.initPipeline("rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4");
	CHECK_STATUS(sts);
	std::thread pipeline(&VideoReader::startProcessing, &reader);
	std::map<std::string, std::string> parameters = { {"name", "first"}, {"delay", "0"}, {"format", std::to_string(RGB24)}, {"width", "720"}, {"height", "480"}};
	std::thread get(get_cycle, parameters);
	/*
	parameters = { {"name", "second"}, {"delay", "0"}, {"format", std::to_string(RGB24)} };
	std::thread get2(get_cycle, parameters);
	parameters = { {"name", "third"}, {"delay", "0"}, {"format", std::to_string(BGR24)} };
	std::thread get3(get_cycle, parameters);
	*/
	get.join();
	/*
	get2.join();
	get3.join();
	*/
	reader.endProcessing(HARD);
	pipeline.join();
	return 0;
}