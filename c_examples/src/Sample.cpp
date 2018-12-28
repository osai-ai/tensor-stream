#include "Wrapper.h"

void get_cycle(std::map<std::string, std::string> parameters) {
	try {
		for (int i = 0; i < 100; i++) {
			getFrame(parameters["name"], std::atoi(parameters["delay"].c_str()), std::atoi(parameters["format"].c_str()),
				std::atoi(parameters["width"].c_str()), std::atoi(parameters["height"].c_str()));
		}
	}
	catch (std::runtime_error e) {
		return;
	}

}

int main()
{

	enableLogs(-MEDIUM);
	//int sts = initPipeline("rtmp://b.sportlevel.com/relay/pooltop");
	int sts = initPipeline("rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4");
	//int sts = initPipeline("../streams/Without_first_non-IDR.h264");
	//int sts = initPipeline("../bitstream.h264");
	CHECK_STATUS(sts);
	std::thread pipeline(processingWrapper);
	std::map<std::string, std::string> parameters = { {"name", "first"}, {"delay", "0"}, {"format", std::to_string(BGR24)} };
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
	endProcessing(HARD);
	pipeline.join();
	return 0;
}