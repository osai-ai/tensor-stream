#include "WrapperC.h"

TensorStream reader;

void get_cycle(VPPParameters videoOptions, std::map<std::string, std::string> parameters) {
	try {
		int frames = std::atoi(parameters["frames"].c_str());
		std::shared_ptr<FILE> dumpFile(std::shared_ptr<FILE>(fopen(parameters["dumpName"].c_str(), "ab"), std::fclose));
		for (int i = 0; i < frames; i++) {
			auto result = reader.getFrame(parameters["name"], std::atoi(parameters["delay"].c_str()), videoOptions);
			int status = reader.dumpFrame((float*) std::get<0>(result), videoOptions, dumpFile);
			
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
	//int sts = reader.initPipeline("rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4");
	int sts = reader.initPipeline("rtmp://b.sportlevel.com/relay/pooltop");
	CHECK_STATUS(sts);
	std::thread pipeline(&TensorStream::startProcessing, &reader);
	int dstWidth = 720;
	int dstHeight = 480;
	ColorParameters colorOption = { false, Planes::MERGED, BGR24 };
	VPPParameters videoOptions = {dstWidth, dstHeight, colorOption};

	std::map<std::string, std::string> parameters = { {"name", "first"}, {"delay", "0"}, {"frames", "100"}, {"dumpName", "sample_output.yuv"} };
	std::thread get(get_cycle, videoOptions, parameters);
	get.join();
	reader.endProcessing(HARD);
	pipeline.join();
	return 0;
}