#include <gtest/gtest.h>

#include "WrapperC.h"
extern "C" {
#include "libavutil/crc.h"
}

void get_cycle(std::map<std::string, std::string> parameters, VideoReader& reader) {
	try {
		int width = std::atoi(parameters["width"].c_str());
		int height = std::atoi(parameters["height"].c_str());
		FourCC format = (FourCC)std::atoi(parameters["format"].c_str());
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

TEST(Wrapper_Init, CorrectParams) {
	VideoReader reader;
	ASSERT_EQ(reader.initPipeline("../resources/bbb_1080x608_420_10.h264"), VREADER_OK);
	std::thread pipeline(&VideoReader::startProcessing, &reader);
	std::map<std::string, std::string> parameters = { {"name", "first"}, {"delay", "0"}, {"format", std::to_string(RGB24)}, {"width", "720"}, {"height", "480"}, 
													  {"frames", "10"}, {"dumpName", "bbb_dump.yuv"} };
	//Remove artifacts from previous runs
	remove(parameters["dumpName"].c_str());
	std::thread get(get_cycle, parameters, std::ref(reader));
	get.join();
	reader.endProcessing(HARD);
	pipeline.join();
	//let's compare output

	int width = std::atoi(parameters["width"].c_str());
	int height = std::atoi(parameters["height"].c_str());
	int channels = 3;
	if ((FourCC)std::atoi(parameters["format"].c_str()) == Y800)
		channels = 1;

	int frames = std::atoi(parameters["frames"].c_str());
	{
		std::vector<uint8_t> fileRGBProcessing(width * height * channels * frames);
		std::shared_ptr<FILE> readFile(fopen(parameters["dumpName"].c_str(), "rb"), fclose);
		fread(&fileRGBProcessing[0], fileRGBProcessing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileRGBProcessing[0], width * height * channels * frames), 702350073);
	}
	ASSERT_EQ(remove(parameters["dumpName"].c_str()), 0);
}