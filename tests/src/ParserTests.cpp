#include <gtest/gtest.h>
#include "Parser.h"

int main(int argc, char *argv[])
{
	testing::InitGoogleTest(&argc, argv);
	//	::testing::GTEST_FLAG(filter) = "Pointer.Callstack";
	
	//Disable cout output from library
	std::ofstream   fout("/dev/null");
	std::cout.rdbuf(fout.rdbuf());
	
	return RUN_ALL_TESTS();
	std::getchar(); // keep console window open until Return keystroke
}

TEST(Parser_Init, WrongInputPath) {
	Parser parser;
	ParserParameters parserArgs = { "wrong_path" };
	EXPECT_NE(parser.Init(parserArgs), VREADER_OK);
	parserArgs = { };
	EXPECT_NE(parser.Init(parserArgs), VREADER_OK);
}

TEST(Parser_Init, CorrectInputPath) {
	Parser parser;
	ParserParameters parserArgs = { "rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4" };
	EXPECT_EQ(parser.Init(parserArgs), VREADER_OK);
	parser.Close();
	parserArgs = { "../resources/bbb_1080x608_10.h264" };
	EXPECT_EQ(parser.Init(parserArgs), VREADER_OK);
	EXPECT_EQ(parser.getWidth(), 1080);
	EXPECT_EQ(parser.getHeight(), 608);
	auto codec = parser.getFormatContext()->streams[parser.getVideoIndex()]->codec;
	EXPECT_EQ((int) (codec->framerate.num / codec->framerate.den), 25);
}

TEST(Parser_ReadGet, CheckFrame) {
	Parser parser;
	ParserParameters parserArgs = { "../resources/bbb_1080x608_10.h264" };
	parser.Init(parserArgs);
	//Read SPS/PPS/SEI + IDR frame
	EXPECT_EQ(parser.Read(), VREADER_OK);
	AVPacket parsed;
	EXPECT_EQ(parser.Get(&parsed), VREADER_OK);
	std::ifstream firstFrameFile("../resources/bbb_1080x608_headers_IDR.h264", std::ifstream::binary);
	std::string firstFrame((std::istreambuf_iterator<char>(firstFrameFile)),
		std::istreambuf_iterator<char>());
	firstFrameFile.close();
	EXPECT_EQ(firstFrame.size(), parsed.size);
	EXPECT_EQ(memcmp(parsed.data, firstFrame.c_str(), parsed.size), 0);

	//Read non-IDR frame
	EXPECT_EQ(parser.Read(), VREADER_OK);
	EXPECT_EQ(parser.Get(&parsed), VREADER_OK);
	std::ifstream secondFrameFile("../resources/bbb_1080x608_first_non-IDR.h264", std::ifstream::binary);
	std::string secondFrame((std::istreambuf_iterator<char>(secondFrameFile)),
		std::istreambuf_iterator<char>());
	secondFrameFile.close();
	EXPECT_EQ(secondFrame.size(), parsed.size);
	EXPECT_EQ(memcmp(parsed.data, secondFrame.c_str(), parsed.size), 0);
}

TEST(Parser_ReadGet, BitstreamEnd) {
	Parser parser;
	ParserParameters parserArgs = { "../resources/bbb_1080x608_10.h264" };
	parser.Init(parserArgs);
	AVPacket parsed;
	//Read all frames except last
	for (int i = 0; i < 9; i++) {
		EXPECT_EQ(parser.Read(), VREADER_OK);
		EXPECT_EQ(parser.Get(&parsed), VREADER_OK);
	}
	//Read the last frame
	EXPECT_EQ(parser.Read(), VREADER_OK);
	EXPECT_EQ(parser.Get(&parsed), VREADER_OK);
	//Expect EOF
	EXPECT_EQ(parser.Read(), AVERROR_EOF);
}

TEST(Parser_Bitreader, Convert) {
	std::vector<bool> SPS = { 1, 1, 1, 0, 0, 0, 0, 0 };
	std::vector<bool> PPS = { 0, 0, 0, 1, 0, 0, 0, 0 };
	std::vector<bool> SliceIDR = { 1, 0, 1, 0, 0, 0, 0, 0 };
	BitReader reader;
	EXPECT_EQ(reader.Convert(SPS, BitReader::Type::RAW, BitReader::Base::DEC), 7);
	EXPECT_EQ(reader.Convert(PPS, BitReader::Type::RAW, BitReader::Base::DEC), 8);
	EXPECT_EQ(reader.Convert(SliceIDR, BitReader::Type::RAW, BitReader::Base::DEC), 5);
	//bits obtained by golomb convert procedure, so golomb code is {0, 0, 0, 0, 1, 0, 1, 0, 1}
	std::vector<bool> golomb = { 0, 1, 0, 1 };
	EXPECT_EQ(reader.Convert(golomb, BitReader::Type::GOLOMB, BitReader::Base::DEC), 25);
	EXPECT_EQ(reader.Convert(golomb, BitReader::Type::SGOLOMB, BitReader::Base::DEC), 12);
}

//In next tests need to have initialized internal buffer
class Parser_Bitreader_Internal : public ::testing::Test {
protected:
	void SetUp()
	{
		std::ifstream firstFrameFile("../resources/bbb_1080x608_headers_IDR.h264", std::ifstream::binary);
		file = std::string((std::istreambuf_iterator<char>(firstFrameFile)),
			std::istreambuf_iterator<char>());
		firstFrameFile.close();
		reader = BitReader((uint8_t*) file.c_str(), file.size());
	}
	BitReader reader;
	std::string file;
};

TEST_F(Parser_Bitreader_Internal, ReadBits) {
	reader.Convert(reader.ReadBits(8), BitReader::Type::RAW, BitReader::Base::DEC);
	reader.Convert(reader.ReadBits(8), BitReader::Type::RAW, BitReader::Base::DEC);
	reader.Convert(reader.ReadBits(8), BitReader::Type::RAW, BitReader::Base::DEC);
}

TEST(Parser_Bitreader, ReadBits) {

}

TEST(Parser_Analyze, WithoutSPS) {

}

TEST(Parser_Analyze, WithoutPPS) {

}

