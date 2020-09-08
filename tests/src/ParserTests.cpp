#include <gtest/gtest.h>
#include "Parser.h"

TEST(Parser_Init, FrameStartParsingTime) {
	ParserParameters parserArgs = { "../resources/bbb_1080x608_420_10.h264" };
	std::shared_ptr<Parser> parser = std::make_shared<Parser>();
	parser->Init(parserArgs, std::make_shared<Logger>());
	auto parsed = new AVPacket();
	auto output = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	auto sts = parser->Read();
	EXPECT_EQ(sts, VREADER_OK);
	AVFormatContext* formatContext = reinterpret_cast<AVFormatContext*>(parser->getFormatContext());
	ASSERT_NE(formatContext->opaque, nullptr);
	std::chrono::time_point<std::chrono::system_clock> frameTime = *(std::chrono::time_point<std::chrono::system_clock>*)formatContext->opaque;
	std::chrono::time_point<std::chrono::system_clock> currentTime = std::chrono::system_clock::now();
	ASSERT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - frameTime).count(), 1000); //expect that frame should be read faster than 1 ms
}

TEST(Parser_Init, WrongInputPath) {
	Parser parser;
	ParserParameters parserArgs = { "wrong_path" };
	EXPECT_NE(parser.Init(parserArgs, std::make_shared<Logger>()), VREADER_OK);
	parserArgs = { };
	EXPECT_NE(parser.Init(parserArgs, std::make_shared<Logger>()), VREADER_OK);
}

TEST(Parser_Init, CorrectInputPath) {
	Parser parser;
	ParserParameters parserArgs = { "rtmp://37.228.119.44:1935/vod/big_buck_bunny.mp4", true, true };
	EXPECT_EQ(parser.Init(parserArgs, std::make_shared<Logger>()), VREADER_OK);
	EXPECT_EQ(parser.getWidth(), 1280);
	EXPECT_EQ(parser.getHeight(), 720);
	auto codec = parser.getFormatContext()->streams[parser.getVideoIndex()]->codec;
	EXPECT_EQ((int) (codec->framerate.num / codec->framerate.den), 24);
	parser.Close();
}

TEST(Parser_ReadGet, CheckFrame) {
	Parser parser;
	ParserParameters parserArgs = { "../resources/parser_444/bbb_1080x608_10.h264" };
	parser.Init(parserArgs, std::make_shared<Logger>());
	//Read SPS/PPS/SEI + IDR frame
	EXPECT_EQ(parser.Read(), VREADER_OK);
	AVPacket parsed;
	EXPECT_EQ(parser.Get(&parsed), VREADER_OK);
	std::ifstream firstFrameFile("../resources/parser_444/bbb_1080x608_headers_IDR.h264", std::ifstream::binary);
	std::string firstFrame((std::istreambuf_iterator<char>(firstFrameFile)),
		std::istreambuf_iterator<char>());
	firstFrameFile.close();
	EXPECT_EQ(firstFrame.size(), parsed.size);
	EXPECT_EQ(memcmp(parsed.data, firstFrame.c_str(), parsed.size), 0);

	//Read non-IDR frame
	EXPECT_EQ(parser.Read(), VREADER_OK);
	EXPECT_EQ(parser.Get(&parsed), VREADER_OK);
	std::ifstream secondFrameFile("../resources/parser_444/bbb_1080x608_first_non-IDR.h264", std::ifstream::binary);
	std::string secondFrame((std::istreambuf_iterator<char>(secondFrameFile)),
		std::istreambuf_iterator<char>());
	secondFrameFile.close();
	EXPECT_EQ(secondFrame.size(), parsed.size);
	EXPECT_EQ(memcmp(parsed.data, secondFrame.c_str(), parsed.size), 0);
}

int64_t frameToPTS(AVStream* stream, int frame) {
	//1) frameindex * framerate.den / framerate.num = frame time in seconds
	//2) 1) * framerate.den / framerate.num = frame time in time base units
	double scaleCoeff = (double)(stream->r_frame_rate.den * stream->time_base.den) / (int64_t(stream->r_frame_rate.num) * stream->time_base.num);
	return int64_t(frame) * scaleCoeff;
}

int64_t timebaseDTS(AVStream* stream, int64_t dts) {
	//need convert DTS to ms
	//first of all converting DTS to seconds (DTS is measured in timebase.num / timebase.den seconds, so 1 dts = timebase.num / timebase.den seconds)
	//after converting from seconds to ms by dividing by 1000
	double scaleCoeff = (double)stream->time_base.num / (double)stream->time_base.den * (double)1000;
	auto frameDTS = dts * scaleCoeff;
	return frameDTS;
}

TEST(Parser_ReadGet, CheckDTS) {
	Parser parser;
	ParserParameters parserArgs = { "../resources/parser_444/bbb_1080x608_10.h264" };
	parser.Init(parserArgs, std::make_shared<Logger>());
	//Read SPS/PPS/SEI + IDR frame
	EXPECT_EQ(parser.Read(), VREADER_OK);
	AVPacket parsed;
	EXPECT_EQ(parser.Get(&parsed), VREADER_OK);
	EXPECT_EQ(parsed.dts, AV_NOPTS_VALUE);
	auto videoStream = parser.getFormatContext()->streams[parser.getVideoIndex()];
	auto frameDTS = frameToPTS(videoStream, 2);
	EXPECT_EQ(timebaseDTS(videoStream, frameDTS), 80);
	//
	parserArgs = { "rtmp://37.228.119.44:1935/vod/big_buck_bunny.mp4" };
	parser.Init(parserArgs, std::make_shared<Logger>());
	//Read SPS/PPS/SEI + IDR frame
	EXPECT_EQ(parser.Read(), VREADER_OK);
	EXPECT_EQ(parser.Get(&parsed), VREADER_OK);
	EXPECT_EQ(parsed.dts, 0);
	EXPECT_EQ(parser.Read(), VREADER_OK);
	EXPECT_EQ(parser.Get(&parsed), VREADER_OK);
	EXPECT_EQ(parsed.dts, 41);
	frameDTS = frameToPTS(videoStream, 1);
	EXPECT_EQ(timebaseDTS(videoStream, frameDTS), 40);
}

TEST(Parser_ReadGet, BitstreamEnd) {
	Parser parser;
	ParserParameters parserArgs = { "../resources/parser_444/bbb_1080x608_10.h264" };
	parser.Init(parserArgs, std::make_shared<Logger>());
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

//to convert functions bits are sent as they stored in memory, so 
//vector with bits filled by push_back, so indexes are inverted: 0, 1, 0, 1 = 10 not 5
//because 2^0 * 0 + 2^1 * 1 + 2^2 * 0 + 2^3 * 1
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
		std::ifstream firstFrameFile("../resources/parser_444/bbb_1080x608_headers_IDR.h264", std::ifstream::binary);
		file = std::string((std::istreambuf_iterator<char>(firstFrameFile)),
			std::istreambuf_iterator<char>());
		firstFrameFile.close();
		reader = BitReader((uint8_t*) file.c_str(), file.size());
	}
	BitReader reader;
	std::string file;
};

TEST_F(Parser_Bitreader_Internal, ReadBits) {
	//read RAW bits
	EXPECT_EQ(reader.getByteIndex(), 0);
	EXPECT_EQ(reader.Convert(reader.ReadBits(8), BitReader::Type::RAW, BitReader::Base::DEC), 0);
	EXPECT_EQ(reader.getByteIndex(), 1);
	EXPECT_EQ(reader.Convert(reader.ReadBits(8), BitReader::Type::RAW, BitReader::Base::DEC), 0);
	EXPECT_EQ(reader.Convert(reader.ReadBits(8), BitReader::Type::RAW, BitReader::Base::DEC), 0);
	EXPECT_EQ(reader.Convert(reader.ReadBits(8), BitReader::Type::RAW, BitReader::Base::DEC), 1);
	EXPECT_EQ(reader.getShiftInBits(), 0);
	EXPECT_EQ(reader.getByteIndex(), 4);
	//0, 1, 1, 0, 0, 1, 1, 1 (103) -> 0, 1, 1 (3) ; 0, 0, 1, 1, 1 (7)
	EXPECT_EQ(reader.Convert(reader.ReadBits(3), BitReader::Type::RAW, BitReader::Base::DEC), 3);
	EXPECT_EQ(reader.getByteIndex(), 4);
	EXPECT_EQ(reader.getShiftInBits(), 3);
	EXPECT_EQ(reader.Convert(reader.ReadBits(5), BitReader::Type::RAW, BitReader::Base::DEC), 7);
	EXPECT_EQ(reader.getByteIndex(), 5);
	//1, 1, 1, 1, 0, 1, 0, 1 (244), 0, 0, 0, 0, 0, 0, 0, 0 (0) 
	EXPECT_EQ(reader.Convert(reader.ReadBits(16), BitReader::Type::RAW, BitReader::Base::DEC), 62464);
	//read Golomb
	//0, 0, 0, 1, 1, 1, 1, 1
	EXPECT_EQ(reader.Convert(reader.ReadGolomb(), BitReader::Type::GOLOMB, BitReader::Base::DEC), 14);
	EXPECT_EQ(reader.getShiftInBits(), 7);
	EXPECT_EQ(reader.getByteIndex(), 7);
	EXPECT_EQ(reader.Convert(reader.ReadBits(1), BitReader::Type::RAW, BitReader::Base::DEC), 1);
}

TEST_F(Parser_Bitreader_Internal, SkipBits) {
	EXPECT_EQ(reader.SkipBits(32), true);
	EXPECT_EQ(reader.getByteIndex(), 4);
	EXPECT_EQ(reader.getShiftInBits(), 0);
	EXPECT_EQ(reader.Convert(reader.ReadBits(8), BitReader::Type::RAW, BitReader::Base::DEC), 103);
	EXPECT_EQ(reader.getByteIndex(), 5);
	EXPECT_EQ(reader.SkipBits(3), true);
	EXPECT_EQ(reader.getByteIndex(), 5);
	EXPECT_EQ(reader.getShiftInBits(), 3);
	EXPECT_EQ(reader.Convert(reader.ReadBits(13), BitReader::Type::RAW, BitReader::Base::DEC), 5120);
	EXPECT_EQ(reader.getByteIndex(), 7);
	EXPECT_EQ(reader.getShiftInBits(), 0);
}

TEST_F(Parser_Bitreader_Internal, FindNAL) {
	//SPS = 7
	EXPECT_EQ(reader.Convert(reader.FindNALType(), BitReader::Type::RAW, BitReader::Base::DEC), 7);
	//we read start code 00 00 00 01 = 4 bytes
	EXPECT_EQ(reader.getByteIndex(), 5);
	EXPECT_EQ(reader.getShiftInBits(), 0);
	//PPS = 8
	EXPECT_EQ(reader.Convert(reader.FindNALType(), BitReader::Type::RAW, BitReader::Base::DEC), 8);
	//SEI = 6
	EXPECT_EQ(reader.Convert(reader.FindNALType(), BitReader::Type::RAW, BitReader::Base::DEC), 6);
	//SLICE_IDR = 5
	EXPECT_EQ(reader.Convert(reader.FindNALType(), BitReader::Type::RAW, BitReader::Base::DEC), 5);
	//the bitstream contains only 1 frame with 1 slice, so no more NALu
	//LONG TEST
	EXPECT_EQ(reader.Convert(reader.FindNALType(), BitReader::Type::RAW, BitReader::Base::DEC), 0);
}

//Redirect ffmpeg output to avoid noise in cmd
class Parser_Analyze_Broken : public ::testing::Test {
protected:
	void SetUp()
	{
		av_log_set_callback([](void *ptr, int level, const char *fmt, va_list vargs) {
			return;
		});
	}
};

TEST_F(Parser_Analyze_Broken, WithoutIDR) {
	Parser parser;
	ParserParameters parserArgs = { "../resources/broken_420/Without_IDR.h264" };
	parser.Init(parserArgs, std::make_shared<Logger>());
	AVPacket parsed;
	parser.Read();
	parser.Get(&parsed);
	//expect IDR frame, but observe not-IDR
	EXPECT_EQ(parser.Analyze(&parsed), 2);
}

TEST_F(Parser_Analyze_Broken, WithoutFirstNonIDR) {
	Parser parser;
	ParserParameters parserArgs = { "../resources/broken_420/Without_first_non-IDR.h264" };
	parser.Init(parserArgs, std::make_shared<Logger>());
	AVPacket parsed;
	//Read IDR
	parser.Read();
	parser.Get(&parsed);
	EXPECT_EQ(parser.Analyze(&parsed), 0);
	parser.Read();
	parser.Get(&parsed);
	//expect IDR frame, but observe not-IDR
	EXPECT_EQ(parser.Analyze(&parsed), 2);
}

TEST_F(Parser_Analyze_Broken, LastFrameRepeat) {
	Parser parser;
	//this stream contains gaps_in_frame_num_value_allowed_flag flag so don't check correctnes during first 9 frames (Analyze can't handle this flag and return warning)
	ParserParameters parserArgs = { "../resources/bbb_1080x608_420_10.h264" };
	parser.Init(parserArgs, std::make_shared<Logger>());
	AVPacket parsed;
	for (int i = 0; i < 10; i++) {
		parser.Read();
		parser.Get(&parsed);
		parser.Analyze(&parsed);
	}
	parser.Read();
	parser.Get(&parsed);
	//the same frame_num with the same (wrong) POC
	EXPECT_EQ(parser.Analyze(&parsed), 1);
}