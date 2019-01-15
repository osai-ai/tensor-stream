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

TEST(Parser, WrongInputPath) {
	Parser parser;
	ParserParameters parserArgs = { "wrong_path" };
	EXPECT_NE(parser.Init(parserArgs), VREADER_OK);
	parserArgs = { };
	EXPECT_NE(parser.Init(parserArgs), VREADER_OK);
}

TEST(Parser, CorrectInputPath) {
	Parser parser;
	ParserParameters parserArgs = { "rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4" };
	EXPECT_EQ(parser.Init(parserArgs), VREADER_OK);
	parser.Close();
	parserArgs = { "../resources/bbb_1080x608_10.h264" };
	EXPECT_EQ(parser.Init(parserArgs), VREADER_OK);
}