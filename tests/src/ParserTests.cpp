#include <gtest/gtest.h>
#include "Parser.h"

int main(int argc, char *argv[])
{
	testing::InitGoogleTest(&argc, argv);
	//	::testing::GTEST_FLAG(filter) = "Pointer.Callstack";
	return RUN_ALL_TESTS();
	std::getchar(); // keep console window open until Return keystroke
}

TEST(Parser, WrongInputArgs) {
	Parser parser;
	ParserParameters parserArgs = { "wrong_path" };
	EXPECT_EQ(parser.Init(parserArgs), VREADER_OK);
}