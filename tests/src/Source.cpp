#include <gtest/gtest.h>
#include "fstream"

int main(int argc, char *argv[])
{
	testing::InitGoogleTest(&argc, argv);
	//::testing::GTEST_FLAG(filter) = "Wrapper_Init.CorrectParams";

	//Disable cout output from library
	std::ofstream fout("/dev/null");
	std::cout.rdbuf(fout.rdbuf());

	return RUN_ALL_TESTS();
	std::getchar(); // keep console window open until Return keystroke
}