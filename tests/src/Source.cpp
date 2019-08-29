#include <gtest/gtest.h>
#include "fstream"

int main(int argc, char *argv[])
{
	testing::InitGoogleTest(&argc, argv);
	//::testing::GTEST_FLAG(filter) = "Python_Tests.FourCC_RGB24_Bilinear_480x360";

	//Disable cout output from library
	std::cout.setstate(std::ios_base::failbit);
	return RUN_ALL_TESTS();
}