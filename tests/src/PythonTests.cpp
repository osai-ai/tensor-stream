#include <gtest/gtest.h>
#include "VideoProcessor.h"
extern "C" {
	#include "libavutil/crc.h"
}

class Python_Tests : public ::testing::Test {
public:
	static std::string setupCmdLine;

	static void SetUpTestSuite()
	{
		//set path to FFmpeg to environment variables in case of Windows
		chdir("../../");
		system("python setup.py clean --all");
#ifdef WIN32
		if (auto ffmpegPath = std::getenv("FFMPEG_PATH")) {
			std::cout << ffmpegPath << std::endl;
			setupCmdLine += "set PATH=%PATH%;%FFMPEG_PATH%\\bin;";
		}
		else {
			std::cout << "Set FFMPEG_PATH environment variable" << std::endl;
			EXPECT_EQ(0, 1);
		}
		setupCmdLine += " && set CMAKE_GENERATOR_TOOLSET_VERSION=14.11";
		setupCmdLine += " && for /f \"usebackq tokens=*\" %i in (`\"%ProgramFiles(x86)%\\Microsoft Visual Studio\\Installer\\vswhere\.exe\" -version [15^,16^) -products * -latest -property installationPath`) do call \"%i\\VC\\Auxiliary\\Build\\vcvarsall.bat\" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%";
#endif
		setupCmdLine += " && python setup.py install";
#ifdef __unix__
		system(setupCmdLine.c_str());
#endif;
	}
	static void TearDownTestSuite() {
		chdir("tests/build");
	}
};

std::string Python_Tests::setupCmdLine = "";

void fourCCTest(std::string generalCmdLine, std::string input, int width, int height, int frameNumber, std::string dstFourCC, unsigned long crc) {
	std::stringstream cmdLine;
	std::string dumpFileName = "DumpFrame" + dstFourCC + std::string(".yuv");
	std::string normalizationString = "False";
	float channels = channelsByFourCC(dstFourCC);
#ifdef WIN32
	cmdLine << " && python python_examples/simple.py -fc " << dstFourCC << " -w " << std::to_string(width) << " -h " << std::to_string(height)
		<< " --normalize " << normalizationString << " -n " << frameNumber << " -o " << dumpFileName << " -i " << input;
#elif __unix__
	cmdLine << "python python_examples/simple.py -fc " << dstFourCC << " -w " << std::to_string(width) << " -h " << std::to_string(height)
		<< " --normalize " << normalizationString << " -n " << frameNumber << " -o " << dumpFileName << " -i " << input;
#endif

	std::string setupCmdLine = generalCmdLine + cmdLine.str();
	std::cout << setupCmdLine << std::endl;
	system(setupCmdLine.c_str());
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileNV12Processing(width * height * channels);
		fread(&fileNV12Processing[0], fileNV12Processing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileNV12Processing[0], width * height * channels), crc);
	}
	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

void fourCCTestNormalized(std::string generalCmdLine, std::string refPath, std::string refName, std::string input, int width, int height, int frameNumber, std::string dstFourCC, Planes planes) {
	std::stringstream cmdLine;
	std::string dumpFileName = std::string("DumpFrame") + dstFourCC + std::string(".yuv");
	std::string normalizationString = "True";
	float channels = channelsByFourCC(dstFourCC);
#ifdef WIN32
	cmdLine << " && python python_examples/simple.py -fc " << dstFourCC << " -w " << std::to_string(width) << " -h " << std::to_string(height)
		<< " --normalize " << normalizationString << " -n " << frameNumber << " -o " << dumpFileName << " -i " << input;
#elif __unix__
	cmdLine << "python python_examples/simple.py -fc " << dstFourCC << " -w " << std::to_string(width) << " -h " << std::to_string(height)
		<< " --normalize " << normalizationString << " -n " << frameNumber << " -o " << dumpFileName << " -i " << input;
#endif

	std::string setupCmdLine = generalCmdLine + cmdLine.str();
	std::cout << setupCmdLine << std::endl;
	system(setupCmdLine.c_str());
	{
		std::shared_ptr<FILE> readFile(fopen(refName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileRGBProcessing(width * height * channels);
		fread(&fileRGBProcessing[0], fileRGBProcessing.size(), 1, readFile.get());

		std::string refFileName = refPath + refName;
		std::shared_ptr<FILE> readFileRef(fopen(refFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileRGBProcessingRef(width * height * channels);
		fread(&fileRGBProcessingRef[0], fileRGBProcessingRef.size(), 1, readFileRef.get());

		ASSERT_EQ(fileRGBProcessing.size(), fileRGBProcessingRef.size());
		for (int i = 0; i < fileRGBProcessing.size(); i++) {
			ASSERT_EQ(fileRGBProcessing[i], fileRGBProcessingRef[i]);
		}
	}

	ASSERT_EQ(remove(refName.c_str()), 0);
}

TEST_F(Python_Tests, FourCC_NV12) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "NV12", 2957341121);
}

TEST_F(Python_Tests, FourCC_NV12_Downscale) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 720, 480, 1, "NV12", 2944725564);
}

TEST_F(Python_Tests, FourCC_NV12_Normalization) {
	/* The easiest way to check correctness via Python + ffplay not normalized output:
	from matplotlib import pyplot as plt
	import numpy as np
	fd = open('NV12Normalization_320x240.yuv', 'rb')
	rows = 240
	cols = 320
	channels = 1.5
	size = int(rows * cols * channels)
	f = np.fromfile(fd, dtype=np.float32, count=size)
	f = f * 255
	f = f.astype('uint8')
	im = f.reshape((1, int(rows * channels), cols))
	fd.close()
	file = open('NV12_320x240.yuv', 'wb')
	file.write(im)
	file.close()
	*/
	fourCCTestNormalized(setupCmdLine, "../resources/test_references/", "NV12Normalization_320x240.yuv", "tests/resources/bbb_1080x608_420_10.h264", 320, 240, 1, "NV12", Planes::MERGED);
}

TEST_F(Python_Tests, FourCC_Y800) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "Y800", 3265466497);
}

TEST_F(Python_Tests, FourCC_RGB24) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "RGB24", 2816643056);
}

TEST_F(Python_Tests, FourCC_RGB24_Downscale) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080 / 2, 608 / 2, 1, "RGB24", 863907011);
}

TEST_F(Python_Tests, FourCC_BGR24) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "BGR24", 3797413135);
}

TEST_F(Python_Tests, FourCC_UYVY) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "UYVY", 1323730732);
}

TEST_F(Python_Tests, FourCC_UYVY_Downscale) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 720, 480, 1, "UYVY", 971832452);
}

TEST_F(Python_Tests, FourCC_YUV444) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "YUV444", 1110927649);
}

TEST_F(Python_Tests, FourCC_YUV444_Downscale) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 720, 480, 1, "YUV444", 886180025);
}