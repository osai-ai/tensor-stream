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
		system("python setup.py clean --all > nul 2>&1");
#ifdef WIN32
		if (auto ffmpegPath = std::getenv("FFMPEG_PATH")) {
			setupCmdLine += "set PATH=%PATH%;%FFMPEG_PATH%\\bin;";
		}
		else {
			std::cout << "Set FFMPEG_PATH environment variable" << std::endl;
			EXPECT_EQ(0, 1);
		}
		setupCmdLine += " > nul 2>&1 && set CMAKE_GENERATOR_TOOLSET_VERSION=14.11";
		setupCmdLine += " > nul 2>&1 && for /f \"usebackq tokens=*\" %i in (`\"%ProgramFiles(x86)%\\Microsoft Visual Studio\\Installer\\vswhere\.exe\" -version [15^,16^) -products * -latest -property installationPath`) do call \"%i\\VC\\Auxiliary\\Build\\vcvarsall.bat\" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%";
		setupCmdLine += " > nul 2>&1 && python setup.py install";
#endif
		
#ifdef __unix__
		setupCmdLine += "python setup.py install > nul 2>&1";
		system(setupCmdLine.c_str());
#endif;
	}
	static void TearDownTestSuite() {
		chdir("tests/build");
	}
};

std::string Python_Tests::setupCmdLine = "";

void fourCCTest(std::string generalCmdLine, std::string input, int width, int height, int frameNumber, std::string dstFourCC, std::string planes, unsigned long crc) {
	std::stringstream cmdLine;
	std::string dumpFileName = "DumpFrame" + dstFourCC + std::string(".yuv");
	std::string normalizationString = "False";
	float channels = channelsByFourCC(dstFourCC);
#ifdef WIN32
	cmdLine << " > nul 2>&1 && python python_examples/simple.py -fc " << dstFourCC << " -w " << std::to_string(width) << " -h " << std::to_string(height)
		<< " --normalize " << normalizationString << " -n " << frameNumber << " -o " << dumpFileName << " -i " << input << " --planes " << planes;
#elif __unix__
	cmdLine << " > nul 2>&1 && python python_examples/simple.py -fc " << dstFourCC << " -w " << std::to_string(width) << " -h " << std::to_string(height)
		<< " --normalize " << normalizationString << " -n " << frameNumber << " -o " << dumpFileName << " -i " << input << " --planes " << planes;
#endif

	std::string setupCmdLine = generalCmdLine + cmdLine.str();
	setupCmdLine = setupCmdLine + " > nul 2>&1";
	system(setupCmdLine.c_str());
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileNV12Processing(width * height * channels);
		fread(&fileNV12Processing[0], fileNV12Processing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileNV12Processing[0], width * height * channels), crc);
	}
	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

void fourCCTestNormalized(std::string generalCmdLine, std::string refPath, std::string refName, std::string input, int width, int height, int frameNumber, std::string dstFourCC, std::string planes) {
	std::stringstream cmdLine;
	std::string dumpFileName = std::string("DumpFrame") + dstFourCC + std::string(".yuv");
	std::string normalizationString = "True";
	float channels = channelsByFourCC(dstFourCC);
#ifdef WIN32
	cmdLine << " > nul 2>&1 && python python_examples/simple.py -fc " << dstFourCC << " -w " << std::to_string(width) << " -h " << std::to_string(height)
		<< " --normalize " << normalizationString << " -n " << frameNumber << " -o " << dumpFileName << " -i " << input << " --planes " << planes;
#elif __unix__
	cmdLine << " > nul 2>&1 && python python_examples/simple.py -fc " << dstFourCC << " -w " << std::to_string(width) << " -h " << std::to_string(height)
		<< " --normalize " << normalizationString << " -n " << frameNumber << " -o " << dumpFileName << " -i " << input << " --planes " << planes;
#endif

	std::string setupCmdLine = generalCmdLine + cmdLine.str();
	setupCmdLine = setupCmdLine + " > nul 2>&1";
	system(setupCmdLine.c_str());
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
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

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(Python_Tests, FourCC_NV12) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "NV12", "PLANAR", 2957341121);
}

TEST_F(Python_Tests, FourCC_NV12_Downscale) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 720, 480, 1, "NV12", "PLANAR", 2944725564);
}

TEST_F(Python_Tests, FourCC_NV12_Normalize) {
	fourCCTestNormalized(setupCmdLine, "tests/resources/test_references/", "NV12Normalization_320x240.yuv", "tests/resources/bbb_1080x608_420_10.h264", 320, 240, 1, "NV12", "PLANAR");
}

TEST_F(Python_Tests, FourCC_Y800) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "Y800", "PLANAR", 3265466497);
}

TEST_F(Python_Tests, FourCC_Y800_Normalize) {
	fourCCTestNormalized(setupCmdLine, "tests/resources/test_references/", "Y800Normalization_320x240.yuv", "tests/resources/bbb_1080x608_420_10.h264", 320, 240, 1, "Y800", "PLANAR");
}

TEST_F(Python_Tests, FourCC_RGB24) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "RGB24", "MERGED", 2816643056);
}

TEST_F(Python_Tests, FourCC_RGB24_Planar) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "RGB24", "PLANAR", 1381178532);
}

TEST_F(Python_Tests, FourCC_RGB24_Downscale) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080 / 2, 608 / 2, 1, "RGB24", "MERGED", 863907011);
}

TEST_F(Python_Tests, FourCC_RGB24_Normalize) {
	fourCCTestNormalized(setupCmdLine, "tests/resources/test_references/", "RGB24Normalization_320x240.yuv", "tests/resources/bbb_1080x608_420_10.h264", 320, 240, 1, "RGB24", "MERGED");
}

TEST_F(Python_Tests, FourCC_BGR24) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "BGR24", "MERGED", 3797413135);
}

TEST_F(Python_Tests, FourCC_BGR24_Planar) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "BGR24", "PLANAR", 1193620459);
}

TEST_F(Python_Tests, FourCC_BGR24_Downscale) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "BGR24", "MERGED", 3797413135);
}

TEST_F(Python_Tests, FourCC_BGR24_Normalize) {
	fourCCTestNormalized(setupCmdLine, "tests/resources/test_references/", "BGR24Normalization_320x240.yuv", "tests/resources/bbb_1080x608_420_10.h264", 320, 240, 1, "BGR24", "MERGED");
}

TEST_F(Python_Tests, FourCC_UYVY) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "UYVY", "PLANAR", 1323730732);
}

TEST_F(Python_Tests, FourCC_UYVY_Downscale) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 720, 480, 1, "UYVY", "PLANAR", 971832452);
}

TEST_F(Python_Tests, FourCC_UYVY_Normalize) {
	fourCCTestNormalized(setupCmdLine, "tests/resources/test_references/", "UYVYNormalization_320x240.yuv", "tests/resources/bbb_1080x608_420_10.h264", 320, 240, 1, "UYVY", "PLANAR");
}

TEST_F(Python_Tests, FourCC_YUV444) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "YUV444", "PLANAR", 1110927649);
}

TEST_F(Python_Tests, FourCC_YUV444_Downscale) {
	fourCCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 720, 480, 1, "YUV444", "PLANAR", 886180025);
}

TEST_F(Python_Tests, FourCC_YUV444_Normalize) {
	fourCCTestNormalized(setupCmdLine, "tests/resources/test_references/", "YUV444Normalization_320x240.yuv", "tests/resources/bbb_1080x608_420_10.h264", 320, 240, 1, "YUV444", "PLANAR");
}

TEST_F(Python_Tests, FourCC_HSV) {
	fourCCTestNormalized(setupCmdLine, "tests/resources/test_references/", "HSV_320x240.yuv", "tests/resources/bbb_1080x608_420_10.h264", 320, 240, 1, "HSV", "MERGED");
}