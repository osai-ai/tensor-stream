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
			ASSERT_EQ(0, 1);
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

void CRCTest(std::string generalCmdLine, std::string input, int width, int height, int frameNumber, std::string dstFourCC, std::string planes, std::string resize, unsigned long crc) {
	std::stringstream cmdLine;
	std::string dumpFileName = std::string("DumpFrame") + dstFourCC + "_" + std::to_string(width) + "x" + std::to_string(height) + "_" + planes;
	std::string normalizationString = "False";
	float channels = channelsByFourCC(dstFourCC);
	cmdLine << " > nul 2>&1 && python python_examples/simple.py -fc " << dstFourCC << " -w " << std::to_string(width) << " -h " << std::to_string(height)
		<< " --normalize " << normalizationString << " -n " << frameNumber << " -o " << dumpFileName << " -i " << input << " --planes " << planes << " --resize_type " << resize;

	std::string setupCmdLine = generalCmdLine + cmdLine.str();
	setupCmdLine = setupCmdLine + " > nul 2>&1";
	system(setupCmdLine.c_str());
	{
		std::shared_ptr<FILE> readFile(fopen(std::string(dumpFileName + ".yuv").c_str(), "rb"), fclose);
		std::vector<uint8_t> fileNV12Processing(width * height * channels);
		fread(&fileNV12Processing[0], fileNV12Processing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileNV12Processing[0], width * height * channels), crc);
	}
	ASSERT_EQ(remove(std::string(dumpFileName + ".yuv").c_str()), 0);
}

void fourCCTestNormalized(std::string generalCmdLine, std::string refPath, std::string refName, std::string input, int width, int height, int frameNumber, std::string dstFourCC, std::string planes) {
	std::stringstream cmdLine;
	std::string dumpFileName = std::string("DumpFrame") + dstFourCC + "_" + std::to_string(width) + "x" + std::to_string(height) + "_normalized_" + planes;
	std::string normalizationString = "True";
	float channels = channelsByFourCC(dstFourCC);
	cmdLine << " > nul 2>&1 && python python_examples/simple.py -fc " << dstFourCC << " -w " << std::to_string(width) << " -h " << std::to_string(height)
		<< " --normalize " << normalizationString << " -n " << frameNumber << " -o " << dumpFileName << " -i " << input << " --planes " << planes;

	std::string setupCmdLine = generalCmdLine + cmdLine.str();
	setupCmdLine = setupCmdLine + " > nul 2>&1";
	system(setupCmdLine.c_str());
	{
		std::shared_ptr<FILE> readFile(fopen(std::string(dumpFileName + ".yuv").c_str(), "rb"), fclose);
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

	ASSERT_EQ(remove(std::string(dumpFileName + ".yuv").c_str()), 0);
}

//FourCC tests
TEST_F(Python_Tests, FourCC_NV12) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "NV12", "PLANAR", "NEAREST", 2957341121);
}

TEST_F(Python_Tests, FourCC_NV12_Downscale) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 720, 480, 1, "NV12", "PLANAR", "NEAREST", 1200915282);
}

TEST_F(Python_Tests, FourCC_NV12_Normalize) {
	fourCCTestNormalized(setupCmdLine, "tests/resources/test_references/", "NV12Normalization_320x240.yuv", "tests/resources/bbb_1080x608_420_10.h264", 320, 240, 1, "NV12", "PLANAR");
}

TEST_F(Python_Tests, FourCC_Y800) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "Y800", "PLANAR", "NEAREST", 3265466497);
}

TEST_F(Python_Tests, FourCC_Y800_Normalize) {
	fourCCTestNormalized(setupCmdLine, "tests/resources/test_references/", "Y800Normalization_320x240.yuv", "tests/resources/bbb_1080x608_420_10.h264", 320, 240, 1, "Y800", "PLANAR");
}

TEST_F(Python_Tests, FourCC_RGB24) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "RGB24", "MERGED", "NEAREST", 2225932432);
}

TEST_F(Python_Tests, FourCC_RGB24_Planar) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "RGB24", "PLANAR", "NEAREST", 3151499217);
}

TEST_F(Python_Tests, FourCC_RGB24_Downscale) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080 / 2, 608 / 2, 1, "RGB24", "MERGED", "NEAREST", 3545075074);
}

TEST_F(Python_Tests, FourCC_RGB24_Normalize) {
	fourCCTestNormalized(setupCmdLine, "tests/resources/test_references/", "RGB24Normalization_320x240.yuv", "tests/resources/bbb_1080x608_420_10.h264", 320, 240, 1, "RGB24", "MERGED");
}

TEST_F(Python_Tests, FourCC_BGR24) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "BGR24", "MERGED", "NEAREST", 2467105116);
}

TEST_F(Python_Tests, FourCC_BGR24_Planar) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "BGR24", "PLANAR", "NEAREST", 3969775694);
}

TEST_F(Python_Tests, FourCC_BGR24_Downscale) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080 / 2, 608 / 2, 1, "BGR24", "MERGED", "NEAREST", 201454032);
}

TEST_F(Python_Tests, FourCC_BGR24_Normalize) {
	fourCCTestNormalized(setupCmdLine, "tests/resources/test_references/", "BGR24Normalization_320x240.yuv", "tests/resources/bbb_1080x608_420_10.h264", 320, 240, 1, "BGR24", "MERGED");
}

TEST_F(Python_Tests, FourCC_UYVY) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "UYVY", "PLANAR", "NEAREST", 1323730732);
}

TEST_F(Python_Tests, FourCC_UYVY_Downscale) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 720, 480, 1, "UYVY", "PLANAR", "NEAREST", 1564587937);
}

TEST_F(Python_Tests, FourCC_UYVY_Normalize) {
	fourCCTestNormalized(setupCmdLine, "tests/resources/test_references/", "UYVYNormalization_320x240.yuv", "tests/resources/bbb_1080x608_420_10.h264", 320, 240, 1, "UYVY", "PLANAR");
}

TEST_F(Python_Tests, FourCC_YUV444) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1080, 608, 1, "YUV444", "PLANAR", "NEAREST", 1110927649);
}

TEST_F(Python_Tests, FourCC_YUV444_Downscale) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 720, 480, 1, "YUV444", "PLANAR", "NEAREST", 449974214);
}

TEST_F(Python_Tests, FourCC_YUV444_Normalize) {
	fourCCTestNormalized(setupCmdLine, "tests/resources/test_references/", "YUV444Normalization_320x240.yuv", "tests/resources/bbb_1080x608_420_10.h264", 320, 240, 1, "YUV444", "PLANAR");
}

TEST_F(Python_Tests, FourCC_HSV) {
	fourCCTestNormalized(setupCmdLine, "tests/resources/test_references/", "HSV_320x240.yuv", "tests/resources/bbb_1080x608_420_10.h264", 320, 240, 1, "HSV", "MERGED");
}

//Resize tests
TEST_F(Python_Tests, FourCC_RGB24_Nearest_480x360) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 480, 360, 1, "RGB24", "MERGED", "NEAREST", 3234932936);
}

TEST_F(Python_Tests, FourCC_RGB24_Nearest_540x304) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 540, 304, 1, "RGB24", "MERGED", "NEAREST", 3545075074);
}

TEST_F(Python_Tests, FourCC_RGB24_Nearest_1920x1080) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1920, 1080, 1, "RGB24", "MERGED", "NEAREST", 867059050);
}

TEST_F(Python_Tests, FourCC_RGB24_Bilinear_480x360) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 480, 360, 1, "RGB24", "MERGED", "BILINEAR", 1166179972);
}

TEST_F(Python_Tests, FourCC_RGB24_Bilinear_540x304) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 540, 304, 1, "RGB24", "MERGED", "BILINEAR", 2257004891);
}

TEST_F(Python_Tests, FourCC_RGB24_Bilinear_1920x1080) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1920, 1080, 1, "RGB24", "MERGED", "BILINEAR", 930427804);
}

TEST_F(Python_Tests, FourCC_RGB24_Bicubic_480x360) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 480, 360, 1, "RGB24", "MERGED", "BICUBIC", 1772194314);
}

TEST_F(Python_Tests, FourCC_RGB24_Bicubic_540x304) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 540, 304, 1, "RGB24", "MERGED", "BICUBIC", 240232532);
}

TEST_F(Python_Tests, FourCC_RGB24_Bicubic_1920x1080) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1920, 1080, 1, "RGB24", "MERGED", "BICUBIC", 3759932769);
}

TEST_F(Python_Tests, FourCC_RGB24_Area_480x360) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 480, 360, 1, "RGB24", "MERGED", "AREA", 3175240744);
}

TEST_F(Python_Tests, FourCC_RGB24_Area_540x304) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 540, 304, 1, "RGB24", "MERGED", "AREA", 2257004891);
}

TEST_F(Python_Tests, FourCC_RGB24_Area_1920x1080) {
	CRCTest(setupCmdLine, "tests/resources/bbb_1080x608_420_10.h264", 1920, 1080, 1, "RGB24", "MERGED", "AREA", 2026855);
}
