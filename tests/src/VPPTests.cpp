#include <gtest/gtest.h>
#include "VideoProcessor.h"
#include "Parser.h"
#include "Decoder.h"
extern "C" {
	#include "libavutil/crc.h"
}

TEST(VPP_Init, WithoutDumps) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(false), 0);
}

class VPP_Convert : public ::testing::Test {
public:
	std::shared_ptr<Parser> parser;
	Decoder decoder;
	AVPacket parsed;
	std::shared_ptr<AVFrame> output;
protected:
	void SetUp()
	{
		ParserParameters parserArgs = { "../resources/bbb_1080x608_420_10.h264" };
		parser = std::make_shared<Parser>();
		parser->Init(parserArgs);
		//the buffer size is 1 frame, so only the last frame is stored
		DecoderParameters decoderArgs = { parser, false, 4 };
		int sts = decoder.Init(decoderArgs);
		auto parsed = new AVPacket();
		output = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
		sts = parser->Read();
		EXPECT_EQ(sts, VREADER_OK);
		sts = parser->Get(parsed);
		EXPECT_EQ(sts, VREADER_OK);
		int result;
		std::thread get([this, &result]() {
			result = decoder.GetFrame(0, "visualize", output.get());
		});
		sts = decoder.Decode(parsed);
		get.join();
		EXPECT_EQ(sts, VREADER_OK);
		EXPECT_NE(result, VREADER_REPEAT);
	}
};

TEST_F(VPP_Convert, NV12ToRGB) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(false), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = output->width;
	int height = output->height;
	FrameParameters frameArgs = { {width, height } };

	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<float> outputRGBProcessingFloat(frameArgs.resize.width * height * converted->channels);
	EXPECT_EQ(cudaMemcpy(&outputRGBProcessingFloat[0], converted->opaque, converted->channels * width * height * sizeof(float), cudaMemcpyDeviceToHost), CUDA_SUCCESS);
	std::vector<uint8_t> outputRGBProcessing(outputRGBProcessingFloat.begin(), outputRGBProcessingFloat.end());

	//CRC for RGB24 zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 2816643056
	std::string dumpFileName = "DumpFrameRGB.yuv";
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputRGBProcessing[0], width * height * converted->channels), 2816643056);
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<float*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileRGBProcessing(width * height * converted->channels);
		fread(&fileRGBProcessing[0], fileRGBProcessing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileRGBProcessing[0], width * height * converted->channels), 2816643056);
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(VPP_Convert, NV12ToBGR) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(false), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = output->width;
	int height = output->height;

	ColorOptions colorOptions;
	colorOptions.dstFourCC = FourCC::BGR24;

	ResizeOptions resizeOptions;
	resizeOptions.width = width;
	resizeOptions.height = height;

	FrameParameters frameArgs = { resizeOptions, colorOptions };
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<float> outputBGRProcessingFloat(width * height * converted->channels);
	EXPECT_EQ(cudaMemcpy(&outputBGRProcessingFloat[0], converted->opaque, converted->channels * width * height * sizeof(float), cudaMemcpyDeviceToHost), CUDA_SUCCESS);
	std::vector<uint8_t> outputBGRProcessing(outputBGRProcessingFloat.begin(), outputBGRProcessingFloat.end());
	//CRC for BGR24 zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 3797413135
	std::string dumpFileName = "DumpFrameBGR.yuv";
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputBGRProcessing[0], width * height * converted->channels), 3797413135);
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<float*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileBGRProcessing(width * height * converted->channels);
		fread(&fileBGRProcessing[0], fileBGRProcessing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileBGRProcessing[0], width * height * converted->channels), 3797413135);
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(VPP_Convert, NV12ToY800) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(false), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = output->width;
	int height = output->height;

	ColorOptions colorOptions;
	colorOptions.dstFourCC = FourCC::Y800;

	ResizeOptions resizeOptions;
	resizeOptions.width = width;
	resizeOptions.height = height;

	FrameParameters frameArgs = { resizeOptions, colorOptions };
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<float> outputY800ProcessingFloat(width * height * converted->channels);
	EXPECT_EQ(cudaMemcpy(&outputY800ProcessingFloat[0], converted->opaque, converted->channels * width * height * sizeof(float), cudaMemcpyDeviceToHost), CUDA_SUCCESS);
	std::vector<uint8_t> outputY800Processing(outputY800ProcessingFloat.begin(), outputY800ProcessingFloat.end());
	//CRC for Y800 zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 3265466497
	std::string dumpFileName = "DumpFrameY800.yuv";
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputY800Processing[0], width * height * converted->channels), 3265466497);
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<float*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileY800Processing(width * height * converted->channels);
		fread(&fileY800Processing[0], fileY800Processing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileY800Processing[0], width * height * converted->channels), 3265466497);
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(VPP_Convert, NV12ToRGB24Downscale) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(false), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = output->width / 2;
	int height = output->height / 2;
	FrameParameters frameArgs = { width, height };
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<float> outputRGBProcessingFloat(width * height * converted->channels);
	EXPECT_EQ(cudaMemcpy(&outputRGBProcessingFloat[0], converted->opaque, converted->channels * width * height * sizeof(float), cudaMemcpyDeviceToHost), CUDA_SUCCESS);
	std::vector<uint8_t> outputRGBProcessing(outputRGBProcessingFloat.begin(), outputRGBProcessingFloat.end());
	//CRC for resized RGB24 zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 863907011
	std::string dumpFileName = "DumpFrameRGBDownscaled.yuv";
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputRGBProcessing[0], width * height * converted->channels), 863907011);
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<float*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileRGBProcessing(width * height * converted->channels);
		fread(&fileRGBProcessing[0], fileRGBProcessing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileRGBProcessing[0], width * height * converted->channels), 863907011);
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(VPP_Convert, NV12ToRGB24Upscale) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(false), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = output->width * 2;
	int height = output->height * 2;
	FrameParameters frameArgs = { width, height };
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<float> outputRGBProcessingFloat(width * height * converted->channels);
	EXPECT_EQ(cudaMemcpy(&outputRGBProcessingFloat[0], converted->opaque, converted->channels * width * height * sizeof(float), cudaMemcpyDeviceToHost), CUDA_SUCCESS);
	std::vector<uint8_t> outputRGBProcessing(outputRGBProcessingFloat.begin(), outputRGBProcessingFloat.end());
	//CRC for resized RGB24 zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 915070179
	std::string dumpFileName = "DumpFrameRGBUpscaled.yuv";
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputRGBProcessing[0], width * height * converted->channels), 915070179);
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<float*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileRGBProcessing(width * height * converted->channels);
		fread(&fileRGBProcessing[0], fileRGBProcessing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileRGBProcessing[0], width * height * converted->channels), 915070179);
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}