#include <gtest/gtest.h>
#include "VideoProcessor.h"
#include "Parser.h"
#include "Decoder.h"
extern "C" {
	#include "libavutil/crc.h"
}

TEST(VPP_Init, WithoutDumps) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
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
		parser->Init(parserArgs, std::make_shared<Logger>());
		DecoderParameters decoderArgs = { parser, false, 4 };
		int sts = decoder.Init(decoderArgs, std::make_shared<Logger>());
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

TEST_F(VPP_Convert, NV12ToRGB24) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = output->width;
	int height = output->height;
	
	bool normalization = false;
	ColorOptions colorOptions(FourCC::RGB24);
	colorOptions.additionalOptions(Planes::MERGED, normalization);
	FrameParameters frameArgs = { ResizeOptions(width, height), colorOptions };

	float channels = channelsByFourCC(colorOptions.dstFourCC);
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<uint8_t> outputRGBProcessing(frameArgs.resize.width * height * channels);
	EXPECT_EQ(cudaMemcpy(&outputRGBProcessing[0], converted->opaque, channels * width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost), CUDA_SUCCESS);

	//CRC for RGB24 zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 2816643056
	std::string dumpFileName = "DumpFrameRGB.yuv";
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputRGBProcessing[0], width * height * channels), 2816643056);
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<uint8_t*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileRGBProcessing(width * height * channels);
		fread(&fileRGBProcessing[0], fileRGBProcessing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileRGBProcessing[0], width * height * channels), 2816643056);
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(VPP_Convert, NV12ToBGR24) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = output->width;
	int height = output->height;

	ColorOptions colorOptions;
	colorOptions.dstFourCC = FourCC::BGR24;

	ResizeOptions resizeOptions;
	resizeOptions.width = width;
	resizeOptions.height = height;

	FrameParameters frameArgs = { resizeOptions, colorOptions };

	float channels = channelsByFourCC(colorOptions.dstFourCC);
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<uint8_t> outputBGRProcessing(width * height * channels);
	EXPECT_EQ(cudaMemcpy(&outputBGRProcessing[0], converted->opaque, channels * width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost), CUDA_SUCCESS);
	//CRC for BGR24 zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 3797413135
	std::string dumpFileName = "DumpFrameBGR.yuv";
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputBGRProcessing[0], width * height * channels), 3797413135);
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<uint8_t*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileBGRProcessing(width * height * channels);
		fread(&fileBGRProcessing[0], fileBGRProcessing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileBGRProcessing[0], width * height * channels), 3797413135);
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(VPP_Convert, NV12ToY800) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = output->width;
	int height = output->height;

	ColorOptions colorOptions;
	colorOptions.dstFourCC = FourCC::Y800;

	ResizeOptions resizeOptions;
	resizeOptions.width = width;
	resizeOptions.height = height;

	FrameParameters frameArgs = { resizeOptions, colorOptions };

	float channels = channelsByFourCC(colorOptions.dstFourCC);
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<uint8_t> outputY800Processing(width * height * channels);
	EXPECT_EQ(cudaMemcpy(&outputY800Processing[0], converted->opaque, channels * width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost), CUDA_SUCCESS);
	//CRC for Y800 zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 3265466497
	std::string dumpFileName = "DumpFrameY800.yuv";
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputY800Processing[0], width * height * channels), 3265466497);
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<uint8_t*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileY800Processing(width * height * channels);
		fread(&fileY800Processing[0], fileY800Processing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileY800Processing[0], width * height * channels), 3265466497);
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(VPP_Convert, NV12ToRGB24Downscale) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = output->width / 2;
	int height = output->height / 2;
	bool normalization = false;
	ColorOptions colorOptions(FourCC::RGB24);
	colorOptions.additionalOptions(Planes::MERGED, normalization);
	FrameParameters frameArgs = { ResizeOptions(width, height), colorOptions };

	float channels = channelsByFourCC(colorOptions.dstFourCC);
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<uint8_t> outputRGBProcessing(width * height * channels);
	EXPECT_EQ(cudaMemcpy(&outputRGBProcessing[0], converted->opaque, channels * width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost), CUDA_SUCCESS);
	//CRC for resized RGB24 zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 863907011
	std::string dumpFileName = "DumpFrameRGBDownscaled.yuv";
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputRGBProcessing[0], width * height * channels), 863907011);
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<uint8_t*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileRGBProcessing(width * height * channels);
		fread(&fileRGBProcessing[0], fileRGBProcessing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileRGBProcessing[0], width * height * channels), 863907011);
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(VPP_Convert, NV12ToRGB24Upscale) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = output->width * 2;
	int height = output->height * 2;
	bool normalization = false;
	ColorOptions colorOptions(FourCC::RGB24);
	colorOptions.additionalOptions(Planes::MERGED, normalization);
	FrameParameters frameArgs = { ResizeOptions(width, height), colorOptions };

	float channels = channelsByFourCC(colorOptions.dstFourCC);
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<uint8_t> outputRGBProcessing(width * height * channels);
	EXPECT_EQ(cudaMemcpy(&outputRGBProcessing[0], converted->opaque, channels * width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost), CUDA_SUCCESS);
	//CRC for resized RGB24 zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 915070179
	std::string dumpFileName = "DumpFrameRGBUpscaled.yuv";
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputRGBProcessing[0], width * height * channels), 915070179);
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<uint8_t*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileRGBProcessing(width * height * channels);
		fread(&fileRGBProcessing[0], fileRGBProcessing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileRGBProcessing[0], width * height * channels), 915070179);
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(VPP_Convert, NV12ToRGB24Normalization) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = 320;
	int height = 240;
	bool normalization = true;
	ColorOptions colorOptions(FourCC::RGB24);
	colorOptions.additionalOptions(Planes::MERGED, normalization);
	FrameParameters frameArgs = { ResizeOptions(width, height),  colorOptions};

	float channels = channelsByFourCC(colorOptions.dstFourCC);
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<float> outputRGBProcessing(frameArgs.resize.width * height * channels);
	//check correctness of device->host copy
	EXPECT_EQ(cudaMemcpy(&outputRGBProcessing[0], converted->opaque, channels * width * height * sizeof(float), cudaMemcpyDeviceToHost), CUDA_SUCCESS);

	std::string dumpFileName = "RGB24Normalization_320x240.yuv";
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<float*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
		/* The easiest way to check correctness via Python:
		from matplotlib import pyplot as plt
		import numpy as np
		fd = open('RGB24Normalization.yuv', 'rb')
		rows = 240
		cols = 320
		f = np.fromfile(fd, dtype=np.float32,count=rows * cols * 3)
		f = f * 255
		f = f.astype('int32')
		im = f.reshape((rows, cols, 3))
		fd.close()

		plt.imshow(im)
		plt.show()
		*/
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileRGBProcessing(width * height * channels);
		fread(&fileRGBProcessing[0], fileRGBProcessing.size(), 1, readFile.get());
		
		std::string refFileName = std::string("../resources/") + dumpFileName;
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

TEST_F(VPP_Convert, NV12ToBGR24Normalization) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = 320;
	int height = 240;
	bool normalization = true;
	ColorOptions colorOptions(FourCC::BGR24);
	colorOptions.additionalOptions(Planes::MERGED, normalization);
	FrameParameters frameArgs = { ResizeOptions(width, height), colorOptions };

	float channels = channelsByFourCC(colorOptions.dstFourCC);
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<float> outputBGRProcessing(frameArgs.resize.width * height * channels);
	//check correctness of device->host copy
	EXPECT_EQ(cudaMemcpy(&outputBGRProcessing[0], converted->opaque, channels * width * height * sizeof(float), cudaMemcpyDeviceToHost), CUDA_SUCCESS);

	std::string dumpFileName = "BGR24Normalization_320x240.yuv";
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<float*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileBGRProcessing(width * height * channels);
		fread(&fileBGRProcessing[0], fileBGRProcessing.size(), 1, readFile.get());

		std::string refFileName = std::string("../resources/") + dumpFileName;
		std::shared_ptr<FILE> readFileRef(fopen(refFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileBGRProcessingRef(width * height * channels);
		fread(&fileBGRProcessingRef[0], fileBGRProcessingRef.size(), 1, readFileRef.get());

		ASSERT_EQ(fileBGRProcessing.size(), fileBGRProcessingRef.size());
		for (int i = 0; i < fileBGRProcessing.size(); i++) {
			ASSERT_EQ(fileBGRProcessing[i], fileBGRProcessingRef[i]);
		}
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(VPP_Convert, NV12ToYUV800Normalization) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = 320;
	int height = 240;
	bool normalization = true;
	ColorOptions colorOptions(FourCC::Y800);
	colorOptions.additionalOptions(Planes::MERGED, normalization);
	FrameParameters frameArgs = { ResizeOptions(width, height), colorOptions };

	float channels = channelsByFourCC(colorOptions.dstFourCC);
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<float> outputY800Processing(frameArgs.resize.width * height * channels);
	//check correctness of device->host copy
	EXPECT_EQ(cudaMemcpy(&outputY800Processing[0], converted->opaque, channels * width * height * sizeof(float), cudaMemcpyDeviceToHost), CUDA_SUCCESS);

	std::string dumpFileName = "Y800Normalization_320x240.yuv";
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<float*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
		/* The easiest way to check correctness via Python:
		from matplotlib import pyplot as plt
		import numpy as np
		fd = open('Y800Normalization_320x240.yuv', 'rb')
		rows = 240
		cols = 320
		f = np.fromfile(fd, dtype=np.float32,count=rows * cols)
		f = f * 255
		f = f.astype('int32')
		im = f.reshape((rows, cols))
		fd.close()

		plt.imshow(im, cmap='gray')
		plt.show()
		*/
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileY800Processing(width * height * channels);
		fread(&fileY800Processing[0], fileY800Processing.size(), 1, readFile.get());

		std::string refFileName = std::string("../resources/") + dumpFileName;
		std::shared_ptr<FILE> readFileRef(fopen(refFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileY800ProcessingRef(width * height * channels);
		fread(&fileY800ProcessingRef[0], fileY800ProcessingRef.size(), 1, readFileRef.get());

		ASSERT_EQ(fileY800Processing.size(), fileY800ProcessingRef.size());
		for (int i = 0; i < fileY800Processing.size(); i++) {
			ASSERT_EQ(fileY800Processing[i], fileY800ProcessingRef[i]);
		}
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(VPP_Convert, NV12ToRGB24Planar) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = output->width;
	int height = output->height;
	bool normalization = false;
	ColorOptions colorOptions(FourCC::RGB24);
	colorOptions.additionalOptions(Planes::PLANAR, normalization);
	FrameParameters frameArgs = { ResizeOptions(width, height), colorOptions };

	float channels = channelsByFourCC(colorOptions.dstFourCC);
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<uint8_t> outputRGBProcessing(frameArgs.resize.width * height * channels);
	EXPECT_EQ(cudaMemcpy(&outputRGBProcessing[0], converted->opaque, channels * width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost), CUDA_SUCCESS);

	//CRC for RGB24 zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 1381178532
	std::string dumpFileName = "DumpFrameRGB.yuv";
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputRGBProcessing[0], width * height * channels), 1381178532);
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<uint8_t*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileRGBProcessing(width * height * channels);
		fread(&fileRGBProcessing[0], fileRGBProcessing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileRGBProcessing[0], width * height * channels), 1381178532);
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(VPP_Convert, NV12ToBGR24Planar) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = output->width;
	int height = output->height;
	bool normalization = false;
	ColorOptions colorOptions(FourCC::BGR24);
	colorOptions.additionalOptions(Planes::PLANAR, normalization);
	FrameParameters frameArgs = { ResizeOptions(width, height), colorOptions };

	float channels = channelsByFourCC(colorOptions.dstFourCC);
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<uint8_t> outputBGRProcessing(frameArgs.resize.width * height * channels);
	EXPECT_EQ(cudaMemcpy(&outputBGRProcessing[0], converted->opaque, channels * width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost), CUDA_SUCCESS);

	//CRC for BGR24 zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 1193620459
	std::string dumpFileName = "DumpFrameBGR.yuv";
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputBGRProcessing[0], width * height * channels), 1193620459);
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<uint8_t*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileBGRProcessing(width * height * channels);
		fread(&fileBGRProcessing[0], fileBGRProcessing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileBGRProcessing[0], width * height * channels), 1193620459);
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(VPP_Convert, NV12ToUYVY422) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = output->width;
	int height = output->height;
	bool normalization = false;
	ColorOptions colorOptions(FourCC::UYVY);
	colorOptions.additionalOptions(Planes::PLANAR, normalization);
	FrameParameters frameArgs = { ResizeOptions(width, height), colorOptions };

	float channels = channelsByFourCC(colorOptions.dstFourCC);
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<uint8_t> outputUYVYProcessing(frameArgs.resize.width * height * channels);
	EXPECT_EQ(cudaMemcpy(&outputUYVYProcessing[0], converted->opaque, channels * width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost), CUDA_SUCCESS);

	//CRC for UYVY zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 1323730732
	std::string dumpFileName = "DumpFrameUYVY.yuv";
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputUYVYProcessing[0], width * height * channels), 1323730732);
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<uint8_t*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileUYVYProcessing(width * height * channels);
		fread(&fileUYVYProcessing[0], fileUYVYProcessing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileUYVYProcessing[0], width * height * channels), 1323730732);
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(VPP_Convert, NV12ToUYVY422Resized) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = 720;
	int height = 480;
	bool normalization = false;
	ColorOptions colorOptions(FourCC::UYVY);
	colorOptions.additionalOptions(Planes::PLANAR, normalization);
	FrameParameters frameArgs = { ResizeOptions(width, height), colorOptions };

	float channels = channelsByFourCC(colorOptions.dstFourCC);
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<uint8_t> outputUYVYProcessing(frameArgs.resize.width * height * channels);
	EXPECT_EQ(cudaMemcpy(&outputUYVYProcessing[0], converted->opaque, channels * width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost), CUDA_SUCCESS);

	//CRC for UYVY zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 971832452
	std::string dumpFileName = "DumpFrameUYVY.yuv";
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputUYVYProcessing[0], width * height * channels), 971832452);
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<uint8_t*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileUYVYProcessing(width * height * channels);
		fread(&fileUYVYProcessing[0], fileUYVYProcessing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileUYVYProcessing[0], width * height * channels), 971832452);
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(VPP_Convert, NV12ToYUV444) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = output->width;
	int height = output->height;
	bool normalization = false;
	ColorOptions colorOptions(FourCC::YUV444);
	colorOptions.additionalOptions(Planes::PLANAR, normalization);
	FrameParameters frameArgs = { ResizeOptions(width, height), colorOptions };

	float channels = channelsByFourCC(colorOptions.dstFourCC);
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<uint8_t> outputYUV444Processing(frameArgs.resize.width * height * channels);
	EXPECT_EQ(cudaMemcpy(&outputYUV444Processing[0], converted->opaque, channels * width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost), CUDA_SUCCESS);

	//CRC for YUV444 zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 1110927649
	std::string dumpFileName = "DumpFrameYUV444.yuv";
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputYUV444Processing[0], width * height * channels), 1110927649);
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<uint8_t*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileYUV444Processing(width * height * channels);
		fread(&fileYUV444Processing[0], fileYUV444Processing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileYUV444Processing[0], width * height * channels), 1110927649);
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(VPP_Convert, NV12ToYUV444Resized) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = 720;
	int height = 480;
	bool normalization = false;
	ColorOptions colorOptions(FourCC::YUV444);
	colorOptions.additionalOptions(Planes::PLANAR, normalization);
	FrameParameters frameArgs = { ResizeOptions(width, height), colorOptions };

	float channels = channelsByFourCC(colorOptions.dstFourCC);
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<uint8_t> outputYUV444Processing(frameArgs.resize.width * height * channels);
	EXPECT_EQ(cudaMemcpy(&outputYUV444Processing[0], converted->opaque, channels * width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost), CUDA_SUCCESS);

	//CRC for YUV444 zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 886180025
	std::string dumpFileName = "DumpFrameYUV444.yuv";
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputYUV444Processing[0], width * height * channels), 886180025);
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<uint8_t*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileYUV444Processing(width * height * channels);
		fread(&fileYUV444Processing[0], fileYUV444Processing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileYUV444Processing[0], width * height * channels), 886180025);
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(VPP_Convert, NV12ToNV12) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = output->width;
	int height = output->height;
	bool normalization = false;
	ColorOptions colorOptions(FourCC::NV12);
	colorOptions.additionalOptions(Planes::PLANAR, normalization);
	FrameParameters frameArgs = { ResizeOptions(width, height), colorOptions };

	float channels = channelsByFourCC(colorOptions.dstFourCC);
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<uint8_t> outputNV12Processing(frameArgs.resize.width * height * channels);
	EXPECT_EQ(cudaMemcpy(&outputNV12Processing[0], converted->opaque, channels * width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost), CUDA_SUCCESS);

	//CRC for NV12 zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 2957341121
	std::string dumpFileName = "DumpFrameNV12.yuv";
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputNV12Processing[0], width * height * channels), 2957341121);
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<uint8_t*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileNV12Processing(width * height * channels);
		fread(&fileNV12Processing[0], fileNV12Processing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileNV12Processing[0], width * height * channels), 2957341121);
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(VPP_Convert, NV12ToNV12Resized) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = 720;
	int height = 480;
	bool normalization = false;
	ColorOptions colorOptions(FourCC::NV12);
	colorOptions.additionalOptions(Planes::PLANAR, normalization);
	FrameParameters frameArgs = { ResizeOptions(width, height), colorOptions };

	float channels = channelsByFourCC(colorOptions.dstFourCC);
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<uint8_t> outputNV12Processing(frameArgs.resize.width * height * channels);
	EXPECT_EQ(cudaMemcpy(&outputNV12Processing[0], converted->opaque, channels * width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost), CUDA_SUCCESS);

	//CRC for NV12 zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 2944725564
	std::string dumpFileName = "DumpFrameNV12.yuv";
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputNV12Processing[0], width * height * channels), 2944725564);
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<uint8_t*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileNV12Processing(width * height * channels);
		fread(&fileNV12Processing[0], fileNV12Processing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileNV12Processing[0], width * height * channels), 2944725564);
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(VPP_Convert, NV12ToHSV) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	int width = 320;
	int height = 240;
	bool normalization = true;
	ColorOptions colorOptions(FourCC::HSV);
	FrameParameters frameArgs = { ResizeOptions(width, height), colorOptions };

	float channels = channelsByFourCC(colorOptions.dstFourCC);
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<float> outputHSVProcessing(frameArgs.resize.width * height * channels);
	//check correctness of device->host copy
	EXPECT_EQ(cudaMemcpy(&outputHSVProcessing[0], converted->opaque, channels * width * height * sizeof(float), cudaMemcpyDeviceToHost), CUDA_SUCCESS);

	std::string dumpFileName = "HSV_320x240.yuv";
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<float*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
		/* The easiest way to check correctness via Python:
		from matplotlib import pyplot as plt
		from matplotlib.colors import hsv_to_rgb
		import numpy as np
		fd = open('HSV_320x240.yuv', 'rb')
		rows = 320
		cols = 240
		amount = 1
		f = np.fromfile(fd, dtype=np.float32,count=rows * cols * 3)
		im = f.reshape((rows, cols, 3))
		fd.close()

		number = 0
		rgb = hsv_to_rgb(im[number])
		plt.imshow(rgb)
		plt.show()
		*/
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileHSVProcessing(width * height * channels);
		fread(&fileHSVProcessing[0], fileHSVProcessing.size(), 1, readFile.get());

		std::string refFileName = std::string("../resources/") + dumpFileName;
		std::shared_ptr<FILE> readFileRef(fopen(refFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileHSVProcessingRef(width * height * channels);
		fread(&fileHSVProcessingRef[0], fileHSVProcessingRef.size(), 1, readFileRef.get());

		ASSERT_EQ(fileHSVProcessing.size(), fileHSVProcessingRef.size());
		for (int i = 0; i < fileHSVProcessing.size(); i++) {
			ASSERT_EQ(fileHSVProcessing[i], fileHSVProcessingRef[i]);
		}
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}