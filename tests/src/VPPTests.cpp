#include <gtest/gtest.h>
#include "VideoProcessor.h"
#include "Parser.h"
#include "Decoder.h"
#include "WrapperC.h"
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
	Decoder decoderHW;
	Decoder decoderSW;
	AVPacket parsed;
	std::shared_ptr<AVFrame> outputHW, outputSW;
protected:
	void SetUp()
	{
		ParserParameters parserArgs = { "../resources/bbb_1080x608_420_10.h264" };
		parser = std::make_shared<Parser>();
		parser->Init(parserArgs, std::make_shared<Logger>());
		DecoderParameters decoderArgsHW = { parser, false, 4 };
		int sts = decoderHW.Init(decoderArgsHW, std::make_shared<Logger>());
		EXPECT_EQ(sts, VREADER_OK);
		DecoderParameters decoderArgsSW = { parser, false, 4, 0 };
		sts = decoderSW.Init(decoderArgsSW, std::make_shared<Logger>());
		EXPECT_EQ(sts, VREADER_OK);
		auto parsedHW = new AVPacket();
		auto parsedSW = new AVPacket();
		outputHW = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
		outputSW = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
		sts = parser->Read();
		EXPECT_EQ(sts, VREADER_OK);
		sts = parser->Get(parsedHW);
		EXPECT_EQ(sts, VREADER_OK);
		sts = av_packet_ref(parsedSW, parsedHW);
		int resultSW, resultHW;
		std::thread getHW([this, &resultHW]() {
			resultHW = decoderHW.GetFrame(0, "visualize", outputHW.get());
		});
		sts = decoderHW.Decode(parsedHW);
		getHW.join();
		EXPECT_EQ(sts, VREADER_OK);
		EXPECT_NE(resultHW, VREADER_REPEAT);

		std::thread getSW([this, &resultSW]() {
			resultSW = decoderSW.GetFrame(0, "visualize", outputSW.get());
		});
		sts = decoderSW.Decode(parsedSW);
		getSW.join();
		EXPECT_EQ(sts, VREADER_OK);
		EXPECT_NE(resultSW, VREADER_REPEAT);
	}
};

void checkCropCorrectness(std::shared_ptr<AVFrame> output, uint32_t crc, ColorOptions colorOptions = ColorOptions(), ResizeOptions resizeOptions = ResizeOptions(), CropOptions cropOptions = CropOptions()) {
	int inputWidth = output->width;
	int inputHeight = output->height;
	std::vector<uint8_t> outputY(inputWidth * inputHeight);
	std::vector<uint8_t> outputUV(inputWidth * inputHeight / 2);
	ASSERT_EQ(cudaMemcpy2D(&outputY[0], inputWidth, output->data[0], output->linesize[0], inputWidth, inputHeight, cudaMemcpyDeviceToHost), 0);
	ASSERT_EQ(cudaMemcpy2D(&outputUV[0], inputWidth, output->data[1], output->linesize[1], inputWidth, inputHeight / 2, cudaMemcpyDeviceToHost), 0);

	std::shared_ptr<AVFrame> outputDump = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	av_frame_ref(outputDump.get(), output.get());
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> cropConverted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	FrameParameters frameArgs = { resizeOptions, colorOptions, cropOptions };
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(outputDump.get(), cropConverted.get(), frameArgs, "visualize"), VREADER_OK);

	auto channels = channelsByFourCC(colorOptions.dstFourCC);
	auto cropWidth = cropConverted->width;
	auto cropHeight = cropConverted->height;

	std::vector<uint8_t> crop(cropWidth * cropHeight * channels);
	EXPECT_EQ(cudaMemcpy(&crop[0], cropConverted->opaque, channels * cropWidth * cropHeight * sizeof(uint8_t), cudaMemcpyDeviceToHost), CUDA_SUCCESS);
	//check Y
	for (int i = 0; i < cropHeight; i++) {
		for (int j = 0; j < cropWidth; j++) {
			EXPECT_EQ(crop[j + i * cropWidth], outputY[std::get<0>(cropOptions.leftTopCorner) + j + (i + std::get<1>(cropOptions.leftTopCorner)) * inputWidth]);
		}
	}
	//check UV
	int UVRow = std::get<1>(cropOptions.leftTopCorner) / 2;
	int UVCol = std::get<0>(cropOptions.leftTopCorner) % 2 == 0 ? std::get<0>(cropOptions.leftTopCorner) : std::get<0>(cropOptions.leftTopCorner) - 1;
	for (int i = 0; i < cropHeight / 2; i++) {
		for (int j = 0; j < cropWidth; j++) {
			EXPECT_EQ(crop[cropWidth * cropHeight + j + i * cropWidth], 
				outputUV[UVCol + j + (i + UVRow) * inputWidth]);
		}
	}

	std::string dumpFileName = "Cropped_" + std::to_string(cropWidth) + "x" + std::to_string(cropHeight) + ".yuv";
	EXPECT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &crop[0], cropWidth * cropHeight * channels), crc);
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<uint8_t*>(cropConverted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileProcessing(cropWidth * cropHeight * channels);
		fread(&fileProcessing[0], fileProcessing.size(), 1, readFile.get());
		EXPECT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileProcessing[0], cropWidth * cropHeight * channels), crc);
	}

	EXPECT_EQ(remove(dumpFileName.c_str()), 0);
}

void fourCCTest(std::shared_ptr<AVFrame> output, unsigned long crc, ColorOptions colorOptions = ColorOptions(), ResizeOptions resizeOptions = ResizeOptions(), CropOptions cropOptions = CropOptions()) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);

	FrameParameters frameArgs = { resizeOptions, colorOptions, cropOptions };

	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	
	float channels = channelsByFourCC(colorOptions.dstFourCC);
	int width = converted->width;
	int height = converted->height;
	
	std::vector<uint8_t> outputProcessing(width * height * channels);
	EXPECT_EQ(cudaMemcpy(&outputProcessing[0], converted->opaque, channels * width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost), CUDA_SUCCESS);

	std::string dumpFileName = "DumpFrame.yuv";
	EXPECT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputProcessing[0], width * height * channels), crc);
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<uint8_t*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
	{
		std::shared_ptr<FILE> readFile(fopen(dumpFileName.c_str(), "rb"), fclose);
		std::vector<uint8_t> fileProcessing(width * height * channels);
		fread(&fileProcessing[0], fileProcessing.size(), 1, readFile.get());
		ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &fileProcessing[0], width * height * channels), crc);
	}

	ASSERT_EQ(remove(dumpFileName.c_str()), 0);
}

TEST_F(VPP_Convert, NV12ToRGB24) {
	ColorOptions colorOptions(RGB24);
	colorOptions.planesPos = Planes::MERGED;
	ResizeOptions resizeOptions(1080, 608);
	fourCCTest(outputHW, 2225932432, colorOptions, resizeOptions);
	fourCCTest(outputSW, 2225932432, colorOptions, resizeOptions);
}

TEST_F(VPP_Convert, NV12ToRGB24Planar) {
	ColorOptions colorOptions(RGB24);
	colorOptions.planesPos = Planes::PLANAR;
	ResizeOptions resizeOptions(1080, 608);
	fourCCTest(outputHW, 3151499217, colorOptions, resizeOptions);
	fourCCTest(outputSW, 3151499217, colorOptions, resizeOptions);
}

TEST_F(VPP_Convert, NV12ToRGB24Downscale) {
	ColorOptions colorOptions(RGB24);
	colorOptions.planesPos = Planes::MERGED;
	ResizeOptions resizeOptions(1080 / 2, 608 / 2);
	fourCCTest(outputHW, 3545075074, colorOptions, resizeOptions);
	fourCCTest(outputSW, 3545075074, colorOptions, resizeOptions);
}

TEST_F(VPP_Convert, NV12ToRGB24Upscale) {
	ColorOptions colorOptions(RGB24);
	colorOptions.planesPos = Planes::MERGED;
	ResizeOptions resizeOptions(1080 * 2, 608 * 2);
	fourCCTest(outputHW, 97423732, colorOptions, resizeOptions);
	fourCCTest(outputSW, 97423732, colorOptions, resizeOptions);
}

TEST_F(VPP_Convert, NV12ToBGR24) {
	ColorOptions colorOptions(BGR24);
	colorOptions.planesPos = Planes::MERGED;
	ResizeOptions resizeOptions(1080, 608);
	fourCCTest(outputHW, 2467105116, colorOptions, resizeOptions);
	fourCCTest(outputSW, 2467105116, colorOptions, resizeOptions);
}

TEST_F(VPP_Convert, NV12ToBGR24Planar) {
	ColorOptions colorOptions(BGR24);
	colorOptions.planesPos = Planes::PLANAR;
	ResizeOptions resizeOptions(1080, 608);
	fourCCTest(outputHW, 3969775694, colorOptions, resizeOptions);
	fourCCTest(outputSW, 3969775694, colorOptions, resizeOptions);
}

TEST_F(VPP_Convert, NV12ToY800) {
	ColorOptions colorOptions(Y800);
	colorOptions.planesPos = Planes::PLANAR;
	ResizeOptions resizeOptions(1080, 608);
	fourCCTest(outputHW, 3265466497, colorOptions, resizeOptions);
	fourCCTest(outputSW, 3265466497, colorOptions, resizeOptions);
}

TEST_F(VPP_Convert, NV12ToUYVY422) {
	ColorOptions colorOptions(UYVY);
	colorOptions.planesPos = Planes::MERGED;
	ResizeOptions resizeOptions(1080, 608);
	fourCCTest(outputHW, 1323730732, colorOptions, resizeOptions);
	fourCCTest(outputSW, 1323730732, colorOptions, resizeOptions);
}

TEST_F(VPP_Convert, NV12ToUYVY422Downscale) {
	ColorOptions colorOptions(UYVY);
	colorOptions.planesPos = Planes::MERGED;
	ResizeOptions resizeOptions(720, 480);
	fourCCTest(outputHW, 1564587937, colorOptions, resizeOptions);
	fourCCTest(outputSW, 1564587937, colorOptions, resizeOptions);
}

TEST_F(VPP_Convert, NV12ToYUV444) {
	ColorOptions colorOptions(YUV444);
	colorOptions.planesPos = Planes::MERGED;
	ResizeOptions resizeOptions(1080, 608);
	fourCCTest(outputHW, 1110927649, colorOptions, resizeOptions);
	fourCCTest(outputSW, 1110927649, colorOptions, resizeOptions);
}

TEST_F(VPP_Convert, NV12ToYUV444Downscale) {
	ColorOptions colorOptions(YUV444);
	colorOptions.planesPos = Planes::MERGED;
	ResizeOptions resizeOptions(720, 480);
	fourCCTest(outputHW, 449974214, colorOptions, resizeOptions);
	fourCCTest(outputSW, 449974214, colorOptions, resizeOptions);
}

TEST_F(VPP_Convert, NV12ToNV12) {
	ColorOptions colorOptions(NV12);
	colorOptions.planesPos = Planes::PLANAR;
	ResizeOptions resizeOptions(1080, 608);
	fourCCTest(outputHW, 2957341121, colorOptions, resizeOptions);
	fourCCTest(outputSW, 2957341121, colorOptions, resizeOptions);
}

TEST_F(VPP_Convert, NV12ToNV12Downscale) {
	ColorOptions colorOptions(NV12);
	colorOptions.planesPos = Planes::PLANAR;
	ResizeOptions resizeOptions(720, 480);
	fourCCTest(outputHW, 1200915282, colorOptions, resizeOptions);
	fourCCTest(outputSW, 1200915282, colorOptions, resizeOptions);
}

TEST_F(VPP_Convert, NV12ToNV12CropLeftWithoutResize) {
	ColorOptions colorOptions(NV12);
	colorOptions.planesPos = Planes::PLANAR;
	//force native stream width and height
	ResizeOptions resizeOptions(0, 0);
	CropOptions cropOptions({ 0, 0 }, { 320, 240 });
	checkCropCorrectness(outputHW, 3435719157, colorOptions, resizeOptions, cropOptions);
	//don't check SW here because crop without resize function is written only for HW
}

TEST_F(VPP_Convert, NV12ToNV12CropCenterWithoutResize) {
	ColorOptions colorOptions(NV12);
	colorOptions.planesPos = Planes::PLANAR;
	//force native stream width and height
	ResizeOptions resizeOptions(0, 0);
	CropOptions cropOptions({ 320, 240 }, { 720, 480 });
	checkCropCorrectness(outputHW, 1515981907, colorOptions, resizeOptions, cropOptions);
}

TEST_F(VPP_Convert, NV12ToNV12CropCenter2WithoutResize) {
	ColorOptions colorOptions(NV12);
	colorOptions.planesPos = Planes::PLANAR;
	//force native stream width and height
	ResizeOptions resizeOptions(0, 0);
	CropOptions cropOptions({ 400, 240 }, { 720, 480 });
	checkCropCorrectness(outputHW, 655388614, colorOptions, resizeOptions, cropOptions);
}

TEST_F(VPP_Convert, NV12ToNV12CropRightWithoutResize) {
	ColorOptions colorOptions(NV12);
	colorOptions.planesPos = Planes::PLANAR;
	//force native stream width and height
	ResizeOptions resizeOptions(0, 0);
	CropOptions cropOptions({ 640, 360 }, { 1080, 608 });
	checkCropCorrectness(outputHW, 602193072, colorOptions, resizeOptions, cropOptions);
}

TEST_F(VPP_Convert, NV12ToNV12CropResizeUpscaleLeft) {
	ColorOptions colorOptions(NV12);
	colorOptions.planesPos = Planes::PLANAR;
	ResizeOptions resizeOptions(720, 480);
	CropOptions cropOptions({ 0, 0 }, {320, 240});
	fourCCTest(outputHW, 1764198598, colorOptions, resizeOptions, cropOptions);
	fourCCTest(outputSW, 1764198598, colorOptions, resizeOptions, cropOptions);
}

TEST_F(VPP_Convert, NV12ToNV12CropResizeUpscaleCenter) {
	ColorOptions colorOptions(NV12);
	colorOptions.planesPos = Planes::PLANAR;
	ResizeOptions resizeOptions(720, 480);
	CropOptions cropOptions({ 160, 120 }, { 480, 360 });
	fourCCTest(outputHW, 1834204062, colorOptions, resizeOptions, cropOptions);
	fourCCTest(outputSW, 1834204062, colorOptions, resizeOptions, cropOptions);
}

TEST_F(VPP_Convert, NV12ToNV12CropResizeUpscaleRight) {
	ColorOptions colorOptions(NV12);
	colorOptions.planesPos = Planes::PLANAR;
	ResizeOptions resizeOptions(720, 480);
	CropOptions cropOptions({ 400, 240 }, { 720, 480 });
	fourCCTest(outputHW, 1750083777, colorOptions, resizeOptions, cropOptions);
	fourCCTest(outputSW, 1750083777, colorOptions, resizeOptions, cropOptions);
}

TEST_F(VPP_Convert, NV12ToNV12CropResizeDownscaleLeft) {
	ColorOptions colorOptions(NV12);
	colorOptions.planesPos = Planes::PLANAR;
	ResizeOptions resizeOptions(480, 320);
	CropOptions cropOptions({ 0, 0 }, { 720, 480 });
	fourCCTest(outputHW, 3477030875, colorOptions, resizeOptions, cropOptions);
	fourCCTest(outputSW, 3477030875, colorOptions, resizeOptions, cropOptions);
}

TEST_F(VPP_Convert, NV12ToNV12CropResizeDownscaleRight) {
	ColorOptions colorOptions(NV12);
	colorOptions.planesPos = Planes::PLANAR;
	ResizeOptions resizeOptions(480, 320);
	CropOptions cropOptions({ 480, 340 }, { 1080, 608 });
	fourCCTest(outputHW, 2394953726, colorOptions, resizeOptions, cropOptions);
	fourCCTest(outputSW, 2394953726, colorOptions, resizeOptions, cropOptions);
}

void fourCCTestNormalized(std::string refPath, std::string refName, std::shared_ptr<AVFrame> output, ColorOptions colorOptions = ColorOptions(), ResizeOptions resizeOptions = ResizeOptions(), CropOptions cropOptions = CropOptions()) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	FrameParameters frameArgs = { resizeOptions,  colorOptions, cropOptions };

	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);

	float channels = channelsByFourCC(colorOptions.dstFourCC);
	int width = converted->width;
	int height = converted->height;

	std::vector<float> outputRGBProcessing(frameArgs.resize.width * height * channels);
	//check correctness of device->host copy
	EXPECT_EQ(cudaMemcpy(&outputRGBProcessing[0], converted->opaque, channels * width * height * sizeof(float), cudaMemcpyDeviceToHost), CUDA_SUCCESS);

	{
		std::shared_ptr<FILE> writeFile(fopen(refName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(static_cast<float*>(converted->opaque), frameArgs, writeFile), VREADER_OK);
	}
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
TEST_F(VPP_Convert, NV12ToRGB24Normalization) {
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
	ColorOptions colorOptions(RGB24);
	colorOptions.planesPos = Planes::MERGED;
	colorOptions.normalization = true;
	ResizeOptions resizeOptions(320, 240);
	fourCCTestNormalized("../resources/test_references/", "RGB24Normalization_320x240.yuv", outputHW, colorOptions, resizeOptions);
	fourCCTestNormalized("../resources/test_references/", "RGB24Normalization_320x240.yuv", outputSW, colorOptions, resizeOptions);
}

TEST_F(VPP_Convert, NV12ToBGR24Normalization) {
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
	im = im.transpose(2, 0, 1)
	fd.close()

	plt.imshow(im)
	plt.show()
	*/
	ColorOptions colorOptions(BGR24);
	colorOptions.planesPos = Planes::MERGED;
	colorOptions.normalization = true;
	ResizeOptions resizeOptions(320, 240);
	fourCCTestNormalized("../resources/test_references/", "BGR24Normalization_320x240.yuv", outputHW, colorOptions, resizeOptions);
	fourCCTestNormalized("../resources/test_references/", "BGR24Normalization_320x240.yuv", outputSW, colorOptions, resizeOptions);
}

TEST_F(VPP_Convert, NV12ToYUV800Normalization) {
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
	ColorOptions colorOptions(Y800);
	colorOptions.planesPos = Planes::MERGED;
	colorOptions.normalization = true;
	ResizeOptions resizeOptions(320, 240);
	fourCCTestNormalized("../resources/test_references/", "Y800Normalization_320x240.yuv", outputHW, colorOptions, resizeOptions);
	fourCCTestNormalized("../resources/test_references/", "Y800Normalization_320x240.yuv", outputSW, colorOptions, resizeOptions);
}



TEST_F(VPP_Convert, NV12ToUYVY422Normalization) {
	/* The easiest way to check correctness via Python + ffplay not normalized output:
	from matplotlib import pyplot as plt
	import numpy as np
	fd = open('UYVYNormalization_320x240.yuv', 'rb')
	rows = 240
	cols = 320
	channels = 2
	f = np.fromfile(fd, dtype=np.float32,count=channels * rows * cols)
	f = f * 255
	f = f.astype('uint8')
	im = f.reshape((channels, rows, cols))
	fd.close()
	file = open('UYVY_320x240.yuv', 'wb')
	file.write(im)
	file.close()
	*/
	ColorOptions colorOptions(UYVY);
	colorOptions.planesPos = Planes::MERGED;
	colorOptions.normalization = true;
	ResizeOptions resizeOptions(320, 240);
	fourCCTestNormalized("../resources/test_references/", "UYVYNormalization_320x240.yuv", outputHW, colorOptions, resizeOptions);
	fourCCTestNormalized("../resources/test_references/", "UYVYNormalization_320x240.yuv", outputSW, colorOptions, resizeOptions);
}



TEST_F(VPP_Convert, NV12ToYUV444Normalization) {
	/* The easiest way to check correctness via Python + ffplay not normalized output:
	from matplotlib import pyplot as plt
	import numpy as np
	fd = open('YUV444Normalization_320x240.yuv', 'rb')
	rows = 240
	cols = 320
	channels = 3
	f = np.fromfile(fd, dtype=np.float32,count=channels * rows * cols)
	f = f * 255
	f = f.astype('uint8')
	im = f.reshape((channels, rows, cols))
	fd.close()
	file = open('YUV444_320x240.yuv', 'wb')
	file.write(im)
	file.close()
	*/

	ColorOptions colorOptions(YUV444);
	colorOptions.planesPos = Planes::MERGED;
	colorOptions.normalization = true;
	ResizeOptions resizeOptions(320, 240);
	fourCCTestNormalized("../resources/test_references/", "YUV444Normalization_320x240.yuv", outputHW, colorOptions, resizeOptions);
	fourCCTestNormalized("../resources/test_references/", "YUV444Normalization_320x240.yuv", outputSW, colorOptions, resizeOptions);
}

TEST_F(VPP_Convert, NV12ToNV12Normalization) {
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

	ColorOptions colorOptions(NV12);
	colorOptions.planesPos = Planes::MERGED;
	colorOptions.normalization = true;
	ResizeOptions resizeOptions(320, 240);
	fourCCTestNormalized("../resources/test_references/", "NV12Normalization_320x240.yuv", outputHW, colorOptions, resizeOptions);
	fourCCTestNormalized("../resources/test_references/", "NV12Normalization_320x240.yuv", outputSW, colorOptions, resizeOptions);
}


TEST_F(VPP_Convert, NV12ToHSV) {
	/* The easiest way to check correctness via Python:
	from matplotlib import pyplot as plt
	from matplotlib.colors import hsv_to_rgb
	import numpy as np
	fd = open('HSV_320x240.yuv', 'rb')
	rows = 240
	cols = 320
	amount = 1
	f = np.fromfile(fd, dtype=np.float32,count=amount * rows * cols * 3)
	im = f.reshape((amount, rows, cols, 3))
	fd.close()

	number = 0
	rgb = hsv_to_rgb(im[number])
	plt.imshow(rgb)
	plt.show()
	*/
	ColorOptions colorOptions(HSV);
	colorOptions.planesPos = Planes::MERGED;
	ResizeOptions resizeOptions(320, 240);
	fourCCTestNormalized("../resources/test_references/", "HSV_320x240.yuv", outputHW, colorOptions, resizeOptions);
	fourCCTestNormalized("../resources/test_references/", "HSV_320x240.yuv", outputSW, colorOptions, resizeOptions);
}

//monochrome reference and monochrome input (noisy approximation)
double checkPSNR(uint8_t* reference, uint8_t* input, int width, int height) {
	//we have reference and input in RGB format
	std::vector<double> mseChannels(3);
	double normCoeff = (height * width);
	double maxValue = 255;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < 3 * width; j += 3) {
			mseChannels[0] += std::pow(std::abs(reference[j + i * width] - input[j + i * width]), 2);
			mseChannels[1] += std::pow(std::abs(reference[j + i * width + 1] - input[j + i * width + 1]), 2);
			mseChannels[2] += std::pow(std::abs(reference[j + i * width + 2] - input[j + i * width + 2]), 2);
		}
	}

	for (auto &channel : mseChannels) {
		channel /= normCoeff;
	}

	double mse = 0;
	for (auto &channel : mseChannels)
		mse += channel;

	mse /= mseChannels.size();

	double psnr = 10 * std::log10(std::pow(maxValue, 2) / mse);
	return psnr;
}

uint8_t* getFrame(std::string path, FrameParameters frameParams, bool cuda) {
	TensorStream reader;
	//we read only one image, so we have to set blocking mode
	auto sts = reader.initPipeline(path, 5, 0, 5, FrameRateMode::BLOCKING, cuda);
	reader.skipAnalyzeStage();
	std::thread pipeline(&TensorStream::startProcessing, &reader);
	std::tuple<uint8_t*, int> result;
	std::thread get([&reader, &result, &frameParams]() {
		try {
			int frames = 1;
			for (int i = 0; i < frames; i++) {
				result = reader.getFrame<uint8_t>("frame", 0, frameParams);
			}
		}
		catch (std::runtime_error e) {
			return;
		}
	});
	get.join();
	reader.endProcessing();
	pipeline.join();
	return std::get<0>(result);
}

uint8_t* getFrame(uint8_t* frame, int width, int height, FrameParameters frameParams) {
	VideoProcessor vpp;
	EXPECT_EQ(vpp.Init(std::make_shared<Logger>()), 0);
	AVFrame* inputAV = av_frame_alloc();
	AVFrame* outputAV = av_frame_alloc();

	//need to split YUV to AVFrame
	uint8_t* destinationY;
	uint8_t* destinationUV;
	cudaError err;
	err = cudaMalloc(&destinationY, width * height * sizeof(uint8_t));
	err = cudaMalloc(&destinationUV, width * height * sizeof(uint8_t) / 2);
	err = cudaMemcpy(destinationY, frame, width * height * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
	err = cudaMemcpy(destinationUV, frame + width * height * sizeof(uint8_t), width * height * sizeof(uint8_t) / 2, cudaMemcpyDeviceToDevice);
	inputAV->data[0] = destinationY;
	inputAV->data[1] = destinationUV;
	inputAV->linesize[0] = inputAV->linesize[1] = inputAV->width = width;
	inputAV->height = height;
	inputAV->format = AV_PIX_FMT_NV12;
	vpp.Convert(inputAV, outputAV, frameParams, "PSNR");

	cudaFree(destinationY);
	cudaFree(destinationUV);
	av_frame_free(&inputAV);
	return (uint8_t*)outputAV->opaque;
}

double calculatePSNR(std::string imagePath, int dstWidth, int dstHeight, int resizeWidth, int resizeHeight, ResizeType resizeType, FourCC dstFourCC, bool cuda) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);

	ResizeOptions resizeOptions;
	resizeOptions.width = dstWidth;
	resizeOptions.height = dstHeight;
	resizeOptions.type = NEAREST;
	ColorOptions colorOptions;
	colorOptions.dstFourCC = NV12;
	FrameParameters frameArgs = { resizeOptions, colorOptions };

	auto source = getFrame(imagePath, frameArgs, cuda);
	if (source == nullptr)
		return -1;
	/*
	std::string dumpFileName = "Dump_NV12_" + std::to_string(dstWidth) + "x" + std::to_string(dstHeight) + ".yuv";
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(source, frameArgs, writeFile), VREADER_OK);
	}
	*/
	resizeOptions.width = dstWidth;
	resizeOptions.height = dstHeight;
	resizeOptions.type = NEAREST;
	colorOptions.dstFourCC = RGB24;
	frameArgs = { resizeOptions, colorOptions };
	auto converted = getFrame(imagePath, frameArgs, cuda);
	if (converted == nullptr)
		return -1;
	/*
	dumpFileName = "Dump_RGB24_" + std::to_string(dstWidth) + "x" + std::to_string(dstHeight) + ".yuv";
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(converted, frameArgs, writeFile), VREADER_OK);
	}
	*/
	resizeOptions.width = resizeWidth;
	resizeOptions.height = resizeHeight;
	resizeOptions.type = resizeType;
	colorOptions.dstFourCC = NV12;
	frameArgs = { resizeOptions, colorOptions };
	auto scaled = getFrame(source, dstWidth, dstHeight, frameArgs);
	if (scaled == nullptr)
		return -1;
	/*
	dumpFileName = "Dump_NV12_" + std::to_string(resizeWidth) + "x" + std::to_string(resizeHeight) + ".yuv";
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(scaled, frameArgs, writeFile), VREADER_OK);
	}
	*/
	resizeOptions.width = resizeWidth;
	resizeOptions.height = resizeHeight;
	colorOptions.dstFourCC = RGB24;
	frameArgs = { resizeOptions, colorOptions };
	auto scaledRGB = getFrame(scaled, resizeWidth, resizeHeight, frameArgs);
	if (scaledRGB == nullptr)
		return -1;
	/*
	dumpFileName = "Dump_RGB24_" + std::to_string(resizeWidth) + "x" + std::to_string(resizeHeight) + ".yuv";;
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(scaledRGB, frameArgs, writeFile), VREADER_OK);
	}
	*/
	resizeOptions.width = dstWidth;
	resizeOptions.height = dstHeight;
	colorOptions.dstFourCC = dstFourCC;
	frameArgs = { resizeOptions, colorOptions };
	auto rescaled = getFrame(scaled, resizeWidth, resizeHeight, frameArgs);
	if (rescaled == nullptr)
		return -1;
	/*
	dumpFileName = "Dump_RGB24_Rescaled_" + std::to_string(dstWidth) + "x" + std::to_string(dstHeight) + ".yuv";;
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(rescaled, frameArgs, writeFile), VREADER_OK);
	}
	*/
	uint8_t* sourceHost = new uint8_t[(int)(dstWidth * dstHeight * channelsByFourCC(dstFourCC))];
	uint8_t* rescaledHost = new uint8_t[(int)(dstWidth * dstHeight * channelsByFourCC(dstFourCC))];;
	auto err = cudaMemcpy(sourceHost, converted, dstWidth * dstHeight * channelsByFourCC(dstFourCC) * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	err = cudaMemcpy(rescaledHost, rescaled, dstWidth * dstHeight * channelsByFourCC(dstFourCC) * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	double psnr = checkPSNR(sourceHost, rescaledHost, dstWidth, dstHeight);
	delete[] sourceHost;
	delete[] rescaledHost;
	return psnr;
}

TEST_F(VPP_Convert, PSNRSWvsHWComparisonRGBDownscaledBilinear) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 480;
	int resizeHeight = 360;
	std::string imagePath = "../resources/billiard_1920x1080_420_100.h264";
	FourCC dstFourCC = RGB24;
	//----------------
	ResizeType resizeType = NEAREST;
	double psnrNearestSW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 0);
	resizeType = BILINEAR;
	double psnrBilinearSW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 0);
	ASSERT_GT(psnrBilinearSW, psnrNearestSW);
	resizeType = NEAREST;
	double psnrNearestHW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	resizeType = BILINEAR;
	double psnrBilinearHW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	ASSERT_GT(psnrBilinearHW, psnrNearestHW);
	ASSERT_EQ(psnrNearestSW, psnrNearestHW);
	ASSERT_EQ(psnrBilinearSW, psnrBilinearHW);
}

TEST_F(VPP_Convert, PSNRSWvsHWComparisonRGBUpscaledBilinear) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 1920;
	int resizeHeight = 1080;
	std::string imagePath = "../resources/billiard_1920x1080_420_100.h264";
	FourCC dstFourCC = RGB24;
	//----------------
	ResizeType resizeType = NEAREST;
	double psnrNearestSW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 0);
	resizeType = BILINEAR;
	double psnrBilinearSW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 0);
	ASSERT_GT(psnrBilinearSW, psnrNearestSW);
	resizeType = NEAREST;
	double psnrNearestHW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	resizeType = BILINEAR;
	double psnrBilinearHW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	ASSERT_GT(psnrBilinearHW, psnrNearestHW);
	ASSERT_EQ(psnrNearestSW, psnrNearestHW);
	ASSERT_EQ(psnrBilinearSW, psnrBilinearHW);
}

TEST_F(VPP_Convert, PSNRSWvsHWComparisonRGBDownscaledBicubic) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 480;
	int resizeHeight = 360;
	std::string imagePath = "../resources/billiard_1920x1080_420_100.h264";
	FourCC dstFourCC = RGB24;
	//----------------
	ResizeType resizeType = NEAREST;
	double psnrNearestSW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 0);
	resizeType = BICUBIC;
	double psnrBicubicSW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 0);
	ASSERT_GT(psnrBicubicSW, psnrNearestSW);
	resizeType = NEAREST;
	double psnrNearestHW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	resizeType = BICUBIC;
	double psnrBicubicHW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	ASSERT_GT(psnrBicubicHW, psnrNearestHW);
	ASSERT_EQ(psnrNearestSW, psnrNearestHW);
	ASSERT_EQ(psnrBicubicSW, psnrBicubicHW);
}

TEST_F(VPP_Convert, PSNRSWvsHWComparisonRGBUpscaledBicubic) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 1920;
	int resizeHeight = 1080;
	std::string imagePath = "../resources/billiard_1920x1080_420_100.h264";
	FourCC dstFourCC = RGB24;
	//----------------
	ResizeType resizeType = NEAREST;
	double psnrNearestSW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 0);
	resizeType = BICUBIC;
	double psnrBicubicSW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 0);
	ASSERT_GT(psnrBicubicSW, psnrNearestSW);
	resizeType = NEAREST;
	double psnrNearestHW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	resizeType = BICUBIC;
	double psnrBicubicHW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	ASSERT_GT(psnrBicubicHW, psnrNearestHW);
	ASSERT_EQ(psnrNearestSW, psnrNearestHW);
	ASSERT_EQ(psnrBicubicSW, psnrBicubicHW);
}

TEST_F(VPP_Convert, PSNRSWvsHWComparisonRGBDownscaledArea) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 480;
	int resizeHeight = 360;
	std::string imagePath = "../resources/billiard_1920x1080_420_100.h264";
	FourCC dstFourCC = RGB24;
	//----------------
	ResizeType resizeType = NEAREST;
	double psnrNearestSW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 0);
	resizeType = AREA;
	double psnrAreaSW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 0);
	ASSERT_GT(psnrAreaSW, psnrNearestSW);
	resizeType = NEAREST;
	double psnrNearestHW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	resizeType = AREA;
	double psnrAreaHW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	ASSERT_GT(psnrAreaHW, psnrNearestHW);
	ASSERT_EQ(psnrNearestSW, psnrNearestHW);
	ASSERT_EQ(psnrAreaSW, psnrAreaHW);
}

TEST_F(VPP_Convert, PSNRSWvsHWComparisonRGBUpscaledArea) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 1920;
	int resizeHeight = 1080;
	std::string imagePath = "../resources/billiard_1920x1080_420_100.h264";
	FourCC dstFourCC = RGB24;
	//----------------
	ResizeType resizeType = NEAREST;
	double psnrNearestSW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 0);
	resizeType = AREA;
	double psnrAreaSW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 0);
	ASSERT_GT(psnrAreaSW, psnrNearestSW);
	resizeType = NEAREST;
	double psnrNearestHW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	resizeType = AREA;
	double psnrAreaHW = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	ASSERT_GT(psnrAreaHW, psnrNearestHW);
	ASSERT_EQ(psnrNearestSW, psnrNearestHW);
	ASSERT_EQ(psnrAreaSW, psnrAreaHW);
}

TEST_F(VPP_Convert, PSNRTVTemplateRGBDownscaledComparison) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 480;
	int resizeHeight = 360;
	ResizeType resizeType = NEAREST;
	std::string imagePath = "../resources/test_resize/tv_template.jpg";
	FourCC dstFourCC = RGB24;
	//----------------
	double psnrNearest = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	resizeType = BILINEAR;
	double psnrBilinear = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	ASSERT_GT(psnrBilinear, psnrNearest);
}

TEST_F(VPP_Convert, PSNRTVTemplateRGBDownscaledBilinear) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 480;
	int resizeHeight = 360;
	ResizeType resizeType = BILINEAR;
	std::string imagePath = "../resources/test_resize/tv_template.jpg";
	FourCC dstFourCC = RGB24;
	//----------------
	double psnrBilinear = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	EXPECT_NEAR(psnrBilinear, 26.07, 0.01);
}

TEST_F(VPP_Convert, PSNRTVTemplateRGBDownscaledNearest) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 480;
	int resizeHeight = 360;
	ResizeType resizeType = NEAREST;
	std::string imagePath = "../resources/test_resize/tv_template.jpg";
	FourCC dstFourCC = RGB24;
	//----------------
	double psnrNearest = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	EXPECT_NEAR(psnrNearest, 19.14, 0.01);
}

TEST_F(VPP_Convert, PSNRTVTemplateRGBDownscaledBicubic) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 480;
	int resizeHeight = 360;
	ResizeType resizeType = BICUBIC;
	std::string imagePath = "../resources/test_resize/tv_template.jpg";
	FourCC dstFourCC = RGB24;
	//----------------
	double psnrBicubic = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	EXPECT_NEAR(psnrBicubic, 25.80, 0.01);
}

TEST_F(VPP_Convert, PSNRTVTemplateRGBDownscaledArea) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 480;
	int resizeHeight = 360;
	ResizeType resizeType = AREA;
	std::string imagePath = "../resources/test_resize/tv_template.jpg";
	FourCC dstFourCC = RGB24;
	//----------------
	double psnrArea = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	EXPECT_NEAR(psnrArea, 25.89, 0.01);
}

TEST_F(VPP_Convert, PSNRTVTemplateRGBUpscaledBilinear) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 1920;
	int resizeHeight = 1080;
	ResizeType resizeType = BILINEAR;
	std::string imagePath = "../resources/test_resize/tv_template.jpg";
	FourCC dstFourCC = RGB24;
	//----------------
	double psnrBilinear = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	EXPECT_NEAR(psnrBilinear, 39.27, 0.01);
}

TEST_F(VPP_Convert, PSNRTVTemplateRGBUpscaledNearest) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 1920;
	int resizeHeight = 1080;
	ResizeType resizeType = NEAREST;
	std::string imagePath = "../resources/test_resize/tv_template.jpg";
	FourCC dstFourCC = RGB24;
	//----------------
	double psnrNearest = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	EXPECT_NEAR(psnrNearest, 19.14, 0.01);
}

TEST_F(VPP_Convert, PSNRTVTemplateRGBUpscaledBicubic) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 1920;
	int resizeHeight = 1080;
	ResizeType resizeType = BICUBIC;
	std::string imagePath = "../resources/test_resize/tv_template.jpg";
	FourCC dstFourCC = RGB24;
	//----------------
	double psnrBicubic = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	EXPECT_NEAR(psnrBicubic, 30.45, 0.01);
}

TEST_F(VPP_Convert, PSNRTVTemplateRGBUpscaledArea) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 1920;
	int resizeHeight = 1080;
	ResizeType resizeType = AREA;
	std::string imagePath = "../resources/test_resize/tv_template.jpg";
	FourCC dstFourCC = RGB24;
	//----------------
	double psnrArea = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	EXPECT_NEAR(psnrArea, 39.34, 0.01);
}

TEST_F(VPP_Convert, PSNRForestTemplateRGBDownscaledNearest) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 480;
	int resizeHeight = 360;
	ResizeType resizeType = NEAREST;
	std::string imagePath = "../resources/test_resize/forest.jpg";
	FourCC dstFourCC = RGB24;
	//----------------
	double psnrNearest = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	EXPECT_NEAR(psnrNearest, 14.15, 0.01);
}

TEST_F(VPP_Convert, PSNRForestTemplateRGBDownscaledBilinear) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 480;
	int resizeHeight = 360;
	ResizeType resizeType = BILINEAR;
	std::string imagePath = "../resources/test_resize/forest.jpg";
	FourCC dstFourCC = RGB24;
	//----------------
	double psnrNearest = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	EXPECT_NEAR(psnrNearest, 19.51, 0.01);
}

TEST_F(VPP_Convert, PSNRForestTemplateRGBDownscaledBicubic) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 480;
	int resizeHeight = 360;
	ResizeType resizeType = BICUBIC;
	std::string imagePath = "../resources/test_resize/forest.jpg";
	FourCC dstFourCC = RGB24;
	//----------------
	double psnrNearest = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	EXPECT_NEAR(psnrNearest, 20.81, 0.01);
}

TEST_F(VPP_Convert, PSNRForestTemplateRGBDownscaledArea) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 480;
	int resizeHeight = 360;
	ResizeType resizeType = AREA;
	std::string imagePath = "../resources/test_resize/forest.jpg";
	FourCC dstFourCC = RGB24;
	//----------------
	double psnrNearest = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	EXPECT_NEAR(psnrNearest, 19.95, 0.01);
}

TEST_F(VPP_Convert, PSNRForestTemplateRGBUpscaledNearest) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 1920;
	int resizeHeight = 1080;
	ResizeType resizeType = NEAREST;
	std::string imagePath = "../resources/test_resize/forest.jpg";
	FourCC dstFourCC = RGB24;
	//----------------
	double psnrNearest = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	EXPECT_NEAR(psnrNearest, 14.15, 0.01);
}

TEST_F(VPP_Convert, PSNRForestTemplateRGBUpscaledBilinear) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 1920;
	int resizeHeight = 1080;
	ResizeType resizeType = BILINEAR;
	std::string imagePath = "../resources/test_resize/forest.jpg";
	FourCC dstFourCC = RGB24;
	//----------------
	double psnrNearest = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	EXPECT_NEAR(psnrNearest, 28.00, 0.01);
}

TEST_F(VPP_Convert, PSNRForestTemplateRGBUpscaledBicubic) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 1920;
	int resizeHeight = 1080;
	ResizeType resizeType = BICUBIC;
	std::string imagePath = "../resources/test_resize/forest.jpg";
	FourCC dstFourCC = RGB24;
	//----------------
	double psnrNearest = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	EXPECT_NEAR(psnrNearest, 43.08, 0.01);
}

TEST_F(VPP_Convert, PSNRForestTemplateRGBUpscaledArea) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 1920;
	int resizeHeight = 1080;
	ResizeType resizeType = AREA;
	std::string imagePath = "../resources/test_resize/forest.jpg";
	FourCC dstFourCC = RGB24;
	//----------------
	double psnrNearest = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC, 1);
	EXPECT_NEAR(psnrNearest, 30.14, 0.01);
}