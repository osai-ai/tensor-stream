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

void fourCCTest(std::shared_ptr<AVFrame> output, int width, int height, FourCC dstFourCC, Planes planes, unsigned long crc) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);

	ColorOptions colorOptions(dstFourCC);
	colorOptions.normalization = false;
	colorOptions.planesPos = planes;
	FrameParameters frameArgs = { ResizeOptions(width, height), colorOptions };

	float channels = channelsByFourCC(colorOptions.dstFourCC);
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
	std::vector<uint8_t> outputProcessing(frameArgs.resize.width * height * channels);
	EXPECT_EQ(cudaMemcpy(&outputProcessing[0], converted->opaque, channels * width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost), CUDA_SUCCESS);

	std::string dumpFileName = "DumpFrame.yuv";
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputProcessing[0], width * height * channels), crc);
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
	fourCCTest(output, 1080, 608, RGB24, Planes::MERGED, 2816643056);
}

TEST_F(VPP_Convert, NV12ToBGR24) {
	fourCCTest(output, 1080, 608, BGR24, Planes::MERGED, 3797413135);
}

TEST_F(VPP_Convert, NV12ToY800) {
	fourCCTest(output, 1080, 608, Y800, Planes::MERGED, 3265466497);
}

TEST_F(VPP_Convert, NV12ToRGB24Downscale) {
	fourCCTest(output, 1080 / 2, 608 / 2, RGB24, Planes::MERGED, 863907011);
}

TEST_F(VPP_Convert, NV12ToRGB24Upscale) {
	fourCCTest(output, 1080 * 2, 608 * 2, RGB24, Planes::MERGED, 915070179);
}

TEST_F(VPP_Convert, NV12ToRGB24Planar) {
	fourCCTest(output, 1080, 608, RGB24, Planes::PLANAR, 1381178532);
}

TEST_F(VPP_Convert, NV12ToBGR24Planar) {
	fourCCTest(output, 1080, 608, BGR24, Planes::PLANAR, 1193620459);
}

TEST_F(VPP_Convert, NV12ToUYVY422) {
	fourCCTest(output, 1080, 608, UYVY, Planes::MERGED, 1323730732);
}

TEST_F(VPP_Convert, NV12ToUYVY422Resized) {
	fourCCTest(output, 720, 480, UYVY, Planes::MERGED, 971832452);
}

TEST_F(VPP_Convert, NV12ToYUV444) {
	fourCCTest(output, 1080, 608, YUV444, Planes::MERGED, 1110927649);
}

TEST_F(VPP_Convert, NV12ToYUV444Resized) {
	fourCCTest(output, 720, 480, YUV444, Planes::MERGED, 886180025);
}

TEST_F(VPP_Convert, NV12ToNV12) {
	fourCCTest(output, 1080, 608, NV12, Planes::PLANAR, 2957341121);
}

TEST_F(VPP_Convert, NV12ToNV12Resized) {
	fourCCTest(output, 720, 480, NV12, Planes::PLANAR, 2944725564);
}

void fourCCTestNormalized(std::string refPath, std::string refName, std::shared_ptr<AVFrame> output, int width, int height, FourCC dstFourCC, Planes planes) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);
	std::shared_ptr<AVFrame> converted = std::shared_ptr<AVFrame>(av_frame_alloc(), av_frame_unref);
	ColorOptions colorOptions(dstFourCC);
	colorOptions.normalization = true;
	colorOptions.planesPos = planes;
	FrameParameters frameArgs = { ResizeOptions(width, height),  colorOptions };

	float channels = channelsByFourCC(colorOptions.dstFourCC);
	//Convert function unreference output variable
	EXPECT_EQ(VPP.Convert(output.get(), converted.get(), frameArgs, "visualize"), VREADER_OK);
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
	fourCCTestNormalized("../resources/test_references/", "RGB24Normalization_320x240.yuv", output, 320, 240, RGB24, Planes::MERGED);
}

TEST_F(VPP_Convert, NV12ToBGR24Normalization) {
	fourCCTestNormalized("../resources/test_references/", "BGR24Normalization_320x240.yuv", output, 320, 240, BGR24, Planes::MERGED);
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

	fourCCTestNormalized("../resources/test_references/", "Y800Normalization_320x240.yuv", output, 320, 240, Y800, Planes::MERGED);
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

	fourCCTestNormalized("../resources/test_references/", "UYVYNormalization_320x240.yuv", output, 320, 240, UYVY, Planes::MERGED);
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
	fourCCTestNormalized("../resources/test_references/", "YUV444Normalization_320x240.yuv", output, 320, 240, YUV444, Planes::MERGED);
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
	fourCCTestNormalized("../resources/test_references/", "NV12Normalization_320x240.yuv", output, 320, 240, NV12, Planes::MERGED);
}


TEST_F(VPP_Convert, NV12ToHSVResized) {
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
	fourCCTestNormalized("../resources/test_references/", "HSV_320x240.yuv", output, 320, 240, HSV, Planes::MERGED);
}