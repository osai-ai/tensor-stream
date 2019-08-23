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

TEST_F(VPP_Convert, NV12ToRGB24Planar) {
	fourCCTest(output, 1080, 608, RGB24, Planes::PLANAR, 1381178532);
}

TEST_F(VPP_Convert, NV12ToRGB24Downscale) {
	fourCCTest(output, 1080 / 2, 608 / 2, RGB24, Planes::MERGED, 863907011);
}

TEST_F(VPP_Convert, NV12ToRGB24Upscale) {
	fourCCTest(output, 1080 * 2, 608 * 2, RGB24, Planes::MERGED, 915070179);
}

TEST_F(VPP_Convert, NV12ToBGR24) {
	fourCCTest(output, 1080, 608, BGR24, Planes::MERGED, 3797413135);
}

TEST_F(VPP_Convert, NV12ToBGR24Planar) {
	fourCCTest(output, 1080, 608, BGR24, Planes::PLANAR, 1193620459);
}

TEST_F(VPP_Convert, NV12ToY800) {
	fourCCTest(output, 1080, 608, Y800, Planes::MERGED, 3265466497);
}

TEST_F(VPP_Convert, NV12ToUYVY422) {
	fourCCTest(output, 1080, 608, UYVY, Planes::MERGED, 1323730732);
}

TEST_F(VPP_Convert, NV12ToUYVY422Downscale) {
	fourCCTest(output, 720, 480, UYVY, Planes::MERGED, 971832452);
}

TEST_F(VPP_Convert, NV12ToYUV444) {
	fourCCTest(output, 1080, 608, YUV444, Planes::MERGED, 1110927649);
}

TEST_F(VPP_Convert, NV12ToYUV444Downscale) {
	fourCCTest(output, 720, 480, YUV444, Planes::MERGED, 886180025);
}

TEST_F(VPP_Convert, NV12ToNV12) {
	fourCCTest(output, 1080, 608, NV12, Planes::PLANAR, 2957341121);
}

TEST_F(VPP_Convert, NV12ToNV12Downscale) {
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
	fourCCTestNormalized("../resources/test_references/", "HSV_320x240.yuv", output, 320, 240, HSV, Planes::MERGED);
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

uint8_t* getFrame(std::string path, FrameParameters frameParams) {
	TensorStream reader;
	auto sts = reader.initPipeline(path);
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
	vpp.Convert(inputAV, outputAV, frameParams, "PSNR");

	cudaFree(destinationY);
	cudaFree(destinationUV);
	av_frame_free(&inputAV);
	return (uint8_t*)outputAV->opaque;
}

double calculatePSNR(std::string imagePath, int dstWidth, int dstHeight, int resizeWidth, int resizeHeight, ResizeType resizeType, FourCC dstFourCC) {
	VideoProcessor VPP;
	EXPECT_EQ(VPP.Init(std::make_shared<Logger>()), 0);

	ResizeOptions resizeOptions;
	resizeOptions.width = dstWidth;
	resizeOptions.height = dstHeight;
	resizeOptions.type = NEAREST;
	ColorOptions colorOptions;
	colorOptions.dstFourCC = NV12;
	FrameParameters frameArgs = { resizeOptions, colorOptions };

	auto source = getFrame(imagePath, frameArgs);
	std::string dumpFileName = "Dump_NV12_" + std::to_string(dstWidth) + "x" + std::to_string(dstHeight) + ".yuv";
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(source, frameArgs, writeFile), VREADER_OK);
	}

	resizeOptions.width = dstWidth;
	resizeOptions.height = dstHeight;
	resizeOptions.type = NEAREST;
	colorOptions.dstFourCC = RGB24;
	frameArgs = { resizeOptions, colorOptions };
	auto converted = getFrame(imagePath, frameArgs);

	dumpFileName = "Dump_RGB24_" + std::to_string(dstWidth) + "x" + std::to_string(dstHeight) + ".yuv";
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(converted, frameArgs, writeFile), VREADER_OK);
	}


	resizeOptions.width = resizeWidth;
	resizeOptions.height = resizeHeight;
	resizeOptions.type = resizeType;
	colorOptions.dstFourCC = NV12;
	frameArgs = { resizeOptions, colorOptions };
	auto scaled = getFrame(source, dstWidth, dstHeight, frameArgs);

	dumpFileName = "Dump_NV12_" + std::to_string(resizeWidth) + "x" + std::to_string(resizeHeight) + ".yuv";
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(scaled, frameArgs, writeFile), VREADER_OK);
	}

	resizeOptions.width = resizeWidth;
	resizeOptions.height = resizeHeight;
	colorOptions.dstFourCC = RGB24;
	frameArgs = { resizeOptions, colorOptions };
	auto scaledRGB = getFrame(scaled, resizeWidth, resizeHeight, frameArgs);

	dumpFileName = "Dump_RGB24_" + std::to_string(resizeWidth) + "x" + std::to_string(resizeHeight) + ".yuv";;
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(scaledRGB, frameArgs, writeFile), VREADER_OK);
	}


	resizeOptions.width = dstWidth;
	resizeOptions.height = dstHeight;
	colorOptions.dstFourCC = dstFourCC;
	frameArgs = { resizeOptions, colorOptions };
	auto rescaled = getFrame(scaled, resizeWidth, resizeHeight, frameArgs);

	dumpFileName = "Dump_RGB24_Rescaled_" + std::to_string(dstWidth) + "x" + std::to_string(dstHeight) + ".yuv";;
	{
		std::shared_ptr<FILE> writeFile(fopen(dumpFileName.c_str(), "wb"), fclose);
		EXPECT_EQ(VPP.DumpFrame(rescaled, frameArgs, writeFile), VREADER_OK);
	}

	uint8_t* sourceHost = new uint8_t[(int)(dstWidth * dstHeight * channelsByFourCC(dstFourCC))];
	uint8_t* rescaledHost = new uint8_t[(int)(dstWidth * dstHeight * channelsByFourCC(dstFourCC))];;
	auto err = cudaMemcpy(sourceHost, source, dstWidth * dstHeight * channelsByFourCC(dstFourCC) * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	err = cudaMemcpy(rescaledHost, rescaled, dstWidth * dstHeight * channelsByFourCC(dstFourCC) * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	double psnr = checkPSNR(sourceHost, rescaledHost, dstWidth, dstHeight);
	delete[] sourceHost;
	delete[] rescaledHost;
	return psnr;
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
	double psnrNearest = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC);
	resizeType = BILINEAR;
	double psnrBilinear = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC);
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
	double psnrNearest = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC);
	EXPECT_NEAR(psnrNearest, 23.19, 0.01);
}

void compareNV12() {
	//Test parameters
	int srcWidth = 720;
	int srcHeight = 480;

	int dstWidth = 360;
	int dstHeight = 240;

	float scaleX = (float)srcWidth / (float)dstWidth;
	float scaleY = (float)srcHeight / (float)dstHeight;
	if (scaleX != scaleY)
		return;
	float scale = scaleX;
	std::string referenceNV12 = "C:\\Users\\Home\\Desktop\\Work\\VideoReader_test\\argus-video-reader\\tests\\build\\area_Dump_NV12_720x480.yuv";
	std::string resizeTensorStreamNV12 = //"C:\\Users\\Home\\Desktop\\Work\\VideoReader_test\\argus-video-reader\\tests\\build\\bilinear_Dump_NV12_360x240.yuv";
										 "C:\\Users\\Home\\Desktop\\Work\\VideoReader_test\\argus-video-reader\\tests\\build\\area_Dump_NV12_360x240.yuv";
	//std::string resizeTensorStream = ;

	int startDstX = 0;
	int startDstY = 0;
	int windowSizeDst = 100;

	std::ifstream inputReference(referenceNV12, std::ios::binary);
	// copies all data into buffer
	std::vector<unsigned char> referenceBuffer(std::istreambuf_iterator<char>(inputReference), {});

	std::ifstream inputTensorStream(resizeTensorStreamNV12, std::ios::binary);
	// copies all data into buffer
	std::vector<unsigned char> tensorStreamBuffer(std::istreambuf_iterator<char>(inputTensorStream), {});

	std::shared_ptr<FILE> referenceWrite(fopen("referenceNV12.txt", "wb"), fclose);
	std::shared_ptr<FILE> tensorWrite(fopen("resizedNV12.txt", "wb"), fclose);

	//Print Y plane
	fwrite("Y\n", sizeof(char), 2, tensorWrite.get());
	fflush(tensorWrite.get());
	for (int i = startDstY; i < startDstY + windowSizeDst; i++) {
		for (int j = startDstX; j < startDstX + windowSizeDst; j += 1) {
			int valueTensor = tensorStreamBuffer[j + i * dstWidth];
			int size = strlen(std::to_string(valueTensor).c_str());
			std::string space = " ";
			fwrite(std::to_string(valueTensor).c_str(), sizeof(char), strlen(std::to_string(valueTensor).c_str()), tensorWrite.get());
			fflush(tensorWrite.get());
			while (size++ < 3) {
				fwrite(space.c_str(), sizeof(char), 1, tensorWrite.get());
				fflush(tensorWrite.get());
			}
			fwrite(space.c_str(), sizeof(char), 1, tensorWrite.get());
			fflush(tensorWrite.get());
		}
		fwrite("\n", sizeof(char), 1, tensorWrite.get());
		fflush(tensorWrite.get());
	}

	fwrite("UV\n", sizeof(char), 3, tensorWrite.get());
	fflush(tensorWrite.get());
	for (int i = startDstY / 2; i < startDstY / 2 + windowSizeDst / 2; i++) {
		for (int j = startDstX; j < startDstX + windowSizeDst; j += 1) {
			int valueTensor = tensorStreamBuffer[dstWidth* dstHeight + j + i * dstWidth];
			int size = strlen(std::to_string(valueTensor).c_str());
			std::string space = " ";
			fwrite(std::to_string(valueTensor).c_str(), sizeof(char), size, tensorWrite.get());
			fflush(tensorWrite.get());
			while (size++ < 3) {
				fwrite(space.c_str(), sizeof(char), 1, tensorWrite.get());
				fflush(tensorWrite.get());
			}
			fwrite(space.c_str(), sizeof(char), 1, tensorWrite.get());
			fflush(tensorWrite.get());
		}
		fwrite("\n", sizeof(char), 1, tensorWrite.get());
		fflush(tensorWrite.get());
	}

	//Print Y plane
	fwrite("Y\n", sizeof(char), 2, referenceWrite.get());
	fflush(referenceWrite.get());
	for (int i = startDstY * scale; i < startDstY * scale + windowSizeDst * scale; i++) {
		for (int j = startDstX * scale; j < startDstX * scale + windowSizeDst * scale; j += 1) {
			int valueRef = referenceBuffer[j + i * srcWidth];
			std::string space = " ";
			int size = strlen(std::to_string(valueRef).c_str());
			fwrite(std::to_string(valueRef).c_str(), sizeof(char), size, referenceWrite.get());
			fflush(referenceWrite.get());
			while (size++ < 3) {
				fwrite(space.c_str(), sizeof(char), 1, referenceWrite.get());
				fflush(referenceWrite.get());
			}
			fwrite(space.c_str(), sizeof(char), 1, referenceWrite.get());
			fflush(referenceWrite.get());
		}
		fwrite("\n", sizeof(char), 1, referenceWrite.get());
		fflush(referenceWrite.get());
	}

	fwrite("UV\n", sizeof(char), 3, referenceWrite.get());
	fflush(referenceWrite.get());
	for (int i = startDstY * scale / 2; i < startDstY * scale / 2 + windowSizeDst * scale / 2; i++) {
		for (int j = startDstX * scale; j < startDstX  * scale + windowSizeDst * scale; j += 2) {
			int valueTensorU = referenceBuffer[srcWidth* srcHeight + j + i * srcWidth];
			int valueTensorV = referenceBuffer[srcWidth* srcHeight + j + i * srcWidth + 1];
			int sizeU = strlen(std::to_string(valueTensorU).c_str());
			int sizeV = strlen(std::to_string(valueTensorV).c_str());
			std::string space = " ";
			fwrite(std::to_string(valueTensorU).c_str(), sizeof(char), sizeU, referenceWrite.get());
			fflush(referenceWrite.get());
			while (sizeU++ < 3) {
				fwrite(space.c_str(), sizeof(char), 1, referenceWrite.get());
				fflush(referenceWrite.get());
			}
			fwrite(space.c_str(), sizeof(char), 1, referenceWrite.get());
			fflush(referenceWrite.get());

			fwrite(std::to_string(valueTensorV).c_str(), sizeof(char), sizeV, referenceWrite.get());
			fflush(referenceWrite.get());
			while (sizeV++ < 3) {
				fwrite(space.c_str(), sizeof(char), 1, referenceWrite.get());
				fflush(referenceWrite.get());
			}
			fwrite(space.c_str(), sizeof(char), 1, referenceWrite.get());
			fflush(referenceWrite.get());

		}
		fwrite("\n", sizeof(char), 1, referenceWrite.get());
		fflush(referenceWrite.get());
	}
}

TEST_F(VPP_Convert, Compare) {
	//Test parameters
	int srcWidth = 720;
	int srcHeight = 480;

	int dstWidth = 360;
	int dstHeight = 240;

	float scaleX = (float)srcWidth / (float)dstWidth;
	float scaleY = (float)srcHeight / (float)dstHeight;
	if (scaleX != scaleY)
		return;
	float scale = scaleX;
	std::string reference = "C:\\Users\\Home\\Desktop\\Work\\VideoReader_test\\argus-video-reader\\tests\\build\\Dump_RGB24_360x240.yuv";
	std::string resizeOpenCV = "C:\\Users\\Home\\Desktop\\Work\\VideoReader_test\\argus-video-reader\\tests\\build\\Dump_RGB24_360x240.yuv";
	std::string resizeTensorStream = "C:\\Users\\Home\\Desktop\\Work\\VideoReader_test\\argus-video-reader\\tests\\build\\Dump_RGB24_720x480.yuv";
	int startDstX = 760;
	int startDstY = 0;
	int windowSizeDst = 18;

	std::ifstream inputReference(reference, std::ios::binary);
	// copies all data into buffer
	std::vector<unsigned char> referenceBuffer(std::istreambuf_iterator<char>(inputReference), {});

	std::ifstream inputOpenCV(resizeOpenCV, std::ios::binary);
	// copies all data into buffer
	std::vector<unsigned char> openCVBuffer(std::istreambuf_iterator<char>(inputOpenCV), {});

	std::ifstream inputTensorStream(resizeTensorStream, std::ios::binary);
	// copies all data into buffer
	std::vector<unsigned char> tensorStreamBuffer(std::istreambuf_iterator<char>(inputTensorStream), {});

	int xDiff = 0;
	int yDiff = 0;
	for (int i = 0; i < dstHeight; i++) {
		for (int j = 0; j < dstWidth * 3; j += 3) {
			int valueCV = openCVBuffer[j + i * 3 * dstWidth];
			int valueTensor = tensorStreamBuffer[j + i * 3 * dstWidth];

			if (valueCV != valueTensor) {
				xDiff = j;
				yDiff = i;
				goto end;
			}
		}
	}

	end:

	std::shared_ptr<FILE> openCVWrite(fopen("cv.txt", "wb"), fclose);
	std::shared_ptr<FILE> referenceWrite(fopen("reference.txt", "wb"), fclose);
	std::shared_ptr<FILE> tensorWrite(fopen("tensor.txt", "wb"), fclose);

	for (int i = startDstY; i < startDstY + windowSizeDst; i++) {
		for (int j = startDstX; j < startDstX + windowSizeDst * 3; j+= 3) {
			int valueCV = openCVBuffer[j + i * 3 * dstWidth];
			int valueTensor = tensorStreamBuffer[j + i * 3 * dstWidth];
			int size = strlen(std::to_string(valueCV).c_str());
			std::string space = " ";

			fwrite(std::to_string(valueCV).c_str(), sizeof(char), size, openCVWrite.get());
			fflush(openCVWrite.get());
			while (size++ < 3) {
				fwrite(space.c_str(), sizeof(char), 1, openCVWrite.get());
				fflush(openCVWrite.get());
			}
			
			fwrite(space.c_str(), sizeof(char), 1, openCVWrite.get());
			fflush(openCVWrite.get());

			size = strlen(std::to_string(valueTensor).c_str());
			fwrite(std::to_string(valueTensor).c_str(), sizeof(char), size, tensorWrite.get());
			fflush(tensorWrite.get());
			while (size++ < 3) {
				fwrite(space.c_str(), sizeof(char), 1, tensorWrite.get());
				fflush(tensorWrite.get());
			}
			fwrite(space.c_str(), sizeof(char), 1, tensorWrite.get());
			fflush(tensorWrite.get());
		}
		fwrite("\n", sizeof(char), 1, openCVWrite.get());
		fflush(openCVWrite.get());
		fwrite("\n", sizeof(char), 1, tensorWrite.get());
		fflush(tensorWrite.get());
	}
	for (int i = startDstY * scale; i < startDstY * scale + windowSizeDst * scale; i++) {
		for (int j = startDstX * scale; j < startDstX * scale + windowSizeDst * scale * 3; j+= 3) {
			int valueRef = referenceBuffer[j + i * 3 * srcWidth];
			std::string space = " ";
			int size = strlen(std::to_string(valueRef).c_str());
			fwrite(std::to_string(valueRef).c_str(), sizeof(char), size, referenceWrite.get());
			fflush(referenceWrite.get());
			while (size++ < 3) {
				fwrite(space.c_str(), sizeof(char), 1, referenceWrite.get());
				fflush(referenceWrite.get());
			}
			fwrite(space.c_str(), sizeof(char), 1, referenceWrite.get());
			fflush(referenceWrite.get());
		}
		fwrite("\n", sizeof(char), 1, referenceWrite.get());
		fflush(referenceWrite.get());
	}

	compareNV12();
}

TEST_F(VPP_Convert, PSNRTVTemplateRGBDownscaledNearest) {
	//Test parameters
	int dstWidth = 720;
	int dstHeight = 480;
	int resizeWidth = 360;
	int resizeHeight = 240;
	ResizeType resizeType = AREA;
	//std::string imagePath = "../resources/test_resize/tv_template.jpg";
	std::string imagePath = "../resources/test_resize/forest.jpg";
	FourCC dstFourCC = RGB24;
	//----------------
	double psnrNearest = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC);
	EXPECT_NEAR(psnrNearest, 20.59, 0.01);
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
	double psnrNearest = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC);
	EXPECT_NEAR(psnrNearest, 20.57, 0.01);
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
	double psnrNearest = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC);
	EXPECT_NEAR(psnrNearest, 25.22, 0.01);
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
	//15.438857816749369
	double psnrNearest = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC);
	EXPECT_NEAR(psnrNearest, 15.43, 0.01);
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
	//18.299720207976222
	double psnrNearest = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC);
	EXPECT_NEAR(psnrNearest, 18.29, 0.01);
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
	//15.432385445207258
	double psnrNearest = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC);
	EXPECT_NEAR(psnrNearest, 15.43, 0.01);
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
	//20.312481953913306
	double psnrNearest = calculatePSNR(imagePath, dstWidth, dstHeight, resizeWidth, resizeHeight, resizeType, dstFourCC);
	EXPECT_NEAR(psnrNearest, 20.31, 0.01);
}