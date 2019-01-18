#include <gtest/gtest.h>
#include "Decoder.h"
#include <vector>
#include <future>
extern "C" {
	#include "libavutil/crc.h"
}
#include "cuda.h"
#include <cuda_runtime.h>
//All decoders tests should be executed with YUV420 otherwise no HW acceleration

class Decoder_Init : public ::testing::Test {
protected:
	void SetUp()
	{
		ParserParameters parserArgs = { "../resources/bbb_1080x608_420_10.h264" };
		parser = std::make_shared<Parser>();
		parser->Init(parserArgs);
	}

	std::shared_ptr<Parser> parser;
	AVPacket parsed;
};

TEST_F(Decoder_Init, CorrectInit) {
	Decoder decoder;
	DecoderParameters decoderArgs = { parser, false };
	EXPECT_EQ(decoder.Init(decoderArgs), VREADER_OK);
}

//if index is out of bounds the (index + buffer size) index returned
TEST_F(Decoder_Init, IndexOutOfBuffer) {
	Decoder decoder;
	//the buffer size is 1 frame, so only the last frame is stored
	DecoderParameters decoderArgs = { parser, false, 1 };
	int sts = decoder.Init(decoderArgs);
	sts = parser->Read();
	sts = parser->Get(&parsed);
	auto output = av_frame_alloc();
	std::thread get([&decoder, &output]() {
		decoder.GetFrame(-1, "visualize", output);
	});
	sts = decoder.Decode(&parsed);
	get.join();
	//returned 0 frame because -1 required + 1 buffer size = 0
	std::vector<uint8_t> outputY(output->width * output->height);
	std::vector<uint8_t> outputUV(output->width * output->height / 2);
	ASSERT_EQ(cudaMemcpy2D(&outputY[0], output->width, output->data[0], output->linesize[0], output->width, output->height, cudaMemcpyDeviceToHost), 0);
	ASSERT_EQ(cudaMemcpy2D(&outputUV[0], output->width, output->data[1], output->linesize[1], output->width, output->height / 2, cudaMemcpyDeviceToHost), 0);
	//CRC for zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 3265466497
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputY[0], output->width * output->height), 3265466497);
	//CRC32 - 2183362287
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputUV[0], output->width * output->height / 2), 2183362287);
}

//if index is out of bounds the (index + buffer size) index returned
TEST_F(Decoder_Init, IndexOutOfBuffer_2) {
	Decoder decoder;
	//the buffer size is 1 frame, so only the last frame is stored
	DecoderParameters decoderArgs = { parser, false, 2 };
	int sts = decoder.Init(decoderArgs);
	sts = parser->Read();
	sts = parser->Get(&parsed);
	auto output = av_frame_alloc();
	std::future<int> result = std::async([&decoder, &output]() {
		return decoder.GetFrame(-3, "visualize", output);
		
	});
	//Decoder after frame decoding frees memory of parsed frame
	AVPacket parsedDump;
	av_packet_ref(&parsedDump, &parsed);
	sts = decoder.Decode(&parsed);
	//
	EXPECT_EQ(result.get(), VREADER_REPEAT);
	result = std::async([&decoder, &output]() {
		return decoder.GetFrame(-3, "visualize", output);

	});
	sts = decoder.Decode(&parsedDump);
	//Took the latest frame (shift is equal to 0)
	EXPECT_NE(result.get(), VREADER_OK);
	//returned 0 frame because -1 required + 1 buffer size = 0
	std::vector<uint8_t> outputY(output->width * output->height);
	std::vector<uint8_t> outputUV(output->width * output->height / 2);
	ASSERT_EQ(cudaMemcpy2D(&outputY[0], output->width, output->data[0], output->linesize[0], output->width, output->height, cudaMemcpyDeviceToHost), 0);
	ASSERT_EQ(cudaMemcpy2D(&outputUV[0], output->width, output->data[1], output->linesize[1], output->width, output->height / 2, cudaMemcpyDeviceToHost), 0);
	//CRC for zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 3265466497
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputY[0], output->width * output->height), 3265466497);
	//CRC32 - 2183362287
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputUV[0], output->width * output->height / 2), 2183362287);
}

//if index > 0 (so we want to obtain frame from future) index = 0, warning printed
TEST_F(Decoder_Init, PositiveIndexBuffer) {
	Decoder decoder;
	//the buffer size is 1 frame, so only the last frame is stored
	DecoderParameters decoderArgs = { parser, false, 1 };
	int sts = decoder.Init(decoderArgs);
	sts = parser->Read();
	sts = parser->Get(&parsed);
	auto output = av_frame_alloc();
	std::future<int> result = std::async([&decoder, &output]() {
		return decoder.GetFrame(1, "visualize", output);

	});
	//Decoder after frame decoding frees memory of parsed frame
	sts = decoder.Decode(&parsed);
	EXPECT_NE(result.get(), VREADER_REPEAT);
	std::vector<uint8_t> outputY(output->width * output->height);
	std::vector<uint8_t> outputUV(output->width * output->height / 2);
	ASSERT_EQ(cudaMemcpy2D(&outputY[0], output->width, output->data[0], output->linesize[0], output->width, output->height, cudaMemcpyDeviceToHost), 0);
	ASSERT_EQ(cudaMemcpy2D(&outputUV[0], output->width, output->data[1], output->linesize[1], output->width, output->height / 2, cudaMemcpyDeviceToHost), 0);
	//CRC for zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 3265466497
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputY[0], output->width * output->height), 3265466497);
	//CRC32 - 2183362287
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputUV[0], output->width * output->height / 2), 2183362287);
}