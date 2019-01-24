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
		//ParserParameters parserArgs = { "rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4" };
		parser = std::make_shared<Parser>();
		parser->Init(parserArgs);
	}


public:
	std::shared_ptr<Parser> parser;
	AVPacket parsed;
};

void processing(std::shared_ptr<Parser>& parser, Decoder& decoder, AVPacket& parsed, int number) {
	for (int i = 0; i < number; i++) {
		int sts = VREADER_OK;
		sts = parser->Read();
		sts = parser->Get(&parsed);
		sts = decoder.Decode(&parsed);
	}
}

TEST_F(Decoder_Init, CorrectInit) {
	Decoder decoder;
	DecoderParameters decoderArgs = { parser, false };
	EXPECT_EQ(decoder.Init(decoderArgs), VREADER_OK);
}

//if index is out of bounds the (index + buffer size) index returned
TEST_F(Decoder_Init, IndexOutOfBuffer) {
	Decoder decoder;
	//the buffer size is 1 frame, so only the last frame is stored
	DecoderParameters decoderArgs = { parser, false, 2 };
	int sts = decoder.Init(decoderArgs);
	std::thread startProcessing(processing, std::ref(parser), std::ref(decoder), std::ref(parsed), 2);
	auto output = av_frame_alloc();
	std::thread get([&decoder, &output]() {
		int sts = VREADER_REPEAT;
		while (sts == VREADER_REPEAT)
			sts = decoder.GetFrame(-1, "visualize", output);
	});
	get.join();
	startProcessing.join();
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
	//wait some time to gurantee that GetFrame will be executed before Decode
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
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

TEST_F(Decoder_Init, CheckHWPixelFormat) {
	Decoder decoder;
	//the buffer size is 1 frame, so only the last frame is stored
	DecoderParameters decoderArgs = { parser, false, 1 };
	int sts = decoder.Init(decoderArgs);
	sts = parser->Read();
	sts = parser->Get(&parsed);
	auto output = av_frame_alloc();
	std::future<int> result = std::async([&decoder, &output]() {
		return decoder.GetFrame(0, "visualize", output);

	});
	//wait some time to gurantee that GetFrame will be executed before Decode
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	AVCodecContext* context = decoder.getDecoderContext();
	ASSERT_EQ(context->pix_fmt, AV_PIX_FMT_YUV420P);
	//Decoder after frame decoding frees memory of parsed frame
	sts = decoder.Decode(&parsed);
	ASSERT_EQ(context->pix_fmt, AV_PIX_FMT_CUDA);
	EXPECT_NE(result.get(), VREADER_REPEAT);

}

TEST(Decoder_Init_YUV444, HWUsupportedPixelFormat) {
	av_log_set_callback([](void *ptr, int level, const char *fmt, va_list vargs) {
		return;
	});
	std::shared_ptr<Parser> parser;
	AVPacket parsed;
	ParserParameters parserArgs = { "../resources/parser_444/bbb_1080x608_10.h264" };
	parser = std::make_shared<Parser>();
	parser->Init(parserArgs);
	Decoder decoder;
	//the buffer size is 1 frame, so only the last frame is stored
	DecoderParameters decoderArgs = { parser, false, 1 };
	int sts = decoder.Init(decoderArgs);
	sts = parser->Read();
	sts = parser->Get(&parsed);
	auto output = av_frame_alloc();
	std::future<int> result = std::async([&decoder, &output]() {
		return decoder.GetFrame(0, "visualize", output);

	});
	//wait some time to gurantee that GetFrame will be executed before Decode
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	AVCodecContext* context = decoder.getDecoderContext();
	ASSERT_EQ(context->pix_fmt, AV_PIX_FMT_YUV444P);
	//Decoder after frame decoding frees memory of parsed frame
	sts = decoder.Decode(&parsed);
	ASSERT_EQ(context->pix_fmt, AV_PIX_FMT_YUV444P);
	EXPECT_NE(result.get(), VREADER_REPEAT);

}

//Notice that we have buffer with decoded surfaces(!) which holds references to decoder surfaces from DPB,
//so if DPB is equal to x but our buffer size is greater than x so we will get the error "No decoder surfaces left"
//so need either change decoder buffer in DecoderParameters or change DPB
TEST_F(Decoder_Init, DPBBiggerBuffer) {
	Decoder decoder;
	//the buffer size is 1 frame, so only the last frame is stored
	DecoderParameters decoderArgs = { parser, false, 4 };
	int sts = decoder.Init(decoderArgs);
	auto parsed = new AVPacket();
	for (int i = 0; i < 10; i++) {
		auto output = av_frame_alloc();
		sts = parser->Read();
		EXPECT_EQ(sts, VREADER_OK);
		sts = parser->Get(parsed);
		EXPECT_EQ(sts, VREADER_OK);
		std::future<int> result = std::async([&decoder, &output]() {
			return decoder.GetFrame(0, "visualize", output);

		});
		//wait some time to gurantee that GetFrame will be executed before Decode
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		EXPECT_EQ(decoder.getFrameIndex(), i);
		sts = decoder.Decode(parsed);
		EXPECT_EQ(sts, VREADER_OK);
		EXPECT_NE(result.get(), VREADER_REPEAT);
		EXPECT_EQ(decoder.getFrameIndex(), i + 1);
		//if use parser + decoder without VideoReader class need to ensure that frame from decoder will be deleted due to DPB buffer
		av_frame_unref(output);
	}
}

//DPB is less than internal buffer
TEST_F(Decoder_Init, DPBLessBuffer) {
	av_log_set_callback([](void *ptr, int level, const char *fmt, va_list vargs) {
		return;
	});
	Decoder decoder;
	//the buffer size is 1 frame, so only the last frame is stored
	DecoderParameters decoderArgs = { parser, false, 12 };
	int sts = decoder.Init(decoderArgs);
	int decoderSts = VREADER_OK;
	auto parsed = new AVPacket();
	for (int i = 0; i < 10; i++) {
		auto output = av_frame_alloc();
		sts = parser->Read();
		EXPECT_EQ(sts, VREADER_OK);
		sts = parser->Get(parsed);
		EXPECT_EQ(sts, VREADER_OK);
		std::future<int> result = std::async([&decoder, &output]() {
			return decoder.GetFrame(0, "visualize", output);

		});
		//wait some time to gurantee that GetFrame will be executed before Decode
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		EXPECT_EQ(decoder.getFrameIndex(), i);
		sts = decoder.Decode(parsed);
		if (sts != 0) {
			EXPECT_NE(i, 10);
			decoder.notifyConsumers();
			break;
		}
		EXPECT_NE(result.get(), VREADER_REPEAT);
		EXPECT_EQ(decoder.getFrameIndex(), i + 1);
		//if use parser + decoder without VideoReader class need to ensure that frame from decoder will be deleted due to DPB buffer
		av_frame_unref(output);
	}
}

TEST_F(Decoder_Init, SeveralThreads) {
	Decoder decoder;
	//the buffer size is 1 frame, so only the last frame is stored
	DecoderParameters decoderArgs = { parser, false, 4 };
	int sts = decoder.Init(decoderArgs);
	std::vector<std::shared_ptr<AVFrame> > visualizeFrames;
	std::vector<std::shared_ptr<AVFrame> > processingFrames;
	std::thread startProcessing;
	std::thread resultVisualize = std::thread([&decoder, &visualizeFrames]() {
		for (int i = 0; i < 5; i++) {
			//We save frames from decoder to vector without copying, so we store reference to DPB, so we iterate over only fisrt few frames
			auto output = av_frame_alloc();
			decoder.GetFrame(0, "visualize", output);
			visualizeFrames.push_back(std::shared_ptr<AVFrame>(output, av_frame_unref));
		}
	});
	std::thread resultProcessing = std::thread([&decoder, &processingFrames]() {
		for (int i = 0; i < 5; i++) {
			auto output = av_frame_alloc();
			decoder.GetFrame(-1, "processing", output);
			processingFrames.push_back(std::shared_ptr<AVFrame>(output, av_frame_unref));
		}
	});
	//wait some time to gurantee that GetFrame will be executed before Decode
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	startProcessing = std::thread(processing, std::ref(parser), std::ref(decoder), std::ref(parsed), 5);
	startProcessing.join();
	resultProcessing.join();
	resultVisualize.join();

	int width = visualizeFrames[0]->width;
	int height = visualizeFrames[0]->height;
	//returned 0 frame because -1 required + 1 buffer size = 0
	std::vector<uint8_t> outputYVisualize(width * height);
	std::vector<uint8_t> outputUVVisualize(width * height / 2);
	ASSERT_EQ(cudaMemcpy2D(&outputYVisualize[0], width, visualizeFrames[0]->data[0], visualizeFrames[0]->linesize[0], width, height, cudaMemcpyDeviceToHost), 0);
	ASSERT_EQ(cudaMemcpy2D(&outputUVVisualize[0], width, visualizeFrames[0]->data[1], visualizeFrames[0]->linesize[1], width, height / 2, cudaMemcpyDeviceToHost), 0);

	std::vector<uint8_t> outputYProcessing(width * height);
	std::vector<uint8_t> outputUVProcessing(width * height / 2);
	ASSERT_EQ(cudaMemcpy2D(&outputYProcessing[0], width, processingFrames[1]->data[0], processingFrames[1]->linesize[0], width, height, cudaMemcpyDeviceToHost), 0);
	ASSERT_EQ(cudaMemcpy2D(&outputUVProcessing[0], width, processingFrames[1]->data[1], processingFrames[1]->linesize[1], width, height / 2, cudaMemcpyDeviceToHost), 0);
	//CRC for zero frame of bbb_1080x608_420_10.h264
	//CRC32 - 3265466497
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputYVisualize[0], width * height), 
		             av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputYProcessing[0], width * height));

	//CRC32 - 2183362287
	ASSERT_EQ(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputUVVisualize[0], width * height / 2), 
			  av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1, &outputUVProcessing[0], width * height / 2));

	//first frame from "processing" should be empty
	ASSERT_EQ(processingFrames[0]->data[0], nullptr);
	ASSERT_EQ(processingFrames[0]->data[1], nullptr);
}
