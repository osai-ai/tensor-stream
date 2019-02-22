#include <cuda.h>
#include <cuda_runtime.h>
#include "WrapperC.h"
#include <thread>
#include <time.h>
#include <chrono>
#include <algorithm>

void logCallback(void *ptr, int level, const char *fmt, va_list vargs) {
	if (level > AV_LOG_ERROR)
		return;

	if (logsLevel) {
		std::vector<char> buffer(256);
		vsnprintf(&buffer[0], buffer.size(), fmt, vargs);
		std::string logMessage(&buffer[0]);
		logMessage.erase(std::remove(logMessage.begin(), logMessage.end(), '\n'), logMessage.end());
		LOG_VALUE(std::string("[FFMPEG] ") + logMessage);
	}
}

int VideoReader::initPipeline(std::string inputFile, uint8_t decoderBuffer) {
	int sts = VREADER_OK;
	shouldWork = true;
	av_log_set_callback(logCallback);
	START_LOG_FUNCTION(std::string("Initializing() "));
	parser = std::make_shared<Parser>();
	decoder = std::make_shared<Decoder>();
	vpp = std::make_shared<VideoProcessor>();
	ParserParameters parserArgs = { inputFile, false };
	START_LOG_BLOCK(std::string("parser->Init"));
	sts = parser->Init(parserArgs);
	CHECK_STATUS(sts);
	END_LOG_BLOCK(std::string("parser->Init"));
	DecoderParameters decoderArgs = { parser, false, decoderBuffer };
	START_LOG_BLOCK(std::string("decoder->Init"));
	sts = decoder->Init(decoderArgs);
	CHECK_STATUS(sts);
	END_LOG_BLOCK(std::string("decoder->Init"));
	START_LOG_BLOCK(std::string("VPP->Init"));
	sts = vpp->Init(false);
	CHECK_STATUS(sts);
	END_LOG_BLOCK(std::string("VPP->Init"));
	parsed = new AVPacket();
	for (int i = 0; i < maxConsumers; i++) {
		decodedArr.push_back(std::make_pair(std::string("empty"), av_frame_alloc()));
		processedArr.push_back(std::make_pair(std::string("empty"), av_frame_alloc()));
	}
	auto codecTmp = parser->getFormatContext()->streams[parser->getVideoIndex()]->codec;
	CHECK_STATUS(codecTmp->framerate.num == 0);
	realTimeDelay = ((float)codecTmp->framerate.den /
		(float)codecTmp->framerate.num) * 1000;
	LOG_VALUE(std::string("Native frame rate: ") + std::to_string((int) (codecTmp->framerate.num / codecTmp->framerate.den)));
	END_LOG_FUNCTION(std::string("Initializing() "));
	return sts;
}

std::map<std::string, int> VideoReader::getInitializedParams() {
	auto codecTmp = parser->getFormatContext()->streams[parser->getVideoIndex()]->codec;
	std::map<std::string, int> params;
	params.insert(std::map<std::string, int>::value_type("framerate_num", codecTmp->framerate.num));
	params.insert(std::map<std::string, int>::value_type("framerate_den", codecTmp->framerate.den));
	params.insert(std::map<std::string, int>::value_type("width", decoder->getDecoderContext()->width));
	params.insert(std::map<std::string, int>::value_type("height", decoder->getDecoderContext()->height));
	return params;
}

int VideoReader::processingLoop() {
	std::unique_lock<std::mutex> locker(closeSync);
	int sts = VREADER_OK;
	while (shouldWork) {
		START_LOG_FUNCTION(std::string("Processing() ") + std::to_string(decoder->getFrameIndex() + 1) + std::string(" frame"));
		std::chrono::high_resolution_clock::time_point waitTime = std::chrono::high_resolution_clock::now();
		START_LOG_BLOCK(std::string("parser->Read"));
		sts = parser->Read();
		END_LOG_BLOCK(std::string("parser->Read"));
		if (sts == AVERROR(EAGAIN))
			continue;
		CHECK_STATUS(sts);
		START_LOG_BLOCK(std::string("parser->Get"));
		sts = parser->Get(parsed);
		CHECK_STATUS(sts);
		END_LOG_BLOCK(std::string("parser->Get"));
		START_LOG_BLOCK(std::string("parser->Analyze"));
		//Parse package to find some syntax issues, don't handle errors returned from this function
		sts = parser->Analyze(parsed);
		END_LOG_BLOCK(std::string("parser->Analyze"));
		START_LOG_BLOCK(std::string("decoder->Decode"));
		sts = decoder->Decode(parsed);
		END_LOG_BLOCK(std::string("decoder->Decode"));
		//Need more data for decoding
		if (sts == AVERROR(EAGAIN) || sts == AVERROR_EOF)
			continue;
		CHECK_STATUS(sts);
		
		START_LOG_BLOCK(std::string("sleep"));
		//wait here
		int sleepTime = realTimeDelay - std::chrono::duration_cast<std::chrono::milliseconds>(
											std::chrono::high_resolution_clock::now() - waitTime).count();
		if (sleepTime > 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
		}
		END_LOG_BLOCK(std::string("sleep"));
		END_LOG_FUNCTION(std::string("Processing() ") + std::to_string(decoder->getFrameIndex()) + std::string(" frame"));
	}
	return sts;
}

int VideoReader::startProcessing() {
	int sts = VREADER_OK;
	sts = processingLoop();
	//we should unlock mutex to allow get() function end execution
	decoder->notifyConsumers();
	CHECK_STATUS(sts);
	return sts;
}

std::tuple<std::shared_ptr<uint8_t>, int> VideoReader::getFrame(std::string consumerName, int index, FourCC pixelFormat, int dstWidth, int dstHeight) {
	AVFrame* decoded;
	AVFrame* processedFrame;
	std::tuple<std::shared_ptr<uint8_t>, int> outputTuple;
	FourCC format = static_cast<FourCC>(pixelFormat);
	START_LOG_FUNCTION(std::string("GetFrame()"));
	START_LOG_BLOCK(std::string("findFree decoded frame"));
	{
		std::unique_lock<std::mutex> locker(syncDecoded);
		decoded = findFree<AVFrame*>(consumerName, decodedArr);
	}
	END_LOG_BLOCK(std::string("findFree decoded frame"));
	START_LOG_BLOCK(std::string("findFree converted frame"));
	{
		std::unique_lock<std::mutex> locker(syncRGB);
		processedFrame = findFree<AVFrame*>(consumerName, processedArr);
	}
	END_LOG_BLOCK(std::string("findFree converted frame"));
	int indexFrame = VREADER_REPEAT;
	START_LOG_BLOCK(std::string("decoder->GetFrame"));
	while (indexFrame == VREADER_REPEAT) {
		indexFrame = decoder->GetFrame(index, consumerName, decoded);
	}
	END_LOG_BLOCK(std::string("decoder->GetFrame"));
	START_LOG_BLOCK(std::string("vpp->Convert"));
	int sts = VREADER_OK;
	VPPParameters VPPArgs = { dstWidth, dstHeight, format };
	sts = vpp->Convert(decoded, processedFrame, VPPArgs, consumerName);
	CHECK_STATUS_THROW(sts);
	END_LOG_BLOCK(std::string("vpp->Convert"));
	std::shared_ptr<uint8_t> cudaFrame((uint8_t*) processedFrame->opaque, cudaFree);
	outputTuple = std::make_tuple(cudaFrame, indexFrame);
	/*
	Store tensor to be able get count of references for further releasing CUDA memory if strong_refs = 1
	*/
	START_LOG_BLOCK(std::string("add tensor"));
	std::unique_lock<std::mutex> locker(freeSync);
	END_LOG_BLOCK(std::string("add tensor"));
	END_LOG_FUNCTION(std::string("GetFrame() ") + std::to_string(indexFrame) + std::string(" frame"));
	return outputTuple;
}

/*
Mode 1 - full close, mode 2 - soft close (for reset)
*/
void VideoReader::endProcessing(int mode) {
	shouldWork = false;
	{
		std::unique_lock<std::mutex> locker(closeSync);
		if (mode == HARD && logsFile.is_open()) {
			logsFile.close();
		}
		parser->Close();
		decoder->Close();
		vpp->Close();
		for (auto& item : processedArr)
			av_frame_free(&item.second);
		for (auto& item : decodedArr)
			av_frame_free(&item.second);
		decodedArr.clear();
		processedArr.clear();
		delete parsed;
		parsed = nullptr;
	}
}

void VideoReader::enableLogs(int level) {
	if (level) {
		logsLevel = static_cast<LogsLevel>(level);
		if (!logsFile.is_open() && level > 0) {
			logsFile.open(logFileName);
		}
	}
}

int VideoReader::dumpFrame(std::shared_ptr<uint8_t> frame, int width, int height, FourCC format, std::shared_ptr<FILE> dumpFile) {
	AVFrame* output = av_frame_alloc();
	output->opaque = frame.get();
	output->width = output->linesize[0] = width;
	output->height = output->linesize[1] = height;
	output->channels = (format == RGB24 || format == BGR24) ? 3 : 1;
	switch (format) {
		case RGB24:
			output->format = AV_PIX_FMT_RGB24;
		break;
		case BGR24:
			output->format = AV_PIX_FMT_BGR24;
		break;
		case Y800:
			output->format = AV_PIX_FMT_GRAY8;
		break;
		default:
			return VREADER_UNSUPPORTED;
		break;
	}
	int status = vpp->DumpFrame(output, dumpFile);
	av_frame_free(&output);
	return status;
}

int VideoReader::getDelay() {
	return realTimeDelay;
}