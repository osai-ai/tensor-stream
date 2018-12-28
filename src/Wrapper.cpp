#include <cuda.h>
#include <cuda_runtime.h>
#include "Wrapper.h"
#include <thread>
#include <time.h>
#include <chrono>
#if (__linux__)
#include "nvrtc.h"
#endif


void logCallback(void *ptr, int level, const char *fmt, va_list vargs) {
	if (level > AV_LOG_ERROR)
		return;

	std::vector<char> buffer(256);
	vsnprintf(&buffer[0], buffer.size(), fmt, vargs);
	std::string logMessage(&buffer[0]);
	logMessage.erase(std::remove(logMessage.begin(), logMessage.end(), '\n'), logMessage.end());
	LOG_VALUE(std::string("[FFMPEG] ") + logMessage);
}

int VideoReader::initPipeline(std::string inputFile) {
	int sts = OK;
	shouldWork = true;
	av_log_set_callback(logCallback);
	START_LOG_FUNCTION(std::string("Initializing() "));
	/*avoiding Tensor CUDA lazy initializing for further context attaching*/
	START_LOG_BLOCK(std::string("Tensor CUDA init"));
#if (__linux__)
	int majorCUDAVersion, minorCUDAVersion;
    nvrtcVersion(&majorCUDAVersion, &minorCUDAVersion);
#endif
	at::Tensor gt_target = at::empty({ 1 }, at::CUDA(at::kByte));
	END_LOG_BLOCK(std::string("Tensor CUDA init"));
	parser = std::make_shared<Parser>();
	decoder = std::make_shared<Decoder>();
	vpp = std::make_shared<VideoProcessor>();
	ParserParameters parserArgs = { inputFile, false };
	START_LOG_BLOCK(std::string("parser->Init"));
	sts = parser->Init(parserArgs);
	CHECK_STATUS(sts);
	END_LOG_BLOCK(std::string("parser->Init"));
	DecoderParameters decoderArgs = { parser, false };
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
	int sts = OK;
	//change to end of file
	while (shouldWork) {
		START_LOG_FUNCTION(std::string("Processing() ") + std::to_string(decoder->getFrameIndex() + 1) + std::string(" frame"));
		std::chrono::high_resolution_clock::time_point waitTime = std::chrono::high_resolution_clock::now();
		START_LOG_BLOCK(std::string("parser->Read"));
		sts = parser->Read();
		END_LOG_BLOCK(std::string("parser->Read"));
		if (sts == AVERROR(EAGAIN))
			continue;
		//TODO: expect this behavior only in case of EOF
		CHECK_STATUS(sts);
		START_LOG_BLOCK(std::string("parser->Get"));
		sts = parser->Get(parsed);
		CHECK_STATUS(sts);
		END_LOG_BLOCK(std::string("parser->Get"));
		START_LOG_BLOCK(std::string("parser->Analyze"));
		//Parse package to find some syntax issues
		sts = parser->Analyze(parsed);
		CHECK_STATUS(sts);
		END_LOG_BLOCK(std::string("parser->Analyze"));
		START_LOG_BLOCK(std::string("decoder->Decode"));
		sts = decoder->Decode(parsed);
		END_LOG_BLOCK(std::string("decoder->Decode"));
		//Need more data for decoding
		if (sts == AVERROR(EAGAIN) || sts == AVERROR_EOF)
			continue;
		CHECK_STATUS(sts);
		START_LOG_BLOCK(std::string("check tensor to free"));
		std::unique_lock<std::mutex> locker(freeSync);
		/*
		Need to check count of references of output Tensor and free if strong_refs = 1
		*/
		tensors.erase(
			std::remove_if(
				tensors.begin(),
				tensors.end(),
				[](at::Tensor & item) {
					if (item.use_count() == 1) {
						cudaFree(item.data_ptr());
						return true;
					}
					return false;
				}
			),
			tensors.end()
		);
		END_LOG_BLOCK(std::string("check tensor to free"));

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
	int sts = OK;
	sts = processingLoop();
	//we should unlock mutex to allow get() function end execution
	if (shouldWork)
		decoder->notifyConsumers();
	CHECK_STATUS(sts);
	return sts;
}

std::tuple<at::Tensor, int> VideoReader::getFrame(std::string consumerName, int index, int pixelFormat, int dstWidth, int dstHeight) {
	AVFrame* decoded;
	AVFrame* processedFrame;
	at::Tensor outputTensor;
	std::tuple<at::Tensor, int> outputTuple;
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
	int indexFrame = REPEAT;
	START_LOG_BLOCK(std::string("decoder->GetFrame"));
	while (indexFrame == REPEAT) {
		indexFrame = decoder->GetFrame(index, consumerName, decoded);
	}
	END_LOG_BLOCK(std::string("decoder->GetFrame"));
	START_LOG_BLOCK(std::string("vpp->Convert"));
	int sts = OK;
	VPPParameters VPPArgs = { dstWidth, dstHeight, format };
	sts = vpp->Convert(decoded, processedFrame, VPPArgs, consumerName);
	CHECK_STATUS_THROW(sts);
	END_LOG_BLOCK(std::string("vpp->Convert"));
	START_LOG_BLOCK(std::string("tensor->ConvertFromBlob"));
	outputTensor = torch::CUDA(at::kByte).tensorFromBlob(reinterpret_cast<void*>(processedFrame->opaque), 
		{ processedFrame->height, processedFrame->width, processedFrame->channels});
	outputTuple = std::make_tuple(outputTensor, indexFrame);
	END_LOG_BLOCK(std::string("tensor->ConvertFromBlob"));
	/*
	Store tensor to be able get count of references for further releasing CUDA memory if strong_refs = 1
	*/
	START_LOG_BLOCK(std::string("add tensor"));
	std::unique_lock<std::mutex> locker(freeSync);
	tensors.push_back(outputTensor);
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
		tensors.clear();
		delete parsed;
		parsed = nullptr;
	}
}

void VideoReader::enableLogs(int _logsLevel) {
	if (_logsLevel) {
		logsLevel = static_cast<LogsLevel>(_logsLevel);
		if (!logsFile.is_open() && _logsLevel > 0) {
			logsFile.open(logFileName);
		}
	}
}

int VideoReader::dumpFrame(AVFrame* output, std::shared_ptr<FILE> dumpFile) {
	return vpp->DumpFrame(output, dumpFile);
}