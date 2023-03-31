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
}

int TensorStream::initPipeline(std::string inputFile, uint8_t maxConsumers, uint8_t cudaDevice, uint8_t decoderBuffer, FrameRateMode frameRateMode) {
	int sts = VREADER_OK;
	shouldWork = true;
	skipAnalyze = false;
	this->frameRateMode = frameRateMode;
	if (logger == nullptr) {
		logger = std::make_shared<Logger>();
		logger->initialize(LogsLevel::NONE);
	}
	int cudaDevicesNumber;
	sts = cudaGetDeviceCount(&cudaDevicesNumber);
	CHECK_STATUS(sts);
	if (cudaDevice >= 0 && cudaDevice < cudaDevicesNumber) {
		currentCUDADevice = cudaDevice;
	} else {
		int device;
		auto sts = cudaGetDevice(&device);
		currentCUDADevice = device;
	}

	SET_CUDA_DEVICE();

	PUSH_RANGE("TensorStream::initPipeline", NVTXColors::GREEN);
	av_log_set_callback(logCallback);
	START_LOG_FUNCTION(std::string("Initializing() "));
	LOG_VALUE(std::string("Chosen GPU: ") + std::to_string(currentCUDADevice), LogsLevel::LOW);
	parser = std::make_shared<Parser>();
	decoder = std::make_shared<Decoder>();
	vpp = std::make_shared<VideoProcessor>();
	bool keepBuffer = true;
	if (frameRateMode == NATIVE_LOW_DELAY) {
		keepBuffer = false;
		this->frameRateMode = NATIVE;
	}
	ParserParameters parserArgs = { inputFile, keepBuffer, false };
	START_LOG_BLOCK(std::string("parser->Init"));
	sts = parser->Init(parserArgs, logger);
	CHECK_STATUS(sts);
	END_LOG_BLOCK(std::string("parser->Init"));
	DecoderParameters decoderArgs = { parser, false, decoderBuffer };
	START_LOG_BLOCK(std::string("decoder->Init"));
	sts = decoder->Init(decoderArgs, logger);
	CHECK_STATUS(sts);
	END_LOG_BLOCK(std::string("decoder->Init"));
	START_LOG_BLOCK(std::string("VPP->Init"));
	LOG_VALUE(std::string("Max consumers allowed: ") + std::to_string(maxConsumers), LogsLevel::LOW);
	sts = vpp->Init(logger, maxConsumers, false);
	CHECK_STATUS(sts);
	END_LOG_BLOCK(std::string("VPP->Init"));
	parsed = new AVPacket();
	for (int i = 0; i < maxConsumers; i++) {
		decodedArr.push_back(std::make_pair(std::string("empty"), av_frame_alloc()));
		processedArr.push_back(std::make_pair(std::string("empty"), av_frame_alloc()));
	}
	auto videoStream = parser->getFormatContext()->streams[parser->getVideoIndex()];
	frameRate = std::pair<int, int>(parser->getCodecContext()->framerate.den, parser->getCodecContext()->framerate.num);
	if (!frameRate.second) {
		LOG_VALUE(std::string("Frame rate in bitstream hasn't been found, using guessed value"), LogsLevel::LOW);
		frameRate = std::pair<int, int>(videoStream->r_frame_rate.den, videoStream->r_frame_rate.num);
	}

	CHECK_STATUS(frameRate.second == 0 || frameRate.first == 0);
	CHECK_STATUS((int) (frameRate.second / frameRate.first) > frameRateConstraints);
	realTimeDelay = ((float)frameRate.first /
		(float)frameRate.second) * 1000;
	LOG_VALUE(std::string("Frame rate: ") + std::to_string((int) (frameRate.second / frameRate.first)), LogsLevel::LOW);

	//1) frameindex * framerate.den / framerate.num = frame time in seconds
	//2) 1) * framerate.den / framerate.num = frame time in time base units
	indexToDTSCoeff = (double)(videoStream->r_frame_rate.den * videoStream->time_base.den) / (int64_t(videoStream->r_frame_rate.num) * videoStream->time_base.num);

	//need convert DTS to ms
	//first of all converting DTS to seconds (DTS is measured in timebase.num / timebase.den seconds, so 1 dts = timebase.num / timebase.den seconds)
	//after converting from seconds to ms by dividing by 1000
	DTSToMsCoeff = (double)videoStream->time_base.num / (double)videoStream->time_base.den * (double)1000;
	END_LOG_FUNCTION(std::string("Initializing() "));
	return sts;
}

std::map<std::string, int> TensorStream::getInitializedParams() {
	PUSH_RANGE("TensorStream::getInitializedParams", NVTXColors::GREEN);
	std::map<std::string, int> params;
	params.insert(std::map<std::string, int>::value_type("framerate_num", frameRate.second));
	params.insert(std::map<std::string, int>::value_type("framerate_den", frameRate.first));
	params.insert(std::map<std::string, int>::value_type("width", decoder->getDecoderContext()->width));
	params.insert(std::map<std::string, int>::value_type("height", decoder->getDecoderContext()->height));
	return params;
}

void TensorStream::skipAnalyzeStage() {
	skipAnalyze = true;
}

void TensorStream::setTimeout(int timeout) {
	timeoutFrame = timeout;
}

int TensorStream::getTimeout() {
	return timeoutFrame;
}

int checkGetComplete(std::map<std::string, bool>& blockingStatuses) {
	int numberReady = 0;
	for (auto item : blockingStatuses) {
		if (item.second) {
			numberReady++;
		}
	}
	if (numberReady != 0 && numberReady == blockingStatuses.size()) {
		//return statuses back to unfinished
		for (auto &item : blockingStatuses) {
			item.second = false;
		}
		return true;
	}
	return false;
}

int TensorStream::processingLoop() {
	std::unique_lock<std::mutex> locker(closeSync);
	int sts = VREADER_OK;
	std::pair<int64_t, bool> startDTS = { 0, false };
	std::pair<std::chrono::high_resolution_clock::time_point, bool> startTime = { std::chrono::high_resolution_clock::now(), false };
	SET_CUDA_DEVICE();
	while (shouldWork) {
		PUSH_RANGE("TensorStream::processingLoop", NVTXColors::GREEN);
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
		int64_t frameDTS = parsed->dts;
		LOG_VALUE("Packet dts: " + std::to_string(parsed->dts) + " pts: " + std::to_string(parsed->pts), LogsLevel::LOW);
		if (frameDTS == AV_NOPTS_VALUE && frameRateMode == FrameRateMode::NATIVE) {
			frameDTS = int64_t(decoder->getFrameIndex()) * indexToDTSCoeff;
		}
		if (!skipAnalyze) {
			START_LOG_BLOCK(std::string("parser->Analyze"));
			//Parse package to find some syntax issues, don't handle errors returned from this function
			sts = parser->Analyze(parsed);
			END_LOG_BLOCK(std::string("parser->Analyze"));
		}
		START_LOG_BLOCK(std::string("decoder->Decode"));
		sts = decoder->Decode(parsed);
		END_LOG_BLOCK(std::string("decoder->Decode"));
		//Need more data for decoding
		if (sts == AVERROR(EAGAIN))
			continue;
		CHECK_STATUS(sts);
		START_LOG_BLOCK(std::string("sleep"));
		PUSH_RANGE("TensorStream::Sleep", NVTXColors::PURPLE);
		int sleepTime = 0;
		if (frameRateMode == FrameRateMode::NATIVE_SIMPLE) {
			sleepTime = realTimeDelay - std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::high_resolution_clock::now() - waitTime).count();
		}
		else if (frameRateMode == FrameRateMode::NATIVE) {
			if (!startDTS.second) {
				startDTS.first = frameDTS;
				startDTS.second = true;
			}

			frameDTS -= startDTS.first;

			frameDTS = frameDTS * DTSToMsCoeff;
			if (!startTime.second) {
				startTime.first = std::chrono::high_resolution_clock::now();
				startTime.second = true;
			}

			int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime.first).count();
			LOG_VALUE("Expected time: " + std::to_string(frameDTS) + " now: " + std::to_string(now), LogsLevel::HIGH);
			if (frameDTS > now) {
				sleepTime = frameDTS - now;
			}
		}
		LOG_VALUE(std::string("Should sleep for: ") + std::to_string(sleepTime), LogsLevel::HIGH);
		if (sleepTime > 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
		}
		END_LOG_BLOCK(std::string("sleep"));

		if (frameRateMode == FrameRateMode::BLOCKING) {
			std::unique_lock<std::mutex> locker(blockingSync);
			START_LOG_BLOCK(std::string("blocking wait"));
			PUSH_RANGE("TensorStream::Blocking", NVTXColors::PURPLE);
			std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();
			/*
			wait for end
			*/
			//Should check whether all threads completed their job or not
			bool frameEnd = checkGetComplete(blockingStatuses);
			while (!frameEnd) {
				//wait call release mutex, once control returned back it automatically occupied
				blockingCV.wait(locker);
				//if woke up, need to check status
				frameEnd = checkGetComplete(blockingStatuses);
			}
			/*
			*/
			std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
			int blockingTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
			LOG_VALUE(std::string("Blocking time: ") + std::to_string(blockingTime), LogsLevel::HIGH);
			END_LOG_BLOCK(std::string("blocking wait"));
		}
		END_LOG_FUNCTION(std::string("Processing() ") + std::to_string(decoder->getFrameIndex()) + std::string(" frame"));
	}
	return sts;
}

void TensorStream::drain() {
	int sts = VREADER_OK;
	START_LOG_BLOCK(std::string("TensorStream->Drain"));
	while (sts != AVERROR_EOF) {
		sts = decoder->Drain();
	}
	END_LOG_BLOCK(std::string("TensorStream->Drain"));
}

int TensorStream::startProcessing() {
	int sts = VREADER_OK;
	sts = processingLoop();
	LOG_VALUE(std::string("Processing was interrupted or stream has ended"), LogsLevel::LOW);
	//we should unlock mutex to allow get() function end execution
	if (decoder)
		decoder->notifyConsumers();
	LOG_VALUE(std::string("All consumers were notified about processing end"), LogsLevel::LOW);
	CHECK_STATUS(sts);
	return sts;
}

template <class T>
std::tuple<T*, int> TensorStream::getFrame(std::string consumerName, int index, FrameParameters frameParameters) {
	SET_CUDA_DEVICE_THROW();
	AVFrame* decoded;
	AVFrame* processedFrame;
	std::tuple<T*, int> outputTuple;
	if (frameRateMode == FrameRateMode::BLOCKING) {
		//Critical section because we check map size in processingLoop()
		std::unique_lock<std::mutex> locker(blockingSync);
		//this will be executed only once at the start
		if (blockingStatuses.find(consumerName) == blockingStatuses.end()) {
			blockingStatuses[consumerName] = false;
		}
	}
	PUSH_RANGE("TensorStream::getFrame", NVTXColors::GREEN);
	START_LOG_FUNCTION(std::string("GetFrame()"));
	START_LOG_BLOCK(std::string("findFree decoded frame"));
	{
		std::unique_lock<std::mutex> locker(syncDecoded);
		decoded = findFree<AVFrame*>(consumerName, decodedArr);
		if (decoded == nullptr) {
			throw std::runtime_error(std::to_string(VREADER_ERROR));
		}
	}

	END_LOG_BLOCK(std::string("findFree decoded frame"));
	START_LOG_BLOCK(std::string("findFree converted frame"));
	{
		std::unique_lock<std::mutex> locker(syncRGB);
		processedFrame = findFree<AVFrame*>(consumerName, processedArr);
		if (processedFrame == nullptr) {
			throw std::runtime_error(std::to_string(VREADER_ERROR));
		}
	}
	END_LOG_BLOCK(std::string("findFree converted frame"));
	int indexFrame = VREADER_REPEAT;
	START_LOG_BLOCK(std::string("decoder->GetFrame"));
	while (indexFrame == VREADER_REPEAT) {
		if (decoder == nullptr)
			throw std::runtime_error(std::to_string(VREADER_ERROR));
		
		if (decoder->isDraining())
			indexFrame = decoder->GetFrameDrain(decoded);
		else
			indexFrame = decoder->GetFrame(index, consumerName, decoded);
	}
	END_LOG_BLOCK(std::string("decoder->GetFrame"));
	START_LOG_BLOCK(std::string("vpp->Convert"));
	int sts = VREADER_OK;
	if (vpp == nullptr)
		throw std::runtime_error(std::to_string(VREADER_ERROR));
	sts = vpp->Convert(decoded, processedFrame, frameParameters, consumerName);
	CHECK_STATUS_THROW(sts);
	END_LOG_BLOCK(std::string("vpp->Convert"));
	T* cudaFrame((T*)processedFrame->opaque);
	outputTuple = std::make_tuple(cudaFrame, indexFrame);
	if (frameRateMode == FrameRateMode::BLOCKING) {
		std::unique_lock<std::mutex> locker(blockingSync);
		blockingStatuses[consumerName] = true;
		/*
		send end message
		*/
		blockingCV.notify_all();
		/*
		*/
	}
	END_LOG_FUNCTION(std::string("GetFrame() ") + std::to_string(indexFrame) + std::string(" frame"));
	return outputTuple;
}

template
std::tuple<float*, int> TensorStream::getFrame(std::string consumerName, int index, FrameParameters frameParameters);

template
std::tuple<unsigned char*, int> TensorStream::getFrame(std::string consumerName, int index, FrameParameters frameParameters);

/*
Mode 1 - full close, mode 2 - soft close (for reset)
*/
void TensorStream::endProcessing() {
	shouldWork = false;
	LOG_VALUE(std::string("End processing async part"), LogsLevel::LOW);
	{
		//force processing thread to wake up and end work
		if (frameRateMode == FrameRateMode::BLOCKING) {
			std::unique_lock<std::mutex> locker(blockingSync);
			for (auto &item : blockingStatuses) {
				item.second = true;
			}
			blockingCV.notify_all();
		}
	}
	{
		std::unique_lock<std::mutex> locker(closeSync);
		SET_CUDA_DEVICE_THROW();
		PUSH_RANGE("TensorStream::endProcessing", NVTXColors::GREEN);
		LOG_VALUE(std::string("End processing sync part start"), LogsLevel::LOW);
		if (parser)
			parser->Close();
		if (decoder)
			decoder->Close();
		if (vpp)
			vpp->Close();
		for (auto& item : processedArr)
			av_frame_free(&item.second);
		for (auto& item : decodedArr)
			av_frame_free(&item.second);
		decodedArr.clear();
		processedArr.clear();
		delete parsed;
		parsed = nullptr;
		LOG_VALUE(std::string("End processing sync part end"), LogsLevel::LOW);
	}
}

void TensorStream::enableLogs(int level) {
	auto logsLevel = static_cast<LogsLevel>(level);
	if (logger == nullptr) {
		logger = std::make_shared<Logger>();
	}
	logger->initialize(logsLevel);
}

void TensorStream::enableNVTX() {
	if (logger == nullptr) {
		logger = std::make_shared<Logger>();
		logger->initialize(LogsLevel::NONE);
	}
	logger->enableNVTX = true;
}

template
int TensorStream::dumpFrame<unsigned char>(unsigned char* frame, FrameParameters frameParameters, std::shared_ptr<FILE> dumpFile);

template
int TensorStream::dumpFrame<float>(float* frame, FrameParameters frameParameters, std::shared_ptr<FILE> dumpFile);

template <class T>
int TensorStream::dumpFrame(T* frame, FrameParameters frameParameters, std::shared_ptr<FILE> dumpFile) {
	int status = VREADER_OK;
	PUSH_RANGE("TensorStream::dumpFrame", NVTXColors::YELLOW);
	START_LOG_FUNCTION(std::string("dumpFrame()"));
	status = vpp->DumpFrame(frame, frameParameters, dumpFile);
	END_LOG_FUNCTION(std::string("dumpFrame()"));
	return status;
}

int TensorStream::getDelay() {
	return realTimeDelay;
}