#include "Wrapper.h"
#include "Common.h"

int TensorCommon::initPipeline(std::string inputFile, uint8_t maxConsumers, uint8_t cudaDevice, uint8_t decoderBuffer) {
	int sts = VREADER_OK;
	//initialize logger
	if (_logger == nullptr) {
		_logger = std::make_shared<Logger>();
		_logger->initialize(LogsLevel::NONE);
	}
	//set cuda device
	int cudaDevicesNumber;
	sts = cudaGetDeviceCount(&cudaDevicesNumber);
	CHECK_STATUS(sts);
	if (cudaDevice >= 0 && cudaDevice < cudaDevicesNumber) {
		_currentCUDADevice = cudaDevice;
	}
	else {
		int device;
		auto sts = cudaGetDevice(&device);
		_currentCUDADevice = device;
	}

	SET_CUDA_DEVICE();

	//initialize mandatory parser, decoder and vpp
	LOG_VALUE(std::string("Chosen GPU: ") + std::to_string(_currentCUDADevice), LogsLevel::LOW);
	_parser = std::make_shared<Parser>();
	_decoder = std::make_shared<Decoder>();
	_vpp = std::make_shared<VideoProcessor>();
	ParserParameters parserArgs = { inputFile, false };
	START_LOG_BLOCK(std::string("parser->Init"));
	sts = _parser->Init(parserArgs, _logger);
	CHECK_STATUS(sts);
	END_LOG_BLOCK(std::string("parser->Init"));
	DecoderParameters decoderArgs = { _parser, false, decoderBuffer };
	START_LOG_BLOCK(std::string("decoder->Init"));
	sts = _decoder->Init(decoderArgs, _logger);
	CHECK_STATUS(sts);
	END_LOG_BLOCK(std::string("decoder->Init"));
	START_LOG_BLOCK(std::string("VPP->Init"));
	sts = _vpp->Init(_logger, maxConsumers, false);
	CHECK_STATUS(sts);
	END_LOG_BLOCK(std::string("VPP->Init"));

	return sts;
}

template
int TensorCommon::dumpFrame<unsigned char>(unsigned char* frame, FrameParameters frameParameters, std::shared_ptr<FILE> dumpFile);

template
int TensorCommon::dumpFrame<float>(float* frame, FrameParameters frameParameters, std::shared_ptr<FILE> dumpFile);

template <class T>
int TensorCommon::dumpFrame(T* frame, FrameParameters frameParameters, std::shared_ptr<FILE> dumpFile) {
	int status = VREADER_OK;
	PUSH_RANGE("TensorStream::dumpFrame", NVTXColors::YELLOW);
	START_LOG_FUNCTION(std::string("dumpFrame()"));
	status = vpp->DumpFrame(frame, frameParameters, dumpFile);
	END_LOG_FUNCTION(std::string("dumpFrame()"));
	return status;
}


void TensorCommon::endProcessing() {
	SET_CUDA_DEVICE_THROW();
	PUSH_RANGE("TensorStream::endProcessing", NVTXColors::GREEN);
	LOG_VALUE(std::string("End processing sync part start"), LogsLevel::LOW);
	if (_parser)
		_parser->Close();
	if (_decoder)
		_decoder->Close();
	if (_vpp)
		_vpp->Close();
	LOG_VALUE(std::string("End processing sync part end"), LogsLevel::LOW);
}

void TensorCommon::enableLogs(int level) {
	auto logsLevel = static_cast<LogsLevel>(level);
	if (_logger == nullptr) {
		_logger = std::make_shared<Logger>();
	}
	_logger->initialize(logsLevel);
}

void TensorCommon::enableNVTX() {
	if (_logger == nullptr) {
		_logger = std::make_shared<Logger>();
		_logger->initialize(LogsLevel::NONE);
	}
	_logger->enableNVTX = true;
}

void logCallback(void *ptr, int level, const char *fmt, va_list vargs) {
	if (level > AV_LOG_ERROR)
		return;
}

int TensorStreamCommon::initPipeline(std::string inputFile, uint8_t maxConsumers, uint8_t cudaDevice, uint8_t decoderBuffer, FrameRateMode frameRateMode) {
	int sts = VREADER_OK;
	PUSH_RANGE("TensorStreamCommon::initPipeline", NVTXColors::GREEN);
	START_LOG_FUNCTION(std::string("Initializing() "));
	sts = TensorCommon::initPipeline(inputFile, maxConsumers, cudaDevice, decoderBuffer);

	_shouldWork = true;
	_skipAnalyze = false;
	_frameRateMode = frameRateMode;
	av_log_set_callback(logCallback);
	_parsed = new AVPacket();
	for (int i = 0; i < maxConsumers; i++) {
		_decodedArr.push_back(std::make_pair(std::string("empty"), av_frame_alloc()));
		_processedArr.push_back(std::make_pair(std::string("empty"), av_frame_alloc()));
	}
	auto videoStream = _parser->getFormatContext()->streams[_parser->getVideoIndex()];
	_frameRate = std::pair<int, int>(videoStream->codec->framerate.den, videoStream->codec->framerate.num);
	if (!_frameRate.second) {
		LOG_VALUE(std::string("Frame rate in bitstream hasn't been found, using guessed value"), LogsLevel::LOW);
		_frameRate = std::pair<int, int>(videoStream->r_frame_rate.den, videoStream->r_frame_rate.num);
	}
	CHECK_STATUS(_frameRate.second == 0 || _frameRate.first == 0);
	CHECK_STATUS((int)(_frameRate.second / _frameRate.first) > frameRateConstraints);
	_realTimeDelay = ((float)_frameRate.first /
		(float)_frameRate.second) * 1000;
	LOG_VALUE(std::string("Frame rate: ") + std::to_string((int)(_frameRate.second / _frameRate.first)), LogsLevel::LOW);
	END_LOG_FUNCTION(std::string("Initializing() "));
	return sts;
}

int TensorStreamCommon::startProcessing() {
	int sts = VREADER_OK;
	sts = processingLoop();
	LOG_VALUE(std::string("Processing was interrupted or stream has ended"), LogsLevel::LOW);
	//we should unlock mutex to allow get() function end execution
	if (_decoder)
		_decoder->notifyConsumers();
	LOG_VALUE(std::string("All consumers were notified about processing end"), LogsLevel::LOW);
	CHECK_STATUS(sts);
	return sts;
}

int TensorStreamCommon::checkGetComplete(std::map<std::string, bool>& blockingStatuses) {
	int numberReady = 0;
	for (auto item : blockingStatuses) {
		if (item.second) {
			numberReady++;
		}
	}
	if (numberReady == blockingStatuses.size()) {
		//return statuses back to unfinished
		for (auto &item : blockingStatuses) {
			item.second = false;
		}
		return true;
	}
	return false;
}

int TensorStreamCommon::processingLoop() {
	std::unique_lock<std::mutex> locker(_closeSync);
	int sts = VREADER_OK;
	SET_CUDA_DEVICE();
	while (_shouldWork) {
		PUSH_RANGE("TensorStream::processingLoop", NVTXColors::GREEN);
		START_LOG_FUNCTION(std::string("Processing() ") + std::to_string(_decoder->getFrameIndex() + 1) + std::string(" frame"));
		std::chrono::high_resolution_clock::time_point waitTime = std::chrono::high_resolution_clock::now();
		START_LOG_BLOCK(std::string("parser->Read"));
		sts = _parser->Read();
		END_LOG_BLOCK(std::string("parser->Read"));
		if (sts == AVERROR(EAGAIN))
			continue;
		CHECK_STATUS(sts);
		START_LOG_BLOCK(std::string("parser->Get"));
		sts = _parser->Get(_parsed);
		CHECK_STATUS(sts);
		END_LOG_BLOCK(std::string("parser->Get"));
		if (!_skipAnalyze) {
			START_LOG_BLOCK(std::string("parser->Analyze"));
			//Parse package to find some syntax issues, don't handle errors returned from this function
			sts = _parser->Analyze(_parsed);
			END_LOG_BLOCK(std::string("parser->Analyze"));
		}
		START_LOG_BLOCK(std::string("decoder->Decode"));
		sts = _decoder->Decode(_parsed);
		END_LOG_BLOCK(std::string("decoder->Decode"));
		//Need more data for decoding
		if (sts == AVERROR(EAGAIN) || sts == AVERROR_EOF)
			continue;
		CHECK_STATUS(sts);

		if (_frameRateMode == FrameRateMode::NATIVE) {
			START_LOG_BLOCK(std::string("sleep"));
			PUSH_RANGE("TensorStream::Sleep", NVTXColors::PURPLE);
			//wait here
			int sleepTime = _realTimeDelay - std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::high_resolution_clock::now() - waitTime).count();
			if (sleepTime > 0) {
				std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
			}
			LOG_VALUE(std::string("Should sleep for: ") + std::to_string(sleepTime), LogsLevel::HIGH);
			END_LOG_BLOCK(std::string("sleep"));
		}
		if (_frameRateMode == FrameRateMode::BLOCKING) {
			std::unique_lock<std::mutex> locker(_blockingSync);
			START_LOG_BLOCK(std::string("blocking wait"));
			PUSH_RANGE("TensorStream::Blocking", NVTXColors::PURPLE);
			std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();
			/*
			wait for end
			*/
			//Should check whether all threads completed their job or not
			bool frameEnd = checkGetComplete(_blockingStatuses);
			while (!frameEnd) {
				//wait call release mutex, once control returned back it automatically occupied
				_blockingCV.wait(locker);
				//if woke up, need to check status
				frameEnd = checkGetComplete(_blockingStatuses);
			}
			/*
			*/
			std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
			int blockingTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
			LOG_VALUE(std::string("Blocking time: ") + std::to_string(blockingTime), LogsLevel::HIGH);
			END_LOG_BLOCK(std::string("blocking wait"));
		}
		END_LOG_FUNCTION(std::string("Processing() ") + std::to_string(_decoder->getFrameIndex()) + std::string(" frame"));
	}
	return sts;
}

std::map<std::string, int> TensorStreamCommon::getInitializedParams() {
	PUSH_RANGE("TensorStream::getInitializedParams", NVTXColors::GREEN);
	std::map<std::string, int> params;
	params.insert(std::map<std::string, int>::value_type("framerate_num", _frameRate.second));
	params.insert(std::map<std::string, int>::value_type("framerate_den", _frameRate.first));
	params.insert(std::map<std::string, int>::value_type("width", _decoder->getDecoderContext()->width));
	params.insert(std::map<std::string, int>::value_type("height", _decoder->getDecoderContext()->height));
	return params;
}

template <class T>
std::tuple<T*, int> TensorStreamCommon::getFrame(std::string consumerName, int index, FrameParameters frameParameters) {
	SET_CUDA_DEVICE_THROW();
	AVFrame* decoded;
	AVFrame* processedFrame;
	std::tuple<T*, int> outputTuple;
	if (frameRateMode == FrameRateMode::BLOCKING) {
		//Critical section because we check map size in processingLoop()
		std::unique_lock<std::mutex> locker(blockingSync);
		//this will be executed only once at the start
		if (!blockingStatuses[consumerName]) {
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
std::tuple<float*, int> TensorStreamCommon::getFrame(std::string consumerName, int index, FrameParameters frameParameters);

template
std::tuple<unsigned char*, int> TensorStreamCommon::getFrame(std::string consumerName, int index, FrameParameters frameParameters);

void TensorStreamCommon::skipAnalyzeStage() {
	_skipAnalyze = true;
}

void TensorStreamCommon::setTimeout(int timeout) {
	timeoutFrame = timeout;
}

int TensorStreamCommon::getTimeout() {
	return timeoutFrame;
}

int TensorStreamCommon::getDelay() {
	return _realTimeDelay;
}