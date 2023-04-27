#include "WrapperPython.h"
#include <iostream>

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
	/*avoiding Tensor CUDA lazy initializing for further context attaching*/
	START_LOG_BLOCK(std::string("Tensor CUDA init"));
	at::Tensor gt_target = at::empty({ 1 }, at::CUDA(at::kByte));
	END_LOG_BLOCK(std::string("Tensor CUDA init"));
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
	CHECK_STATUS((int)(frameRate.second / frameRate.first) > frameRateConstraints);
	realTimeDelay = ((float)frameRate.first /
		(float)frameRate.second) * 1000;
	LOG_VALUE(std::string("Frame rate: ") + std::to_string((int)(frameRate.second / frameRate.first)), LogsLevel::LOW);

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
			), tensors.end());
		END_LOG_BLOCK(std::string("check tensor to free"));
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
			LOG_VALUE("Dts: " + std::to_string(frameDTS) + " now: " + std::to_string(now), LogsLevel::HIGH);
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

int TensorStream::startProcessing(int cudaDevice) {
	int sts = VREADER_OK;
	//
	int cudaDevicesNumber;
	sts = cudaGetDeviceCount(&cudaDevicesNumber);
	CHECK_STATUS(sts);
	if (cudaDevice >= 0 && cudaDevice < cudaDevicesNumber) {
		sts = cudaSetDevice(cudaDevice);
		CHECK_STATUS(sts);
	}
	sts = processingLoop();
	LOG_VALUE(std::string("Processing was interrupted or stream has ended"), LogsLevel::LOW);
	//we should unlock mutex to allow get() function end execution
	if (decoder)
		decoder->notifyConsumers();
	LOG_VALUE(std::string("All consumers were notified about processing end"), LogsLevel::LOW);
	CHECK_STATUS(sts);
	return sts;
}

std::tuple<at::Tensor, int> TensorStream::getFrame(std::string consumerName, int index, FrameParameters frameParameters) {
	SET_CUDA_DEVICE_THROW();
	AVFrame* decoded;
	AVFrame* processedFrame;
	at::Tensor outputTensor;
	std::tuple<at::Tensor, int> outputTuple;
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
	START_LOG_BLOCK(std::string("tensor->ConvertFromBlob"));

	float channels = channelsByFourCC(frameParameters.color.dstFourCC);
	switch (frameParameters.color.dstFourCC) {
		case FourCC::RGB24:
		case FourCC::BGR24:
			if (frameParameters.color.planesPos == Planes::MERGED)
				outputTensor = torch::from_blob(processedFrame->opaque, { processedFrame->height, processedFrame->width, (int) channels },
					c10::TensorOptions(frameParameters.color.normalization ? at::kFloat : at::kByte).device(torch::Device(at::kCUDA, currentCUDADevice)));
			else
				outputTensor = torch::from_blob(processedFrame->opaque, { (int) channels, processedFrame->height, processedFrame->width },
					c10::TensorOptions(frameParameters.color.normalization ? at::kFloat : at::kByte).device(torch::Device(at::kCUDA, currentCUDADevice)));
			break;
		case FourCC::YUV444:
			outputTensor = torch::from_blob(processedFrame->opaque, { processedFrame->height, processedFrame->width, (int) channels },
				c10::TensorOptions(frameParameters.color.normalization ? at::kFloat : at::kByte).device(torch::Device(at::kCUDA, currentCUDADevice)));
			break;
		case FourCC::UYVY:
		case FourCC::NV12:
		case FourCC::Y800:
			outputTensor = torch::from_blob(processedFrame->opaque, { 1, (int) (processedFrame->height * channels), processedFrame->width},
				c10::TensorOptions(frameParameters.color.normalization ? at::kFloat : at::kByte).device(torch::Device(at::kCUDA, currentCUDADevice)));
			break;
		case FourCC::HSV:
			outputTensor = torch::from_blob(processedFrame->opaque, { processedFrame->height, processedFrame->width, (int) channels },
				c10::TensorOptions(at::kFloat).device(torch::Device(at::kCUDA, currentCUDADevice)));
	}
	outputTuple = std::make_tuple(outputTensor, indexFrame);
	END_LOG_BLOCK(std::string("tensor->ConvertFromBlob"));
	/*
	Store tensor to be able get count of references for further releasing CUDA memory if strong_refs = 1
	*/
	START_LOG_BLOCK(std::string("add tensor"));
	std::unique_lock<std::mutex> locker(freeSync);
	tensors.push_back(outputTensor);
	END_LOG_BLOCK(std::string("add tensor"));
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
		tensors.clear();
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

int TensorStream::dumpFrame(at::Tensor stream, std::string consumerName, FrameParameters frameParameters) {
	int status = VREADER_OK;
	PUSH_RANGE("TensorStream::dumpFrame", NVTXColors::YELLOW);
	START_LOG_FUNCTION(std::string("dumpFrame()"));
	if (!frameParameters.resize.width) {
		if (channelsByFourCC(frameParameters.color.dstFourCC) == 3) {
			//in this case size of Tensor is (height, width, channels)
			frameParameters.resize.width = stream.size(1);
		}
		else {
			//in this case size of Tensor is (1, height * channels, width)
			frameParameters.resize.width = stream.size(2);
		}
	}

	if (!frameParameters.resize.height) {
		if (channelsByFourCC(frameParameters.color.dstFourCC) == 3) {
			//in this case size of Tensor is (height, width, channels)
			frameParameters.resize.height = stream.size(0);
		}
		else {
			//in this case size of Tensor is (1, height * channels, width)
			frameParameters.resize.height = stream.size(1) / channelsByFourCC(frameParameters.color.dstFourCC);
		}
	}

	//Kind of magic, need to concatenate string from Python with std::string to avoid issues in frame dumping (some strange artifacts appeared if create file using consumerName)
	std::string dumpName = consumerName + std::string(".yuv");
	std::shared_ptr<FILE> dumpFrame = std::shared_ptr<FILE>(fopen(dumpName.c_str(), "ab+"), std::fclose);
	if (frameParameters.color.normalization)
		status = vpp->DumpFrame<float>((float*)stream.data_ptr(), frameParameters, dumpFrame);
	else
		status = vpp->DumpFrame<uint8_t>((uint8_t*)stream.data_ptr(), frameParameters, dumpFrame);
	END_LOG_FUNCTION(std::string("dumpFrame()"));
	return status;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	py::class_<FrameParameters>(m, "FrameParameters")
		.def(py::init<>())
		.def_readwrite("resize", &FrameParameters::resize)
		.def_readwrite("color", &FrameParameters::color)
		.def_readwrite("crop", &FrameParameters::crop);

	py::class_<CropOptions>(m, "CropOptions")
		.def(py::init<>())
		.def_readwrite("leftTopCorner", &CropOptions::leftTopCorner)
		.def_readwrite("rightBottomCorner", &CropOptions::rightBottomCorner);

	py::class_<ResizeOptions>(m, "ResizeOptions")
		.def(py::init<>())
		.def_readwrite("width", &ResizeOptions::width)
		.def_readwrite("height", &ResizeOptions::height)
		.def_readwrite("resizeType", &ResizeOptions::type);

	py::class_<ColorOptions>(m, "ColorOptions")
		.def(py::init<FourCC>())
		.def_readwrite("normalization", &ColorOptions::normalization)
		.def_readwrite("planesPos", &ColorOptions::planesPos)
		.def_readwrite("dstFourCC", &ColorOptions::dstFourCC);

	py::enum_<ResizeType>(m, "ResizeType")
		.value("NEAREST", ResizeType::NEAREST)
		.value("BILINEAR", ResizeType::BILINEAR)
		.value("BICUBIC", ResizeType::BICUBIC)
		.value("AREA", ResizeType::AREA)
		.export_values();

	py::enum_<Planes>(m, "Planes")
		.value("PLANAR", Planes::PLANAR)
		.value("MERGED", Planes::MERGED)
		.export_values();

	py::enum_<FourCC>(m, "FourCC")
		.value("Y800", FourCC::Y800)
		.value("RGB24", FourCC::RGB24)
		.value("BGR24", FourCC::BGR24)
		.value("NV12",   FourCC::NV12)
		.value("UYVY",   FourCC::UYVY)
		.value("YUV444", FourCC::YUV444)
		.value("HSV",    FourCC::HSV)
		.export_values();

	py::enum_<FrameRateMode>(m, "FrameRateMode")
		.value("NATIVE", FrameRateMode::NATIVE)
		.value("NATIVE_SIMPLE", FrameRateMode::NATIVE_SIMPLE)
		.value("NATIVE_LOW_DELAY", FrameRateMode::NATIVE_LOW_DELAY)
		.value("FAST", FrameRateMode::FAST)
		.value("BLOCKING", FrameRateMode::BLOCKING)
		.export_values();

	py::class_<TensorStream>(m, "TensorStream")
		.def(py::init<>())
		.def("init", &TensorStream::initPipeline)
		.def("getPars", &TensorStream::getInitializedParams)
		.def("start", &TensorStream::startProcessing, py::arg("cudaDevice") = defaultCUDADevice, py::call_guard<py::gil_scoped_release>())
		.def("get", &TensorStream::getFrame, py::call_guard<py::gil_scoped_release>())
		.def("dump", &TensorStream::dumpFrame, py::call_guard<py::gil_scoped_release>())
		.def("enableNVTX", &TensorStream::enableNVTX)
		.def("enableLogs", &TensorStream::enableLogs)
		.def("close", &TensorStream::endProcessing)
		.def("skipAnalyze", &TensorStream::skipAnalyzeStage)
		.def("setTimeout", &TensorStream::setTimeout);
}