#include "WrapperPython.h"
#include <iostream>

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

int TensorStream::initPipeline(std::string inputFile) {
	int sts = VREADER_OK;
	shouldWork = true;
	av_log_set_callback(logCallback);
	START_LOG_FUNCTION(std::string("Initializing() "));
	/*avoiding Tensor CUDA lazy initializing for further context attaching*/
	START_LOG_BLOCK(std::string("Tensor CUDA init"));
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
	auto videoStream = parser->getFormatContext()->streams[parser->getVideoIndex()];
	frameRate = std::pair<int, int>(videoStream->codec->framerate.den, videoStream->codec->framerate.num);
	if (!frameRate.second) {
		LOG_VALUE(std::string("Frame rate in bitstream hasn't been found, using guessed value"));
		frameRate = std::pair<int, int>(videoStream->r_frame_rate.den, videoStream->r_frame_rate.num);
	}

	CHECK_STATUS(frameRate.second == 0 || frameRate.first == 0);
	CHECK_STATUS((int)(frameRate.second / frameRate.first) > frameRateConstraints);
	realTimeDelay = ((float)frameRate.first /
		(float)frameRate.second) * 1000;
	LOG_VALUE(std::string("Frame rate: ") + std::to_string((int)(frameRate.second / frameRate.first)));
	END_LOG_FUNCTION(std::string("Initializing() "));
	return sts;
}

std::map<std::string, int> TensorStream::getInitializedParams() {
	std::map<std::string, int> params;
	params.insert(std::map<std::string, int>::value_type("framerate_num", frameRate.second));
	params.insert(std::map<std::string, int>::value_type("framerate_den", frameRate.first));
	params.insert(std::map<std::string, int>::value_type("width", decoder->getDecoderContext()->width));
	params.insert(std::map<std::string, int>::value_type("height", decoder->getDecoderContext()->height));
	return params;
}

int TensorStream::processingLoop() {
	std::unique_lock<std::mutex> locker(closeSync);
	int sts = VREADER_OK;
	//change to end of file
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
		//Parse package to find some syntax issues
		sts = parser->Analyze(parsed);
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

int TensorStream::startProcessing() {
	int sts = VREADER_OK;
	sts = processingLoop();
	LOG_VALUE(std::string("Processing was interrupted or stream has ended"));
	//we should unlock mutex to allow get() function end execution
	decoder->notifyConsumers();
	LOG_VALUE(std::string("All consumers were notified about processing end"));
	CHECK_STATUS(sts);
	return sts;
}

std::tuple<at::Tensor, int> TensorStream::getFrame(std::string consumerName, int index, FrameParameters frameParameters) {
	AVFrame* decoded;
	AVFrame* processedFrame;
	at::Tensor outputTensor;
	std::tuple<at::Tensor, int> outputTuple;
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
	sts = vpp->Convert(decoded, processedFrame, frameParameters, consumerName); 
	CHECK_STATUS_THROW(sts);
	END_LOG_BLOCK(std::string("vpp->Convert"));
	START_LOG_BLOCK(std::string("tensor->ConvertFromBlob"));
	if (frameParameters.color.normalization)
		outputTensor = torch::from_blob(processedFrame->opaque, { 1, processedFrame->channels, processedFrame->height, processedFrame->width }, torch::CUDA(at::kFloat));
	else
		outputTensor = torch::from_blob(processedFrame->opaque, { 1, processedFrame->channels, processedFrame->height, processedFrame->width }, torch::CUDA(at::kByte));

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
void TensorStream::endProcessing(int mode) {
	shouldWork = false;
	LOG_VALUE(std::string("End processing async part"));
	{
		std::unique_lock<std::mutex> locker(closeSync);
		LOG_VALUE(std::string("End processing sync part start"));
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
		LOG_VALUE(std::string("End processing sync part end"));
	}
}

void TensorStream::enableLogs(int _logsLevel) {
	if (_logsLevel) {
		logsLevel = static_cast<LogsLevel>(_logsLevel);
		if (!logsFile.is_open() && _logsLevel > 0) {
			logsFile.open(logFileName);
		}
	}
}

template <class T>
int TensorStream::dumpFrame(T* frame, FrameParameters frameParameters, std::shared_ptr<FILE> dumpFile) {
	int status = VREADER_OK;
	START_LOG_FUNCTION(std::string("dumpFrame()"));
	status = vpp->DumpFrame<T>(frame, frameParameters, dumpFile);
	END_LOG_FUNCTION(std::string("dumpFrame()"));
	return status;
}

template
int TensorStream::dumpFrame(float* frame, FrameParameters frameParameters, std::shared_ptr<FILE> dumpFile);

template
int TensorStream::dumpFrame(uint8_t* frame, FrameParameters frameParameters, std::shared_ptr<FILE> dumpFile);


static TensorStream reader;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	py::class_<FrameParameters>(m, "FrameParameters")
		.def(py::init<>())
		.def_readwrite("resize", &FrameParameters::resize)
		.def_readwrite("color", &FrameParameters::color);

	py::class_<ResizeOptions>(m, "ResizeOptions")
		.def(py::init<>())
		.def_readwrite("width", &ResizeOptions::width)
		.def_readwrite("height", &ResizeOptions::height)
		.def_readwrite("resizeType", &ResizeOptions::type);

	py::class_<ColorOptions>(m, "ColorOptions")
		.def(py::init<>())
		.def_readwrite("normalization", &ColorOptions::normalization)
		.def_readwrite("planesPos", &ColorOptions::planesPos)
		.def_readwrite("dstFourCC", &ColorOptions::dstFourCC);

	py::enum_<ResizeType>(m, "ResizeType")
		.value("NEAREST", ResizeType::NEAREST)
		.value("BILINEAR", ResizeType::BILINEAR)
		.export_values();

	py::enum_<Planes>(m, "Planes")
		.value("PLANAR", Planes::PLANAR)
		.value("MERGED", Planes::MERGED)
		.export_values();

	py::enum_<FourCC>(m, "FourCC")
		.value("Y800", FourCC::Y800)
		.value("RGB24", FourCC::RGB24)
		.value("BGR24", FourCC::BGR24)
		.export_values();


	m.def("init", [](std::string rtmp) -> int {
		return reader.initPipeline(rtmp);
	});

	m.def("getPars", []() -> std::map<std::string, int> {
		return reader.getInitializedParams();
	});

	m.def("start", [](void) {
		py::gil_scoped_release release;
		return reader.startProcessing();
	});
	
	m.def("get", [](std::string name, int delay, FrameParameters frameParameters) {
		py::gil_scoped_release release;

		return reader.getFrame(name, delay, frameParameters);
	});

	m.def("dump", [](at::Tensor stream, std::string consumerName, FrameParameters frameParameters) {
		py::gil_scoped_release release;

		if (!frameParameters.resize.width) {
			frameParameters.resize.width = stream.size(3);
		}
		
		if (!frameParameters.resize.height) {
			frameParameters.resize.height = stream.size(2);
		}

		stream = stream.reshape({ stream.size(2), stream.size(3), stream.size(1) });
		//Kind of magic, need to concatenate string from Python with std::string to avoid issues in frame dumping (some strange artifacts appeared if create file using consumerName)
		std::string dumpName = consumerName + std::string("");
		std::shared_ptr<FILE> dumpFrame = std::shared_ptr<FILE>(fopen(dumpName.c_str(), "ab+"), std::fclose);
		int sts;
		if (frameParameters.color.normalization)
			sts = reader.dumpFrame((float*)stream.data_ptr(), frameParameters, dumpFrame);
		else
			sts = reader.dumpFrame((uint8_t*)stream.data_ptr(), frameParameters, dumpFrame);
		return sts;
	});

	m.def("enableLogs", [](int logsLevel) {
		reader.enableLogs(logsLevel);
	});

	m.def("close", [](int mode) {
		reader.endProcessing(mode);
	});
}