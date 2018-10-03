#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef _DEBUG
#undef _DEBUG
#include <torch/torch.h>
#include <THC/THC.h>
#include <ATen/ATen.h>
#define _DEBUG
#else
#include <torch/torch.h>
#include <THC/THC.h>
#include <ATen/ATen.h>
#endif

#include "Common.h"
#include "Parser.h"
#include "Decoder.h"
#include "VideoProcessor.h"
#include <thread>
#include <time.h>
#include <chrono>

std::shared_ptr<Parser> parser;
std::shared_ptr<Decoder> decoder;
std::shared_ptr<VideoProcessor> vpp;
AVPacket* parsed;
int realTimeDelay = 0;
bool shouldWork;
std::vector<std::pair<std::string, AVFrame*> > decodedArr;
std::vector<std::pair<std::string, AVFrame*> > rgbFrameArr;
std::vector<at::Tensor> tensors;
std::mutex freeSync;

int initPipeline(std::string inputFile) {
	int sts = OK;
	shouldWork = true;
	START_LOG_FUNCTION(std::string("Initializing() "));
	/*avoiding Tensor CUDA lazy initializing for further context attaching*/
	START_LOG_BLOCK(std::string("Tensor CUDA init"));
	at::Tensor gt_target = at::empty(at::CUDA(at::kByte), { 1 });
	END_LOG_BLOCK(std::string("Tensor CUDA init"));
	parser = std::make_shared<Parser>();
	decoder = std::make_shared<Decoder>();
	vpp = std::make_shared<VideoProcessor>();
	//ParserParameters parserArgs = { "rtmp://b.sportlevel.com/relay/pooltop"/*"rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4" */ , false };
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
		rgbFrameArr.push_back(std::make_pair(std::string("empty"), av_frame_alloc()));
	}
	auto codecTmp = parser->getFormatContext()->streams[parser->getVideoIndex()]->codec;
	CHECK_STATUS(codecTmp->framerate.num == 0);
	realTimeDelay = ((float)codecTmp->framerate.den /
		(float)codecTmp->framerate.num) * 1000;
	LOG_VALUE(std::string("Native frame rate: ") + std::to_string(realTimeDelay));
	END_LOG_FUNCTION(std::string("Initializing() "));
	return sts;
}

std::map<std::string, int> getInitializedParams() {
	auto codecTmp = parser->getFormatContext()->streams[parser->getVideoIndex()]->codec;
	std::map<std::string, int> params;
	params.insert(std::map<std::string, int>::value_type("framerate_num", codecTmp->framerate.num));
	params.insert(std::map<std::string, int>::value_type("framerate_den", codecTmp->framerate.den));
	params.insert(std::map<std::string, int>::value_type("width", decoder->getDecoderContext()->width));
	params.insert(std::map<std::string, int>::value_type("height", decoder->getDecoderContext()->height));
	return params;
}

int startProcessing() {
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
			/*auto start = std::chrono::system_clock::now();
			bool sleep = true;
			while (sleep)
			{
				auto now = std::chrono::system_clock::now();
				auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
				if (elapsed.count() > sleepTime)
					sleep = false;
			}
			*/
			std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
		}
		END_LOG_BLOCK(std::string("sleep"));
		END_LOG_FUNCTION(std::string("Processing() ") + std::to_string(decoder->getFrameIndex()) + std::string(" frame"));
	}
	return sts;
}

std::mutex syncDecoded;
std::mutex syncRGB;
at::Tensor getFrame(std::string consumerName, int index) {
	AVFrame* decoded;
	AVFrame* rgbFrame;
	at::Tensor outputTensor;
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
		rgbFrame = findFree<AVFrame*>(consumerName, rgbFrameArr);
	}
	END_LOG_BLOCK(std::string("findFree converted frame"));
	int indexFrame = REPEAT;
	START_LOG_BLOCK(std::string("decoder->GetFrame"));
	while (indexFrame == REPEAT) {
		indexFrame = decoder->GetFrame(index, consumerName, decoded);
	}
	END_LOG_BLOCK(std::string("decoder->GetFrame"));
	START_LOG_BLOCK(std::string("vpp->Convert"));
	VPPParameters VPPArgs = { 0, 0, NV12 };
	vpp->Convert(decoded, rgbFrame, VPPArgs, consumerName);
	END_LOG_BLOCK(std::string("vpp->Convert"));
	START_LOG_BLOCK(std::string("tensor->ConvertFromBlob"));
	int channelsNumber = 1;
	if (rgbFrame->format == AV_PIX_FMT_RGB24)
		channelsNumber = 3;
	outputTensor = torch::CUDA(at::kByte).tensorFromBlob(reinterpret_cast<void*>(rgbFrame->opaque), { rgbFrame->height, rgbFrame->width, channelsNumber});
	END_LOG_BLOCK(std::string("tensor->ConvertFromBlob"));
	/*
	Store tensor to be able get count of references for further releasing CUDA memory if strong_refs = 1
	*/
	START_LOG_BLOCK(std::string("add tensor"));
	std::unique_lock<std::mutex> locker(freeSync);
	tensors.push_back(outputTensor);
	END_LOG_BLOCK(std::string("add tensor"));
	END_LOG_FUNCTION(std::string("GetFrame() ") + std::to_string(indexFrame) + std::string(" frame"));
	return outputTensor;
}

/*
Mode 1 - full close, mode 2 - soft close (for reset)
*/
void endProcessing(int mode = HARD) {
	parser->Close();
	decoder->Close();
	vpp->Close();
	
	if (mode == 1 && logsLevel > 0) {
		logsFile.close();
		shouldWork = false;
	}

	for (auto& item : rgbFrameArr)
		av_frame_free(&item.second);
	for (auto& item : decodedArr)
		av_frame_free(&item.second);
	decodedArr.clear();
	rgbFrameArr.clear();
	tensors.clear();
	delete parsed;
}

void enableLogs(int _logsLevel) {
	if (_logsLevel) {
		logsLevel = static_cast<LogsLevel>(_logsLevel);
		if (!logsFile.is_open() && _logsLevel > 0) {
			logsFile.open(logFileName);
		}
	}
}
auto dumpPyton = std::shared_ptr<FILE>(fopen("DumpPython.yuv", "wb+"));
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("init", [](std::string rtmp) -> int {
		return initPipeline(rtmp);
	});
	m.def("getPars", []() -> std::map<std::string, int> {
		return getInitializedParams();
	});
	m.def("start", [](void) {
		py::gil_scoped_release release;
		return startProcessing();
		});
	m.def("get", [](std::string name, int frame) {
		py::gil_scoped_release release;
		return getFrame(name, frame);
	});
	m.def("enableLogs", [](int logsLevel) {
		enableLogs(logsLevel);
	});

	m.def("close", [](int mode) {
		endProcessing(mode);
	});

	m.def("dump", [](at::Tensor image) {
		uint8_t *dataPython = (uint8_t*)image.data_ptr();
		for (uint32_t i = 0; i < image.size(0); i++) {
			fwrite(dataPython, image.size(1) * image.size(2), 1, dumpPyton.get());
			dataPython += image.size(1) * image.size(2);
		}
		fflush(dumpPyton.get());
	});
}



void get_cycle(std::string name) {
	for (int i = 0; i < 500; i++) {
		getFrame(name, 0);
	}

}

int main()
{
	enableLogs(-MEDIUM);
	//"rtmp://b.sportlevel.com/relay/pooltop"
	int sts = initPipeline("rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4");
	CHECK_STATUS(sts);
	std::thread pipeline(startProcessing);
	std::thread get(get_cycle, "first");
	//std::thread get2(get_cycle, "second");

	get.join();
	//get2.join();
	pipeline.join();
	endProcessing(HARD);
	return 0;
}
