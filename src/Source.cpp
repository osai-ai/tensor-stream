#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
// basic file operations
#include <iostream>
#include <fstream>
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
#include "memory"
#include "Parser.h"
#include "Decoder.h"
#include "Common.h"
#include "VideoProcessor.h"
#include <Python.h>
extern "C"
{
#include <libavformat/avformat.h>
#include <libavutil/timestamp.h>
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/hwcontext_cuda.h>
}
#include <thread>
#include <time.h>
#include <chrono>


AVFrame* findFree(std::string consumerName, std::vector<std::pair<std::string, AVFrame*> >& frames) {
	for (auto& item : frames) {
		if (item.first == consumerName) {
			return item.second;
		}
		else if (item.first == "empty") {
			item.first = consumerName;
			return item.second;
		}
	}
	return nullptr;
}


std::shared_ptr<Parser> parser;
std::shared_ptr<Decoder> decoder;
std::shared_ptr<VideoProcessor> vpp;
AVPacket* parsed;
int realTimeDelay;

std::vector<std::pair<std::string, AVFrame*> > decodedArr;
std::vector<std::pair<std::string, AVFrame*> > rgbFrameArr;
void initPipeline() {
	/*avoiding Tensor CUDA lazy initializing for further context attaching*/
	at::Tensor gt_target = at::empty(at::CUDA(at::kByte), { 1 });
	parser = std::make_shared<Parser>();
	decoder = std::make_shared<Decoder>();
	vpp = std::make_shared<VideoProcessor>();
	ParserParameters parserArgs = { "rtmp://b.sportlevel.com/relay/pooltop"/*"rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4" */ , false };
	int sts = parser->Init(parserArgs);
	DecoderParameters decoderArgs = { parser, false };
	sts = decoder->Init(decoderArgs);
	sts = vpp->Init(true);
	parsed = new AVPacket();
	for (int i = 0; i < maxConsumers; i++) {
		decodedArr.push_back(std::make_pair(std::string("empty"), av_frame_alloc()));
		rgbFrameArr.push_back(std::make_pair(std::string("empty"), av_frame_alloc()));
	}
	realTimeDelay = ((float)parser->getFormatContext()->streams[parser->getVideoIndex()]->codec->framerate.den /
		(float)parser->getFormatContext()->streams[parser->getVideoIndex()]->codec->framerate.num) * 1000;
}

template <typename T>
using duration = std::chrono::duration<T>;

static void sleep_for(double dt)
{
	static constexpr duration<double> MinSleepDuration(0);
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	auto test = duration<double>(std::chrono::high_resolution_clock::now() - start).count();
	while (duration<double>(std::chrono::high_resolution_clock::now() - start).count() < dt) {
		std::this_thread::sleep_for(MinSleepDuration);
	}
}

void startProcessing() {
	int sts = OK;
#ifdef TIMINGS
	int exceptionCount = 0;
	std::vector<int> exceptions;
	std::vector<int> sleepsTime;
	std::vector<int> needSleep;
	std::vector<int> decodeTime;
	std::vector<int> readTime;
	std::vector<int> getTime;
#endif
	//change to end of file
	while (true) {
#ifdef TIMINGS
		clock_t tStart = clock();
#endif
#ifdef TRACER
		nvtxNameOsThread(GetCurrentThreadId(), "DECODE_THREAD");
		nvtxRangePush("Read frame");
		nvtxMark("Reading");
#endif
#ifdef TIMINGS
		clock_t readStart = clock();
#endif
		sts = parser->Read();
#ifdef TIMINGS
		clock_t readEnd = clock();
#endif
#ifdef TRACER
		nvtxRangePop();
#endif
		if (sts == AVERROR(EAGAIN))
			continue;
		//TODO: expect this behavior only in case of EOF
		if (sts < 0)
			break;
#ifdef TIMINGS
		clock_t getStart = clock();
#endif
		parser->Get(parsed);
#ifdef TIMINGS
		clock_t getEnd = clock();
#endif
#ifdef TRACER
		nvtxNameOsThread(GetCurrentThreadId(), "DECODE_THREAD");
		nvtxRangePush("Decode");
		nvtxMark("Decoding");
#endif
#ifdef TIMINGS
		clock_t decodeStart = clock();
#endif
		sts = decoder->Decode(parsed);
#ifdef TIMINGS
		clock_t decodeEnd = clock();
		printf("Time taken for Decode: %f\n", (double)(decodeEnd - decodeStart) / CLOCKS_PER_SEC);
#endif
#ifdef TRACER
		nvtxRangePop();
#endif

		//Need more data for decoding
		if (sts == AVERROR(EAGAIN) || sts == AVERROR_EOF)
			continue;
		
		//wait here
		int sleepTime = realTimeDelay - (clock() - tStart);
		//printf("sleepTime %d\n", sleepTime);
#ifdef TIMINGS
		clock_t sleepStart = clock();
#endif
		if (sleepTime > 0) {
			auto start = std::chrono::system_clock::now();
			bool sleep = true;
			while (sleep)
			{
				auto now = std::chrono::system_clock::now();
				auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
				if (elapsed.count() > sleepTime)
					sleep = false;
			}
			//std::this_thread::sleep_for(std::chrono::milliseconds(x));
		}
#ifdef TIMINGS
		clock_t sleepEnd = clock();
		printf("Time taken for Sleep: %f\n", (double)(sleepEnd - sleepStart) / CLOCKS_PER_SEC);
#endif
#ifdef TIMINGS
		clock_t tEnd = clock();
		printf("Time taken for Pipeline: %f\n", (double)(tEnd - tStart) / CLOCKS_PER_SEC);
#endif
		
#ifdef TIMINGS
		if (tEnd - tStart > 19 && decoder->getFrameIndex() > 1) {
			exceptionCount++;
			exceptions.push_back(tEnd - tStart);
			sleepsTime.push_back(sleepEnd - sleepStart);
			decodeTime.push_back(decodeEnd - decodeStart);
			needSleep.push_back(sleepTime);
			readTime.push_back(readEnd - readEnd);
			getTime.push_back(getEnd - getEnd);
		}
#endif
		printf("Frame number %d\n", decoder->getFrameIndex());
		if (decoder->getFrameIndex() == 5000) {
#ifdef TIMINGS
			printf("Ahtung count %d\n", exceptionCount);
			for (auto item : exceptions)
				printf("ahtung time %d\n", item);
			for (auto item : sleepsTime)
				printf("ahtung time sleep %d\n", item);
			for (auto item : decodeTime)
				printf("ahtung time decode %d\n", item);
			for (auto item : readTime)
				printf("ahtung read time %d\n", item);
			for (auto item : getTime)
				printf("ahtung get time %d\n", item);
			for (auto item : needSleep)
				printf("ahtung need sleep time %d\n", item);
#endif
			return;
		}
	}
}

std::mutex syncDecoded;
std::mutex syncRGB;
at::Tensor getFrame(std::string consumerName, int index) {
	AVFrame* decoded;
	AVFrame* rgbFrame;
	//clock_t tStart = clock();
	{
		std::unique_lock<std::mutex> locker(syncDecoded);
		decoded = findFree(consumerName, decodedArr);
	}
	{
		std::unique_lock<std::mutex> locker(syncRGB);
		rgbFrame = findFree(consumerName, rgbFrameArr);
	}
	//printf("Vectors %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	//tStart = clock();
	int sts = REPEAT;
	while (sts != OK) {
		sts = decoder->GetFrame(index, consumerName, decoded);
	}
	//printf("Get %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	//printf("Time taken for Get: %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	//tStart = clock();
	VPPParameters VPPArgs = { 0, 0, NV12 };
	vpp->Convert(decoded, rgbFrame, VPPArgs, consumerName);
	//printf("Convert %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	//tStart = clock();
	at::Tensor outputTensor = torch::CUDA(at::kByte).tensorFromBlob(reinterpret_cast<void*>(rgbFrame->opaque), { rgbFrame->width * rgbFrame->height });
	//printf("To tensor %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	//printf("Time taken for convert: %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	//application should free memory once it's not needed
	//tStart = clock();
	cudaFree(rgbFrame->opaque);
	return outputTensor;
	//printf("Time taken for Free: %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}

void endProcessing() {
	parser->Close();
	decoder->Close();
	vpp->Close();
	for (auto& item : rgbFrameArr)
		av_frame_free(&item.second);
	for (auto& item : decodedArr)
		av_frame_free(&item.second);
	delete parsed;
}


void startThread() {
	Py_BEGIN_ALLOW_THREADS
	std::thread pipeline(startProcessing);
	Py_END_ALLOW_THREADS
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("init", &initPipeline, "Init pipeline");
	m.def("start", &startProcessing, "Start decoding");
	m.def("startThread", &startThread, "Start decoding");
	m.def("get", &getFrame, "Get frame by index");
	m.def("close", &endProcessing, "Close session");
}



void get_cycle(std::string name) {
	for (int i = 0; i < 5000; i++) {
		clock_t tStart = clock();
		getFrame(name, 0);
		printf("Got number %d, %f\n", i, (double)(clock() - tStart) / CLOCKS_PER_SEC);
	}

}

int main()
{
	std::cout << "Main thread: " << std::this_thread::get_id() << std::endl;
	initPipeline();
	std::thread pipeline(startProcessing);
	std::thread get(get_cycle, "first");
//	std::thread get2(get_cycle, "second");

	get.join();
//	get2.join();
	pipeline.join();
	printf("Before close\n");
	endProcessing();
	printf("After close\n");
	return 0;
}
