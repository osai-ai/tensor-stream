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

std::vector<std::pair<std::string, AVFrame*> > decodedArr;
std::vector<std::pair<std::string, AVFrame*> > rgbFrameArr;
void initPipeline() {
	/*avoiding Tensor CUDA lazy initializing for further context attaching*/
	at::Tensor gt_target = at::empty(at::CUDA(at::kByte), { 1 });
	parser = std::make_shared<Parser>();
	decoder = std::make_shared<Decoder>();
	vpp = std::make_shared<VideoProcessor>();
	ParserParameters parserArgs = { "rtmp://b.sportlevel.com/relay/pooltop" /*"rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4"*/ , false };
	int sts = parser->Init(parserArgs);
	DecoderParameters decoderArgs = { parser, false };
	sts = decoder->Init(decoderArgs);
	sts = vpp->Init(false);
	parsed = new AVPacket();
	for (int i = 0; i < maxConsumers; i++) {
		decodedArr.push_back(std::make_pair(std::string("empty"), av_frame_alloc()));
		rgbFrameArr.push_back(std::make_pair(std::string("empty"), av_frame_alloc()));
	}
}

void startProcessing() {
	int sts = OK;
	//change to end of file
	while (true) {
		clock_t tStart = clock();
#ifdef TRACER
		nvtxNameOsThread(GetCurrentThreadId(), "DECODE_THREAD");
		nvtxRangePush("Read frame");
		nvtxMark("Reading");
#endif
		sts = parser->Read();
#ifdef TRACER
		nvtxRangePop();
#endif
		if (sts == AVERROR(EAGAIN))
			continue;
		//TODO: expect this behavior only in case of EOF
		if (sts < 0)
			break;
		parser->Get(parsed);
#ifdef TRACER
		nvtxNameOsThread(GetCurrentThreadId(), "DECODE_THREAD");
		nvtxRangePush("Decode");
		nvtxMark("Decoding");
#endif
		sts = decoder->Decode(parsed);
#ifdef TRACER
		nvtxRangePop();
#endif

		//Need more data for decoding
		if (sts == AVERROR(EAGAIN) || sts == AVERROR_EOF)
			continue;
		printf("Time taken for Decode: %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
		printf("Frame number %d\n", decoder->getFrameIndex());
		if (decoder->getFrameIndex() == 1100)
			return;
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
	decoder->GetFrame(index, consumerName, decoded);
	//printf("Get %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	//printf("Time taken for Get: %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	//tStart = clock();
	VPPParameters VPPArgs = { 0, 0, NV12 };
	vpp->Convert(decoded, rgbFrame, VPPArgs, consumerName);
	//printf("Convert %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	//tStart = clock();
	at::Tensor outputTensor = torch::CUDA(at::kByte).tensorFromBlob(reinterpret_cast<void*>(rgbFrame->opaque), { rgbFrame->width * rgbFrame->height });
	//printf("To tensor %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	return outputTensor;
	//printf("Time taken for convert: %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	//application should free memory once it's not needed
	//tStart = clock();
	//cudaFree(rgbFrame->opaque);
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
	for (int i = 0; i < 70; i++) {
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
	std::thread get2(get_cycle, "second");

	get.join();
	get2.join();
	pipeline.join();
	printf("Before close\n");
	endProcessing();
	printf("After close\n");
	return 0;
}
