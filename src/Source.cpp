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

void kernel_wrapper(int *a);
unsigned char* change_pixels(AVFrame* src, AVFrame* dst,  CUstream stream);
void test_python(float* test);

FILE* fDump;
FILE* fDumpRGB;
unsigned char* RGB;
static enum AVPixelFormat get_hw_format(AVCodecContext *ctx,
	const enum AVPixelFormat *pix_fmts)
{
	const enum AVPixelFormat *p;

	for (p = pix_fmts; *p != -1; p++) {
		if (*p == AV_PIX_FMT_CUDA)
			return *p;
	}

	fprintf(stderr, "Failed to get HW surface format.\n");
	return AV_PIX_FMT_NONE;
}


void printContext() {
	CUcontext test_cu;
	auto cu_err = cuCtxGetCurrent(&test_cu);
	printf("Context %x\n", test_cu);
}

at::Tensor load() {
	//CUdeviceptr data_done;
	//cudaError_t err2 = cudaMalloc(reinterpret_cast<void**>(&data_done), 16 * sizeof(float));
	printContext();
	at::Tensor f = torch::CUDA(at::kByte).tensorFromBlob(reinterpret_cast<void*>(RGB), { 3264 * 608 });
	return f;
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
	DecoderParameters decoderArgs = {parser, true };
	sts = decoder->Init(decoderArgs);
	VPPParameters VPPArgs = { 0, 0, NV12, false };
	sts = vpp->Init(VPPArgs);
	parsed = new AVPacket();
	for (int i = 0; i < maxConsumers; i++) {
		decodedArr.push_back(std::make_pair(std::string("empty"), av_frame_alloc()));
		rgbFrameArr.push_back(std::make_pair(std::string("empty"), av_frame_alloc()));
	}
}

void start() {
	int sts = OK;
	//change to end of file
	while(true) {
		clock_t tStart = clock();
		sts = parser->Read();
		//TODO: expect this behavior only in case of EOF
		if (sts < 0)
			break;
		parser->Get(parsed);
		sts = decoder->Decode(parsed);
		//Need more data for decoding
		if (sts == AVERROR(EAGAIN) || sts == AVERROR_EOF)
			continue;
		//printf("Time taken for Decode: %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
		//printf("Frame number %d\n", decoder->getFrameIndex());
		if (decoder->getFrameIndex() == 110)
			return;
	}
}
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
std::mutex syncDecoded;
std::mutex syncRGB;
void get(std::string consumerName) {
	AVFrame* decoded;
	AVFrame* rgbFrame;
	clock_t tStart = clock();
	{
		std::unique_lock<std::mutex> locker(syncDecoded);
		decoded = findFree(consumerName, decodedArr);
	}
	{
		std::unique_lock<std::mutex> locker(syncRGB);
		rgbFrame = findFree(consumerName, rgbFrameArr);
	}
	printf("Vectors %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	tStart = clock();
	decoder->GetFrame(0, consumerName, decoded);
	printf("Get %f\n",(double)(clock() - tStart) / CLOCKS_PER_SEC);
	//printf("Time taken for Get: %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	tStart = clock();
	vpp->Convert(decoded, rgbFrame, consumerName);
	printf("Convert %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	//printf("Time taken for convert: %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	//application should free memory once it's not needed
	//tStart = clock();
	//cudaFree(rgbFrame->opaque);
	//printf("Time taken for Free: %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}

void get_cycle(std::string name) {
	for (int i = 0; i < 70; i++) {
		clock_t tStart = clock();
		get(name);
		printf("Got number %d, %f\n", i, (double)(clock() - tStart) / CLOCKS_PER_SEC);
	}

}

void close() {
	parser->Close();
	decoder->Close();
	vpp->Close();
	for (auto& item : rgbFrameArr)
		av_frame_free(&item.second);
	for (auto& item : decodedArr)
		av_frame_free(&item.second);
	delete parsed;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("load", &load, "load data");
	m.def("get", &start, "get data");
}

int main()
{
	std::cout << "Main thread: " << std::this_thread::get_id() << std::endl;
	initPipeline();
	std::thread pipeline(start);
	std::thread get(get_cycle, "first");
	//std::thread get2(get_cycle, "second");

	get.join();
	//get2.join();
	pipeline.join();
	printf("Before close\n");
	close();
	printf("After close\n");
	return 0;
}
