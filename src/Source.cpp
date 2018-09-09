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


void SaveNV12(AVFrame *avFrame)
{
	uint32_t pitchY = avFrame->linesize[0];
	uint32_t pitchUV = avFrame->linesize[1];

	uint8_t *avY = avFrame->data[0];
	uint8_t *avUV = avFrame->data[1];

	for (uint32_t i = 0; i < avFrame->height; i++) {
		fwrite(avY, avFrame->width, 1, fDump);
		avY += pitchY;
	}

	for (uint32_t i = 0; i < avFrame->height / 2; i++) {
		fwrite(avUV, avFrame->width, 1, fDump);
		avUV += pitchUV;
	}
	fflush(fDump);
}

void SaveRGB24(AVFrame *avFrame)
{
	uint8_t *RGB = avFrame->data[0];
	for (uint32_t i = 0; i < avFrame->height; i++) {
		fwrite(RGB, avFrame->width * 3, 1, fDumpRGB);
		RGB += avFrame->linesize[0];
	}
	fflush(fDumpRGB);

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
void start() {
	/*avoiding Tensor CUDA lazy initializing for further context attaching*/
	at::Tensor gt_target = at::empty(at::CUDA(at::kByte), { 1 });
	parser = std::make_shared<Parser>();
	decoder = std::make_shared<Decoder>();
	vpp = std::make_shared<VideoProcessor>();
	ParserParameters parserArgs = { "rtmp://184.72.239.149/vod/mp4:bigbuckbunny_1500.mp4" , true };
	int sts = parser->Init(&parserArgs);
	DecoderParameters decoderArgs = { parser, true };
	sts = decoder->Init(&decoderArgs);
	VPPParameters VPPArgs = { 0, 0, NV12, true };
	sts = vpp->Init(&VPPArgs);
	AVPacket* parsed = new AVPacket();
	AVFrame* decoded = nullptr;
	AVFrame* rgbFrame = av_frame_alloc();
	for (int i = 0; i < 2500; i++) {
		sts = parser->Read();
		parser->Get(parsed);
		sts = decoder->Decode(parsed);
		//Need more data for decoding
		if (sts == AVERROR(EAGAIN) || sts == AVERROR_EOF)
			continue;
		decoder->GetFrame(0, "main", &decoded);
		vpp->Convert(decoded, rgbFrame);
		//application should free memory once it's not needed
		sts = cudaFree(rgbFrame->opaque);
		printf("Frame number %d\n", i);
	}
	parser->Close();
	decoder->Close();
	vpp->Close();
	av_frame_free(&rgbFrame);
	delete parsed;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("load", &load, "load data");
	m.def("get", &start, "get data");
}

int main()
{
	start();
	return 0;
}
