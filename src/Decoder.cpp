#include "Decoder.h"
#include "Common.h"
#include <cuda_runtime.h>

extern "C" {
	#include <libavutil/hwcontext_cuda.h>
}

int Decoder::Init(DecoderParameters& input) {
	state = input;
	int sts;

	decoderContext = avcodec_alloc_context3(state.parser->getStreamHandle()->codec->codec);
	sts = avcodec_parameters_to_context(decoderContext, state.parser->getStreamHandle()->codecpar);
	CHECK_STATUS(sts);
	/*
	CUDA device initialization
	*/
	deviceReference = av_hwdevice_ctx_alloc(av_hwdevice_find_type_by_name("cuda"));
	AVHWDeviceContext* deviceContext = (AVHWDeviceContext*) deviceReference->data;
	AVCUDADeviceContext *CUDAContext = (AVCUDADeviceContext*) deviceContext->hwctx;
	/*
	Assign runtime CUDA context to ffmpeg decoder
	*/
	sts = cuCtxGetCurrent(&CUDAContext->cuda_ctx);
	CHECK_STATUS(sts);
	sts = av_hwdevice_ctx_init(deviceReference);
	CHECK_STATUS(sts);
	decoderContext->hw_device_ctx = av_buffer_ref(deviceReference);
	sts = avcodec_open2(decoderContext, state.parser->getStreamHandle()->codec->codec, NULL);
	CHECK_STATUS(sts);

	if (state.enableDumps) {
		dumpFrame.insert(std::make_pair(std::string("NV12"), std::shared_ptr<FILE>(fopen("NV12.yuv", "wb+"))));
		dumpFrame.insert(std::make_pair(std::string("RGB"), std::shared_ptr<FILE>(fopen("RGB.yuv", "wb+"))));
	}

	return sts;
}

void Decoder::Close() {
	av_buffer_unref(&deviceReference);
	avcodec_close(decoderContext);
}