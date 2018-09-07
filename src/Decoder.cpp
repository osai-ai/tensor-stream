#include "Decoder.h"
#include "Common.h"
#include <cuda_runtime.h>

extern "C" {
	#include <libavutil/hwcontext_cuda.h>
}

Decoder::Decoder() {

}

int Decoder::Init(DecoderParameters* input) {
	state = input;
	int sts;

	decoderContext = avcodec_alloc_context3(state->parser->getStreamHandle()->codec->codec);
	sts = avcodec_parameters_to_context(decoderContext, state->parser->getStreamHandle()->codecpar);
	CHECK_STATUS(sts);

	//CUDA device initialization
	deviceReference = av_hwdevice_ctx_alloc(av_hwdevice_find_type_by_name("cuda"));
	AVHWDeviceContext* deviceContext = (AVHWDeviceContext*) deviceReference->data;
	AVCUDADeviceContext *CUDAContext = (AVCUDADeviceContext*) deviceContext->hwctx;

	//Assign runtime CUDA context to ffmpeg decoder
	sts = cuCtxGetCurrent(&CUDAContext->cuda_ctx);
	CHECK_STATUS(sts);
	sts = av_hwdevice_ctx_init(deviceReference);
	CHECK_STATUS(sts);
	decoderContext->hw_device_ctx = av_buffer_ref(deviceReference);
	sts = avcodec_open2(decoderContext, state->parser->getStreamHandle()->codec->codec, NULL);
	CHECK_STATUS(sts);

	framesBuffer.resize(state->bufferDeep);

	if (state->enableDumps) {
		dumpFrame = std::shared_ptr<FILE>(fopen("NV12.yuv", "wb+"));
	}

	return sts;
}

void Decoder::Close() {
	av_buffer_unref(&deviceReference);
	avcodec_close(decoderContext);
	fclose(dumpFrame.get());
}

void saveNV12(AVFrame *avFrame, FILE* dump)
{
	uint32_t pitchY = avFrame->linesize[0];
	uint32_t pitchUV = avFrame->linesize[1];

	uint8_t *avY = avFrame->data[0];
	uint8_t *avUV = avFrame->data[1];

	for (uint32_t i = 0; i < avFrame->height; i++) {
		fwrite(avY, avFrame->width, 1, dump);
		avY += pitchY;
	}

	for (uint32_t i = 0; i < avFrame->height / 2; i++) {
		fwrite(avUV, avFrame->width, 1, dump);
		avUV += pitchUV;
	}
	fflush(dump);
}

int Decoder::GetFrame(int index, std::string consumerName, AVFrame* outputFrame) {
	if (consumerStatus[consumerName]) {
		consumerStatus[consumerName] = false;
		int allignedIndex = state->bufferDeep + index - 1;
		outputFrame = framesBuffer[allignedIndex];
	}
	else {
		//wait until available
		
		0;
	}
	return OK;
}

int Decoder::Decode(AVPacket* pkt) {
	int sts = OK;
	avcodec_send_packet(decoderContext, pkt);
	if (sts < 0 || sts == AVERROR(EAGAIN) || sts == AVERROR_EOF) {
		return sts;
	}
	AVFrame* decodedFrame = av_frame_alloc();
	while (sts >= 0) {
		sts = avcodec_receive_frame(decoderContext, decodedFrame);
		if (sts == AVERROR(EAGAIN) || sts == AVERROR_EOF) {
			av_frame_free(&decodedFrame);
			return sts;
		}
		currentFrame++;
		//NULL?
		if (framesBuffer[currentFrame % state->bufferDeep]) {
			av_frame_free(&framesBuffer[currentFrame % state->bufferDeep]);
		}
		framesBuffer[currentFrame % state->bufferDeep] = decodedFrame;

		
		//Frame changed, consumers can take it
		
		for (auto item : consumerStatus) {
			item.second = true;
		}

		if (state->enableDumps) {
			AVFrame* NV12Frame = av_frame_alloc();
			NV12Frame->format = AV_PIX_FMT_NV12;

			if (decodedFrame->format == AV_PIX_FMT_CUDA) {
				sts = av_hwframe_transfer_data(NV12Frame, decodedFrame, 0);
				if (sts < 0) {
					av_frame_unref(NV12Frame);
					return sts;
				}
			}

			sts = av_frame_copy_props(NV12Frame, decodedFrame);
			if (sts < 0) {
				av_frame_unref(NV12Frame);
				return sts;
			}
			saveNV12(NV12Frame, dumpFrame.get());
		}
	}
	av_packet_unref(pkt);

	return sts;
}