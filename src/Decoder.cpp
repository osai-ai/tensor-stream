#include "Decoder.h"
#include <cuda_runtime.h>

extern "C" {
	#include <libavutil/hwcontext_cuda.h>
}

Decoder::Decoder() {

}

int Decoder::InitSW(DecoderParameters& input, std::shared_ptr<Logger> logger) {
	PUSH_RANGE("Decoder::Init", NVTXColors::RED);
	state = input;
	int sts;
	this->logger = logger;
	decoderContext = avcodec_alloc_context3(state.parser->getStreamHandle()->codec->codec);
	sts = avcodec_parameters_to_context(decoderContext, state.parser->getStreamHandle()->codecpar);
	sts = avcodec_open2(decoderContext, state.parser->getStreamHandle()->codec->codec, NULL);
	CHECK_STATUS(sts);

	framesBuffer.resize(state.bufferDeep);

	if (state.enableDumps) {
		dumpFrame = std::shared_ptr<FILE>(fopen("NV12.yuv", "wb+"), std::fclose);
	}

	isClosed = false;
	return sts;
}

int Decoder::Init(DecoderParameters& input, std::shared_ptr<Logger> logger) {
	PUSH_RANGE("Decoder::Init", NVTXColors::RED);
	state = input;
	int sts;
	this->logger = logger;
	decoderContext = avcodec_alloc_context3(state.parser->getStreamHandle()->codec->codec);
	sts = avcodec_parameters_to_context(decoderContext, state.parser->getStreamHandle()->codecpar);
	CHECK_STATUS(sts);
	sts = cudaFree(0);
	CHECK_STATUS(sts);
	//CUDA device initialization
	deviceReference = av_hwdevice_ctx_alloc(av_hwdevice_find_type_by_name("cuda"));
	AVHWDeviceContext* deviceContext = (AVHWDeviceContext*) deviceReference->data;
	AVCUDADeviceContext *CUDAContext = (AVCUDADeviceContext*) deviceContext->hwctx;

	//Assign runtime CUDA context to ffmpeg decoder
	sts = cuCtxGetCurrent(&CUDAContext->cuda_ctx);
	CHECK_STATUS(CUDAContext->cuda_ctx == nullptr);
	CHECK_STATUS(sts);
	sts = av_hwdevice_ctx_init(deviceReference);
	CHECK_STATUS(sts);
	decoderContext->hw_device_ctx = av_buffer_ref(deviceReference);
	sts = avcodec_open2(decoderContext, state.parser->getStreamHandle()->codec->codec, NULL);
	CHECK_STATUS(sts);

	framesBuffer.resize(state.bufferDeep);

	if (state.enableDumps) {
		dumpFrame = std::shared_ptr<FILE>(fopen("NV12.yuv", "wb+"), std::fclose);
	}

	isClosed = false;
	return sts;
}

void Decoder::Close() {
	PUSH_RANGE("Decoder::Close", NVTXColors::RED);
	if (isClosed)
		return;
	av_buffer_unref(&deviceReference);
	avcodec_close(decoderContext);
	for (auto item : framesBuffer) {
		if (item != nullptr)
			av_frame_free(&item);
	}
	framesBuffer.clear();
	isClosed = true;
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

int Decoder::notifyConsumers() {
	{
		std::unique_lock<std::mutex> locker(sync);
		for (auto &item : consumerStatus) {
			item.second = true;
		}
		isFinished = true;
		consumerSync.notify_all();
	}
	return VREADER_OK;
}

AVCodecContext* Decoder::getDecoderContext() {
	return decoderContext;
}

int Decoder::GetFrame(int index, std::string consumerName, AVFrame* outputFrame) {
	PUSH_RANGE("Decoder::GetFrame", NVTXColors::RED);
	//element in map will be created after trying to call it
	if (!consumerStatus[consumerName]) {
		consumerStatus[consumerName] = false;
	}
	
	{
		std::unique_lock<std::mutex> locker(sync);
		if (isFinished == false)
			while (!consumerStatus[consumerName]) 
				consumerSync.wait(locker);

		if (isFinished)
			throw std::runtime_error("Decoding finished");

		if (consumerStatus[consumerName] == true) {
			consumerStatus[consumerName] = false;
			if (index > 0) {
				LOG_VALUE(std::string("WARNING: Frame number is greater than zero: ") + std::to_string(index), LogsLevel::LOW);
				index = 0;
			}
			int allignedIndex = (currentFrame - 1) % state.bufferDeep + index;
			if (allignedIndex < 0 || !framesBuffer[allignedIndex]) {
				return VREADER_REPEAT;
			}
			//can decoder overrun us and start using the same frame? Need sync
			av_frame_ref(outputFrame, framesBuffer[allignedIndex]);
		}
	}
	return currentFrame;
}

int Decoder::Decode(AVPacket* pkt) {
	PUSH_RANGE("Decoder::Decode", NVTXColors::RED);
	int sts = VREADER_OK;
	sts = avcodec_send_packet(decoderContext, pkt);
	if (sts < 0 || sts == AVERROR(EAGAIN) || sts == AVERROR_EOF) {
		return sts;
	}
	AVFrame* decodedFrame = av_frame_alloc();
	sts = avcodec_receive_frame(decoderContext, decodedFrame);

	if (sts == AVERROR(EAGAIN) || sts == AVERROR_EOF) {
		av_frame_free(&decodedFrame);
		return sts;
	}
	//deallocate copy(!) of packet from Reader
	av_packet_unref(pkt);
	{
		std::unique_lock<std::mutex> locker(sync);
		if (framesBuffer[(currentFrame) % state.bufferDeep]) {
			av_frame_free(&framesBuffer[(currentFrame) % state.bufferDeep]);
		}
		framesBuffer[(currentFrame) % state.bufferDeep] = decodedFrame;
		//Frame changed, consumers can take it
		currentFrame++;

		for (auto &item : consumerStatus) {
			item.second = true;
		}
		consumerSync.notify_all();
	}
	if (state.enableDumps) {
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
		av_frame_unref(NV12Frame);
	}
	return sts;
}

unsigned int Decoder::getFrameIndex() {
	return currentFrame;
}
