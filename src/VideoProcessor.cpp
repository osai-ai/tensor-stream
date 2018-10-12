#include "VideoProcessor.h"
#include "Common.h"

void SaveRGB24(AVFrame *avFrame, FILE* dump) {
	uint8_t *RGB = avFrame->data[0];
	for (uint32_t i = 0; i < avFrame->height; i++) {
		fwrite(RGB, avFrame->width * avFrame->channels, 1, dump);
		RGB += avFrame->channels * avFrame->width;
	}
	fflush(dump);
}

void SaveBGR24(AVFrame *avFrame, FILE* dump) {
	uint8_t *BGR = avFrame->data[0];
	for (uint32_t i = 0; i < avFrame->height; i++) {
		fwrite(BGR, avFrame->width * avFrame->channels, 1, dump);
		BGR += avFrame->channels * avFrame->width;
	}
	fflush(dump);
}

void SaveY800(AVFrame *avFrame, FILE* dump) {
	uint8_t *Y800 = avFrame->data[0];
	for (uint32_t i = 0; i < avFrame->height; i++) {
		fwrite(Y800, avFrame->width * avFrame->channels, 1, dump);
		Y800 += avFrame->channels * avFrame->width;
	}
	fflush(dump);
}

int VideoProcessor::Init(bool _enableDumps = false) {
	enableDumps = _enableDumps;

	cudaGetDeviceProperties(&prop, 0);
	for (int i = 0; i < maxConsumers; i++) {
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		streamArr.push_back(std::make_pair(std::string("empty"), stream));
	}
	
	if (enableDumps) {
		dumpFrame = std::shared_ptr<FILE>(fopen("Processed.yuv", "wb+"), std::fclose);
	}
	isClosed = false;
	return OK;
}

int VideoProcessor::Convert(AVFrame* input, AVFrame* output, VPPParameters& format, std::string consumerName) {
	/*
	Should decide which method call
	*/
	cudaStream_t stream;
	int sts = OK;
	{
		std::unique_lock<std::mutex> locker(streamSync);
		stream = findFree<cudaStream_t>(consumerName, streamArr);
	}
	output->width = input->width;
	output->height = input->height;
	switch (format.dstFourCC) {
	case (RGB24):
		{
			output->format = AV_PIX_FMT_RGB24;
			//this field is used only for audio so let's write number of planes there
			output->channels = 3;
			sts = NV12ToRGB24(input, output, prop.maxThreadsPerBlock, &stream);
			if (enableDumps) {
				//allocate buffers
				std::shared_ptr<uint8_t> rawData(new uint8_t[output->channels * output->width * output->height], std::default_delete<uint8_t[]>());
				output->data[0] = rawData.get();
				cudaMemcpy(output->data[0], output->opaque, output->channels * output->width * output->height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				SaveRGB24(output, dumpFrame.get());
			}
			break;
		}
	case (BGR24):
		{
			output->format = AV_PIX_FMT_BGR24;
			//this field is used only for audio so let's write number of planes there
			output->channels = 3;
			sts = NV12ToBGR24(input, output, prop.maxThreadsPerBlock, &stream);
			if (enableDumps) {
				//allocate buffers
				std::shared_ptr<uint8_t> rawData(new uint8_t[output->channels * output->width * output->height], std::default_delete<uint8_t[]>());
				output->data[0] = rawData.get();
				cudaMemcpy(output->data[0], output->opaque, output->channels * output->width * output->height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				SaveBGR24(output, dumpFrame.get());
			}
			break;
		}
	case (Y800):
		{
			output->format = AV_PIX_FMT_GRAY8;
			output->channels = 1;
			//NV12 has one plane with Y only component, so need just copy first plane to video memory
			cudaMemcpy(output->opaque, input->data[0], output->channels * output->width * output->height * sizeof(unsigned char), cudaMemcpyHostToDevice);
			if (enableDumps) {
				//allocate buffers
				std::shared_ptr<uint8_t> rawData(new uint8_t[output->channels * output->width * output->height], std::default_delete<uint8_t[]>());
				output->data[0] = rawData.get();
				cudaMemcpy(output->data[0], output->opaque, output->channels * output->width * output->height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				SaveY800(output, dumpFrame.get());
			}
			break;
		}
	}
	av_frame_unref(input);
	return sts;
}

void VideoProcessor::Close() {
	if (isClosed)
		return;
	isClosed = true;
}
