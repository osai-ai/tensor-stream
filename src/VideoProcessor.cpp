#include "VideoProcessor.h"
#include "Common.h"

void SaveRGB24(AVFrame *avFrame, FILE* dump)
{
#ifdef DEBUG_INFO
	int static count = 0;
#endif
	uint8_t *RGB = avFrame->data[0];
	for (uint32_t i = 0; i < avFrame->height; i++) {
		fwrite(RGB, avFrame->width * 3, 1, dump);
		RGB += 3 * avFrame->width;
	}
	fflush(dump);
#ifdef DEBUG_INFO
	count++;
	printf("RGB %d\n", count);
#endif
}

int VideoProcessor::Init(VPPParameters& outputFormat) {
	state = outputFormat;

	cudaGetDeviceProperties(&prop, 0);
	cudaStreamCreate(&stream);
	if (state.enableDumps) {
		dumpFrame = std::shared_ptr<FILE>(fopen("RGB24.yuv", "wb+"));
	}

	return OK;
}

int VideoProcessor::Convert(AVFrame* input, AVFrame* output) {
	/*
	Should decide which method call
	*/
	int sts = OK;
	//std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	if (state.dstFourCC == NV12) {
		output->width = input->width;
		output->height = input->height;
		output->format = AV_PIX_FMT_RGB24;
		if (state.enableDumps) {
			//allocate buffers
			sts = av_frame_get_buffer(output, 2);
			sts = NV12ToRGB24Dump(input, output, prop.maxThreadsPerBlock, &stream);
			SaveRGB24(output, dumpFrame.get());
			void* opaque = output->opaque;
			//deallocate buffers and all custom fields too..
			av_frame_unref(output);
			output->opaque = opaque;
		}
		else {
			printf("Decoded to VPP %x\n", input->data);
			sts = NV12ToRGB24(input, output, prop.maxThreadsPerBlock, &stream);
		}
	}
	av_frame_unref(input);
	return sts;
}

void VideoProcessor::Close() {
	if (state.enableDumps)
		fclose(dumpFrame.get());
}