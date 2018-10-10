#include "VideoProcessor.h"
#include "Common.h"

void SaveRGB24(AVFrame *avFrame, FILE* dump) {
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

void SaveY8() {

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
		dumpFrame = std::shared_ptr<FILE>(fopen("RGB24.yuv", "wb+"));
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
	//std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	if (format.dstFourCC == NV12) {
		output->width = input->width;
		output->height = input->height;
		output->format = AV_PIX_FMT_RGB24;
		if (enableDumps) {
			//allocate buffers
			sts = av_frame_get_buffer(output, 2);
			
			sts = NV12ToRGB24Dump(input, output, prop.maxThreadsPerBlock, &stream);
			clock_t tStart = clock();
			SaveRGB24(output, dumpFrame.get());
			printf("Save %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
			void* opaque = output->opaque;
			//deallocate buffers and all custom fields too..
			av_frame_unref(output);
			output->width = input->width;
			output->height = input->height;
			output->opaque = opaque;
		}
		else {
			//printf("Decoded to VPP %x\n", input->data);
			sts = NV12ToRGB24(input, output, prop.maxThreadsPerBlock, &stream);
		}
	}
	av_frame_unref(input);
	return sts;
}

void VideoProcessor::Close() {
	if (isClosed)
		return;
	if (enableDumps)
		fclose(dumpFrame.get());
	isClosed = true;
}
