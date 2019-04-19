#include "VideoProcessor.h"
#include "Common.h"

void saveFrame(AVFrame *avFrame, FILE* dump) {
	
	uint8_t *frame = avFrame->data[0];
	for (uint32_t i = 0; i < avFrame->height; i++) {
		int sts = fwrite(frame, avFrame->width * avFrame->channels, 1, dump);
		if (avFrame->linesize[0])
			frame += avFrame->channels * avFrame->linesize[0];
		else
			frame += avFrame->channels * avFrame->width;
	}
	
	if (avFrame->format == AV_PIX_FMT_NV12) {
		uint8_t *frame = avFrame->data[1];
		for (uint32_t i = 0; i < avFrame->height / 2; i++) {
			fwrite(frame, avFrame->width * avFrame->channels, 1, dump);
			if (avFrame->linesize[1])
				frame += avFrame->channels * avFrame->linesize[1];
			else
				frame += avFrame->channels * avFrame->width;
		}
	}
	
	fflush(dump);
}

int VideoProcessor::DumpFrame(AVFrame* output, std::shared_ptr<FILE> dumpFile) {
	//allocate buffers
	std::shared_ptr<uint8_t> rawData(new uint8_t[output->channels * output->width * output->height], std::default_delete<uint8_t[]>());
	output->data[0] = rawData.get();
	cudaError err = cudaMemcpy(output->data[0], output->opaque, output->channels * output->width * output->height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	CHECK_STATUS(err);
	saveFrame(output, dumpFile.get());
	return VREADER_OK;
}

int VideoProcessor::Init(bool _enableDumps) {
	enableDumps = _enableDumps;

	cudaGetDeviceProperties(&prop, 0);
	for (int i = 0; i < maxConsumers; i++) {
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		streamArr.push_back(std::make_pair(std::string("empty"), stream));
	}
	
	isClosed = false;
	return VREADER_OK;
}

int VideoProcessor::Convert(AVFrame* input, AVFrame* output, VPPParameters& format, std::string consumerName) {
	/*
	Should decide which method call
	*/
	cudaStream_t stream;
	int sts = VREADER_OK;
	{
		std::unique_lock<std::mutex> locker(streamSync);
		stream = findFree<cudaStream_t>(consumerName, streamArr);
	}

	output->width = format.width;
	output->height = format.height;
	bool resize = false;
	if (output->width && output->height && (input->width != output->width || input->height != output->height)) {
		resize = true;
		resizeKernel(input, output, ResizeType::NEAREST, prop.maxThreadsPerBlock, &stream);
	}
	else if (output->width == 0 || output->height == 0) {
		output->width = input->width;
		output->height = input->height;
	}


	sts = colorConversionKernel(resize ? output : input, output, format.color, prop.maxThreadsPerBlock, &stream);
	
	if (resize) {
		//need to free allocated in resize memory for Y and UV
		cudaError err = cudaFree(output->data[0]);
		CHECK_STATUS(err);
		err = cudaFree(output->data[1]);
		CHECK_STATUS(err);
	}
	if (enableDumps) {
		std::string fileName = std::string("Processed_") + consumerName + std::string(".yuv");
		std::shared_ptr<FILE> dumpFile(std::shared_ptr<FILE>(fopen(fileName.c_str(), "ab"), std::fclose));
		{
			//avoid situations when several threads write to IO (some possible collisions can be observed)
			std::unique_lock<std::mutex> locker(dumpSync);
			DumpFrame(output, dumpFile);
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
