#include "VideoProcessor.h"
#include "Common.h"

template<class T>
void saveFrame(T *frame, VPPParameters options, FILE* dump) {
	int channels = 3;
	if (options.color.dstFourCC == Y800)
		channels = 1;
	//allow dump Y, RGB, BGR
	for (uint32_t i = 0; i < options.height; i++) {
		int sts = fwrite(frame, options.width * channels, sizeof(T), dump);
		frame += channels * options.width;
	}
	
	if (options.color.dstFourCC == AV_PIX_FMT_NV12) {
		for (uint32_t i = 0; i < options.height / 2; i++) {
			fwrite(frame, options.width * channels, 1, dump);
			frame += channels * options.width;
		}
	}
	
	fflush(dump);
}

template <class T>
int VideoProcessor::DumpFrame(T* output, VPPParameters options, std::shared_ptr<FILE> dumpFile) {
	int channels = 3;
	if (options.color.dstFourCC == Y800)
		channels = 1;
	//allocate buffers
	std::shared_ptr<T> rawData(new T[channels * options.width * options.height], std::default_delete<T[]>());
	cudaError err = cudaMemcpy(rawData.get(), output, channels * options.width * options.height * sizeof(T), cudaMemcpyDeviceToHost);
	CHECK_STATUS(err);
	saveFrame(output, options, dumpFile.get());
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

	switch (format.color.dstFourCC) {
	case BGR24:
		output->format = AV_PIX_FMT_BGR24;
		output->channels = 3;
		break;
	case RGB24:
		output->format = AV_PIX_FMT_RGB24;
		output->channels = 3;
		break;
	case Y800:
		output->format = AV_PIX_FMT_GRAY8;
		output->channels = 1;
		break;
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
			//Set linesize to width?
			if (format.color.normalization)
				DumpFrame(static_cast<float*>(output->opaque), format, dumpFile);
			else
				DumpFrame(static_cast<uint8_t*>(output->opaque), format, dumpFile);
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
