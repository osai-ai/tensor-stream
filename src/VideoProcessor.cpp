#include "VideoProcessor.h"
#include "Common.h"

template <class T>
void saveFrame(T* frame, FrameParameters options, FILE* dump) {
	int channels = 3;
	if (options.color.dstFourCC == Y800)
		channels = 1;
	if (options.color.dstFourCC == UYVY)
		channels = 2;
	//allow dump Y, RGB, BGR
	fwrite(frame, options.resize.width * options.resize.height * channels, sizeof(T), dump);
	
	//UV planes should be stored after Y without any strides
	if (options.color.dstFourCC == AV_PIX_FMT_NV12) {
		fwrite(&frame, options.resize.width * options.resize.height / 2 * channels, sizeof(T), dump);
	}
	
	fflush(dump);
}

template <class T>
int VideoProcessor::DumpFrame(T* output, FrameParameters options, std::shared_ptr<FILE> dumpFile) {
	PUSH_RANGE("VideoProcessor::DumpFrame", NVTXColors::YELLOW);
	int channels = 3;
	if (options.color.dstFourCC == Y800)
		channels = 1;
	if (options.color.dstFourCC == UYVY)
		channels = 2;
	//allocate buffers
	std::shared_ptr<T> rawData = std::shared_ptr<T>(new T[channels * options.resize.width * options.resize.height], std::default_delete<T[]>());
	cudaError err = cudaMemcpy(rawData.get(), output, channels * options.resize.width * options.resize.height * sizeof(T), cudaMemcpyDeviceToHost);
	CHECK_STATUS(err);
	saveFrame(rawData.get(), options, dumpFile.get());
	return VREADER_OK;
}

int VideoProcessor::Init(std::shared_ptr<Logger> logger, uint8_t maxConsumers, bool _enableDumps) {
	PUSH_RANGE("VideoProcessor::Init", NVTXColors::YELLOW);
	enableDumps = _enableDumps;
	this->logger = logger;
	cudaGetDeviceProperties(&prop, 0);
	for (int i = 0; i < maxConsumers; i++) {
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		streamArr.push_back(std::make_pair(std::string("empty"), stream));
	}
	
	isClosed = false;
	return VREADER_OK;
}

int VideoProcessor::Convert(AVFrame* input, AVFrame* output, FrameParameters options, std::string consumerName) {
	PUSH_RANGE("VideoProcessor::Convert", NVTXColors::YELLOW);
	/*
	Should decide which method call
	*/
	cudaStream_t stream;
	int sts = VREADER_OK;
	{
		std::unique_lock<std::mutex> locker(streamSync);
		stream = findFree<cudaStream_t>(consumerName, streamArr);
		if (stream == nullptr) {
			CHECK_STATUS(VREADER_ERROR);
		}
	}

	output->width = options.resize.width;
	output->height = options.resize.height;
	bool resize = false;
	if (output->width && output->height && (input->width != output->width || input->height != output->height)) {
		resize = true;
		resizeKernel(input, output, options.resize.type, prop.maxThreadsPerBlock, &stream);
	}
	else if (output->width == 0 || output->height == 0) {
		output->width = options.resize.width = input->width;
		output->height = options.resize.height = input->height;
	}

	switch (options.color.dstFourCC) {
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
	case UYVY:
		output->format = AV_PIX_FMT_UYVY422;
		output->channels = 2;
		break;
	case YUV444:
		output->format = AV_PIX_FMT_YUV444P;
		output->channels = 3;
	}

	if (options.color.normalization)
		sts = colorConversionKernel<float>(resize ? output : input, output, options.color, prop.maxThreadsPerBlock, &stream);
	else
		sts = colorConversionKernel<unsigned char>(resize ? output : input, output, options.color, prop.maxThreadsPerBlock, &stream);
	
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
			if (options.color.normalization)
				DumpFrame(static_cast<float*>(output->opaque), options, dumpFile);
			else
				DumpFrame(static_cast<unsigned char*>(output->opaque), options, dumpFile);
		}
	}
	av_frame_unref(input);
	return sts;
}

void VideoProcessor::Close() {
	PUSH_RANGE("VideoProcessor::Close", NVTXColors::YELLOW);
	if (isClosed)
		return;
	isClosed = true;
}
