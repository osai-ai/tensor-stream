#include "VideoProcessor.h"
#include "Common.h"

void saveFrame(float *frame, FrameParameters options, FILE* dump) {
	int channels = 3;
	if (options.color.dstFourCC != RGB24 && options.color.dstFourCC != BGR24)
		channels = 1;
	//allow dump Y, RGB, BGR
	if (!options.color.normalization) {
		//std::vector<uint8_t> value(frame, frame + options.resize.width * options.resize.height * channels);
		std::shared_ptr<uint8_t> value(new uint8_t[options.resize.width * options.resize.height * channels], std::default_delete<uint8_t[]>());
		std::copy(frame, frame + options.resize.width * options.resize.height * channels, value.get());
		fwrite(value.get(), options.resize.width * options.resize.height * channels, sizeof(unsigned char), dump);
	}
	else {
		fwrite(&frame, options.resize.width * options.resize.height * channels, sizeof(float), dump);
	}
	
	//UV planes should be stored after Y without any strides
	if (options.color.dstFourCC == AV_PIX_FMT_NV12) {
		if (!options.color.normalization) {
			std::shared_ptr<uint8_t> value(new uint8_t[options.resize.width * options.resize.height / 2 * channels], std::default_delete<uint8_t[]>());
			std::copy(frame, frame + options.resize.width * options.resize.height / 2 * channels, value.get());
			fwrite(value.get(), options.resize.width * options.resize.height / 2 * channels, sizeof(unsigned char), dump);
		}
		else {
			fwrite(&frame, options.resize.width * options.resize.height / 2 * channels, sizeof(float), dump);
		}
	}
	
	fflush(dump);
}

int VideoProcessor::DumpFrame(float* output, FrameParameters options, std::shared_ptr<FILE> dumpFile) {
	int channels = 3;
	if (options.color.dstFourCC == Y800)
		channels = 1;
	//allocate buffers
	std::shared_ptr<float> rawData;
	rawData = std::shared_ptr<float>(new float[channels * options.resize.width * options.resize.height], std::default_delete<float[]>());
	cudaError err = cudaMemcpy(rawData.get(), output, channels * options.resize.width * options.resize.height * sizeof(float), cudaMemcpyDeviceToHost);
	CHECK_STATUS(err);
	saveFrame(rawData.get(), options, dumpFile.get());
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

int VideoProcessor::Convert(AVFrame* input, AVFrame* output, FrameParameters options, std::string consumerName) {
	/*
	Should decide which method call
	*/
	cudaStream_t stream;
	int sts = VREADER_OK;
	{
		std::unique_lock<std::mutex> locker(streamSync);
		stream = findFree<cudaStream_t>(consumerName, streamArr);
	}

	output->width = options.resize.width;
	output->height = options.resize.height;
	bool resize = false;
	if (output->width && output->height && (input->width != output->width || input->height != output->height)) {
		resize = true;
		resizeKernel(input, output, options.resize.type, prop.maxThreadsPerBlock, &stream);
	}
	else if (output->width == 0 || output->height == 0) {
		output->width = input->width;
		output->height = input->height;
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
	}

	sts = colorConversionKernel(resize ? output : input, output, options.color, prop.maxThreadsPerBlock, &stream);
	
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
			DumpFrame(static_cast<float*>(output->opaque), options, dumpFile);
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
