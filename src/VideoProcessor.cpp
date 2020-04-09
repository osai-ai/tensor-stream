#include "VideoProcessor.h"
#include "Common.h"

float channelsByFourCC(FourCC fourCC) {
	float channels = 3;
	if (fourCC == Y800)
		channels = 1;
	if (fourCC == UYVY)
		channels = 2;
	if (fourCC == NV12)
		channels = 1.5;
	
	return channels;
}

float channelsByFourCC(std::string fourCC) {
	float channels = 3;
	if (fourCC == "Y800")
		channels = 1;
	if (fourCC == "UYVY")
		channels = 2;
	if (fourCC == "NV12")
		channels = 1.5;

	return channels;
}

template <class T>
void saveFrame(T* frame, FrameParameters options, FILE* dump) {
	float channels = channelsByFourCC(options.color.dstFourCC);
	int dumpWidth = 0;
	int dumpHeight = 0;
	int cropWidth = (std::get<0>(options.crop.rightBottomCorner) - std::get<0>(options.crop.leftTopCorner));
	int cropHeight = (std::get<1>(options.crop.rightBottomCorner) - std::get<1>(options.crop.leftTopCorner));
	if (cropWidth > 0 && cropHeight > 0) {
		dumpWidth = cropWidth;
		dumpHeight = cropHeight;
	}
	if (options.resize.width > 0 && options.resize.height > 0) {
		dumpWidth = options.resize.width;
		dumpHeight = options.resize.height;
	}
	//allow dump Y, RGB, BGR
	fwrite(frame, (int) (dumpWidth * dumpHeight * channels), sizeof(T), dump);

	fflush(dump);
}

template <class T>
int VideoProcessor::DumpFrame(T* output, FrameParameters options, std::shared_ptr<FILE> dumpFile) {
	PUSH_RANGE("VideoProcessor::DumpFrame", NVTXColors::YELLOW);
	float channels = channelsByFourCC(options.color.dstFourCC);
	int dumpWidth = 0;
	int dumpHeight = 0;
	int cropWidth = (std::get<0>(options.crop.rightBottomCorner) - std::get<0>(options.crop.leftTopCorner));
	int cropHeight = (std::get<1>(options.crop.rightBottomCorner) - std::get<1>(options.crop.leftTopCorner));
	if (cropWidth > 0 && cropHeight > 0) {
		dumpWidth = cropWidth;
		dumpHeight = cropHeight;
	}
	if (options.resize.width > 0 && options.resize.height > 0) {
		dumpWidth = options.resize.width;
		dumpHeight = options.resize.height;
	}

	//allocate buffers
	std::shared_ptr<T> rawData = std::shared_ptr<T>(new T[(int)(channels * dumpWidth * dumpHeight)], std::default_delete<T[]>());
	cudaError err = cudaMemcpy(rawData.get(), output, channels * dumpWidth * dumpHeight * sizeof(T), cudaMemcpyDeviceToHost);
	CHECK_STATUS(err);
	saveFrame(rawData.get(), options, dumpFile.get());
	return VREADER_OK;
}

template
int VideoProcessor::DumpFrame(float* output, FrameParameters options, std::shared_ptr<FILE> dumpFile);
template
int VideoProcessor::DumpFrame(uint8_t* output, FrameParameters options, std::shared_ptr<FILE> dumpFile);

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

int VideoProcessor::Convert(AVFrame* input, AVFrame* output, FrameParameters& options, std::string consumerName) {
	PUSH_RANGE("VideoProcessor::Convert", NVTXColors::YELLOW);
	cudaStream_t stream;
	int sts = VREADER_OK;
	{
		std::unique_lock<std::mutex> locker(streamSync);
		stream = findFree<cudaStream_t>(consumerName, streamArr);
		if (stream == nullptr) {
			//will be used default stream
		}
	}
	std::cout << stream << std::endl;
	int cropWidth = std::get<0>(options.crop.rightBottomCorner) - std::get<0>(options.crop.leftTopCorner);
	int cropHeight = std::get<1>(options.crop.rightBottomCorner) - std::get<1>(options.crop.leftTopCorner);
	bool crop = false;
	if (cropWidth > 0 && cropHeight > 0 && cropWidth < input->width && cropHeight < input->height) {
		crop = true;
		cropHost(input, output, options.crop, prop.maxThreadsPerBlock, &stream);
		output->width = cropWidth;
		output->height = cropHeight;
	}
	//

	//Resize (deallocate memory from crop)
	bool resize = false;
	if (options.resize.width && options.resize.height) {
		if (crop && (options.resize.width != output->width || options.resize.height != output->height))
			resize = true;
		if (!crop && (options.resize.width != input->width || options.resize.height != input->height))
			resize = true;

		if (resize) {
			resizeKernel(crop ? output : input, output, crop, options.resize, prop.maxThreadsPerBlock, &stream);
			output->width = options.resize.width;
			output->height = options.resize.height;
		}
	}
	
	if (output->width == 0 || output->height == 0) {
		output->width = options.resize.width = input->width;
		output->height = options.resize.height = input->height;
	}
	//

	//Color conversion
	if (options.color.normalization)
		sts = colorConversionKernel<float>(resize || crop ? output : input, output, options.color, prop.maxThreadsPerBlock, &stream);
	else
		sts = colorConversionKernel<unsigned char>(resize || crop ? output : input, output, options.color, prop.maxThreadsPerBlock, &stream);
	//

	if (resize || crop) {
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

	for (auto item : streamArr) {
		cudaStreamDestroy(std::get<1>(item));
	}

	isClosed = true;
}
