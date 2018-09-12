#pragma once

extern "C" {
	#include <libavformat/avformat.h>
}
#include <memory>
#include <cuda_runtime.h>
#include <vector>
#include <mutex>
/*
List of supported output formats
*/
enum FourCC {
	Y800,
	RGB24,
	NV12
};

/*
Structure contains description of desired conversions.
*/
struct VPPParameters {
	unsigned int width;
	unsigned int height;
	FourCC dstFourCC;
	bool enableDumps;
};

int NV12ToRGB24(AVFrame* src, AVFrame* dst, int maxThreadsPerBlock, cudaStream_t* stream);
int NV12ToRGB24Dump(AVFrame* src, AVFrame* dst, int maxThreadsPerBlock, cudaStream_t* stream);

class VideoProcessor {
public:
	int Init(VPPParameters& outputFormat);
	/*
	Check if VPP conversion for input package is needed and perform conversion.
	Notice: VPP doesn't allocate memory for output frame, so correctly allocated Tensor with correct FourCC and resolution
	should be passed via Python API	and this allocated CUDA memory will be filled.
	*/
	int Convert(AVFrame* input, AVFrame* output, std::string consumerName);
	void Close();

private:
	VPPParameters state;
	std::shared_ptr<FILE> dumpFrame;
	cudaDeviceProp prop;
	//should be map for every consumer
	std::vector<std::pair<std::string, cudaStream_t> > streamArr;
	std::mutex streamSync;
};