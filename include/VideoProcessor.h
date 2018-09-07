#pragma once

extern "C" {
	#include <libavformat/avformat.h>
}
#include <memory>

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

int NV12ToRGB24(AVFrame* src, AVFrame* dst);

class VideoProcessor {
public:
	int Init(VPPParameters* outputFormat);
	/*
	Check if VPP conversion for input package is needed and perform conversion.
	Notice: VPP doesn't allocate memory for output frame, so correctly allocated Tensor with correct FourCC and resolution
	should be passed via Python API	and this allocated CUDA memory will be filled.
	*/
	int Convert(AVFrame* input, AVFrame* output);
private:
	VPPParameters* state;
	std::shared_ptr<FILE> dumpFrame;
};