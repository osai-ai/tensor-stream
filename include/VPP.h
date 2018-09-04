#pragma once
/*should it be under extern clause?*/
#include <libavformat/avformat.h>
#include <cuda.h>

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
};

/*
Declaration of CUDA method
*/
int resize(AVFrame* src, uint8_t* dst, unsigned int width, unsigned int height, CUstream stream);
int conversion(AVFrame* src, uint8_t* dst, FourCC dstFourCC, CUstream stream);

/*
Check if VPP conversion for input package is needed and perform conversion.
Notice: VPP doesn't allocate memory for output frame, so correctly allocated Tensor with correct FourCC and resolution
should be passed via Python API	and this allocated CUDA memory will be filled.
*/
int Convert(AVFrame* input, uint8_t* output, VPPParameters outputFormat);
