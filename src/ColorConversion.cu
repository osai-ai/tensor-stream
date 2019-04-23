#include <libavutil/frame.h>
#include "cuda.h"
#include "VideoProcessor.h"



__device__ void NV12toRGB32Kernel(unsigned char* Y, unsigned char* UV, int* R, int* G, int* B, int i, int j, int pitchNV12) {
	/*
	R = 1.164(Y - 16) + 1.596(V - 128)
	B = 1.164(Y - 16)                   + 2.018(U - 128)
	G = 1.164(Y - 16) - 0.813(V - 128)  - 0.391(U - 128)
*/
/*
in case of NV12 we have Y component for every pixel and UV for every 2x2 Y
*/
	int UVRow = i / 2;
	int UVCol = j % 2 == 0 ? j : j - 1;
	int UIndex = UVRow * pitchNV12 /*pitch?*/ + UVCol;
	int VIndex = UVRow * pitchNV12 /*pitch?*/ + UVCol + 1;
	unsigned char U = UV[UIndex];
	unsigned char V = UV[VIndex];
	int indexNV12 = j + i * pitchNV12; /*indexNV12 and indexRGB with/without pitch*/
	unsigned char YVal = Y[indexNV12];
	*R = 1.164f*(YVal - 16) + 1.596f*(V - 128);
	*R = min(*R, 255);
	*R = max(*R, 0);
	*B = 1.164f*(YVal - 16) + 2.018f*(U - 128);
	*B = min(*B, 255);
	*B = max(*B, 0);
	*G = 1.164f*(YVal - 16) - 0.813f*(V - 128) - 0.391f*(U - 128);
	*G = min(*G, 255);
	*G = max(*G, 0);
}

__global__ void NV12ToRGB32KernelPlanar(unsigned char* Y, unsigned char* UV, float* RGB, int width, int height, int pitchNV12, int pitchRGB, bool swapRB, bool normalization) {
	unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < height && j < width) {
		int R, G, B;
		NV12toRGB32Kernel(Y, UV, &R, &G, &B, i, j, pitchNV12);
		(RGB[j + i * pitchRGB + 0 * (pitchRGB * height) /*R*/]) = (float) R;
		if (swapRB)
			(RGB[j + i * pitchRGB + 0 * (pitchRGB * height)]) = (float) B;
		if (normalization)
			RGB[j + i * pitchRGB + 0 * (pitchRGB * height)] /= 255;

		(RGB[j + i * pitchRGB + 1 * (pitchRGB * height) /*G*/]) = (float) G;
		if (normalization)
			RGB[j + i * pitchRGB + 1 * (pitchRGB * height)] /= 255;

		(RGB[j + i * pitchRGB + 2 * (pitchRGB * height) /*B*/]) = (float) B;
		if (swapRB)
			(RGB[j + i * pitchRGB + 2 * (pitchRGB * height)]) = (float) R;
		if (normalization)
			RGB[j + i * pitchRGB + 2 * (pitchRGB * height)] /= 255;

	}
}

__global__ void NV12ToRGB32KernelMerged(unsigned char* Y, unsigned char* UV, float* RGB, int width, int height, int pitchNV12, int pitchRGB, bool swapRB, bool normalization) {
	unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < height && j < width) {
		int R, G, B;
		NV12toRGB32Kernel(Y, UV, &R, &G, &B, i, j, pitchNV12);
		RGB[j * 3 + i * pitchRGB + 0/*R*/] = (float) R;
		if (swapRB)
			RGB[j * 3 + i * pitchRGB + 0] = (float) B;
		if (normalization)
			RGB[j * 3 + i * pitchRGB + 0] /= 255;

		RGB[j * 3 + i * pitchRGB + 1/*G*/] = (float) G;
		if (normalization)
			RGB[j * 3 + i * pitchRGB + 1] /= 255;

		RGB[j * 3 + i * pitchRGB + 2/*B*/] = (float) B;
		if (swapRB)
			RGB[j * 3 + i * pitchRGB + 2] = (float) R;
		if (normalization)
			RGB[j * 3 + i * pitchRGB + 2] /= 255;

	}
}

int colorConversionKernel(AVFrame* src, AVFrame* dst, ColorParameters color, int maxThreadsPerBlock, cudaStream_t* stream) {
	/*
	src in GPU nv12, dst in CPU rgb (packed)
	*/
	int width = src->width;
	int height = src->height;

	float* destination = nullptr;
	cudaError err = cudaSuccess;
	err = cudaMalloc(&destination, dst->channels * width * height * sizeof(float));

	//need to execute for width and height
	dim3 threadsPerBlock(64, maxThreadsPerBlock / 64);

	//blocks for merged format
	int blockX = std::ceil(dst->channels * width / (float)threadsPerBlock.x);
	int blockY = std::ceil(dst->height / (float)threadsPerBlock.y);
	
	//blocks for planar format
	if (color.planesPos == Planes::PLANAR) {
		blockX = std::ceil(width / (float)threadsPerBlock.x);
		blockY = std::ceil(dst->channels * dst->height / (float)threadsPerBlock.y);
	}

	dim3 numBlocks(blockX, blockY);
	//depends on fact of resize
	int pitchNV12 = src->linesize[0] ? src->linesize[0] : width;
	bool swapRB = false;
	switch (color.dstFourCC) {
		case BGR24:
			swapRB = true;
			if (color.planesPos == Planes::PLANAR) {
				int pitchRGB = width;
				NV12ToRGB32KernelPlanar << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], destination, width, height, pitchNV12, pitchRGB, swapRB, color.normalization);
			}
			else {
				int pitchRGB = dst->channels * width;
				NV12ToRGB32KernelMerged << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], destination, width, height, pitchNV12, pitchRGB, swapRB, color.normalization);
			}
		break;
		case RGB24:
			if (color.planesPos == Planes::PLANAR) {
				int pitchRGB = width;
				NV12ToRGB32KernelPlanar << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], destination, width, height, pitchNV12, pitchRGB, swapRB, color.normalization);
			}
			else {
				int pitchRGB = dst->channels * width;
				NV12ToRGB32KernelMerged << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], destination, width, height, pitchNV12, pitchRGB, swapRB, color.normalization);
			}
		break;
		case Y800:
			err = cudaMemcpy2D(destination, dst->width, dst->data[0], pitchNV12, dst->width, dst->height, cudaMemcpyDeviceToDevice);
		break;
		default:
			err = cudaErrorMissingConfiguration;
	}

	dst->opaque = destination;

	return err;
}
