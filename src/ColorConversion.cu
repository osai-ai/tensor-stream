#include <libavutil/frame.h>
#include "cuda.h"
#include "VideoProcessor.h"
#include <iostream>

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

template< class T >
__global__ void NV12ToRGB32KernelPlanar(unsigned char* Y, unsigned char* UV, T* RGB, int width, int height, int pitchNV12, int pitchRGB, bool swapRB, bool normalization) {
	unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < height && j < width) {
		int R, G, B;
		NV12toRGB32Kernel(Y, UV, &R, &G, &B, i, j, pitchNV12);
		(RGB[j + i * pitchRGB + 0 * (pitchRGB * height) /*R*/]) = (T) R;
		if (swapRB)
			(RGB[j + i * pitchRGB + 0 * (pitchRGB * height)]) = (T) B;
		if (normalization)
			RGB[j + i * pitchRGB + 0 * (pitchRGB * height)] /= 255;

		(RGB[j + i * pitchRGB + 1 * (pitchRGB * height) /*G*/]) = (T) G;
		if (normalization)
			RGB[j + i * pitchRGB + 1 * (pitchRGB * height)] /= 255;

		(RGB[j + i * pitchRGB + 2 * (pitchRGB * height) /*B*/]) = (T) B;
		if (swapRB)
			(RGB[j + i * pitchRGB + 2 * (pitchRGB * height)]) = (T) R;
		if (normalization)
			RGB[j + i * pitchRGB + 2 * (pitchRGB * height)] /= 255;

	}
}

template< class T >
__global__ void NV12ToRGB32KernelMerged(unsigned char* Y, unsigned char* UV, T* RGB, int width, int height, int pitchNV12, int pitchRGB, bool swapRB, bool normalization) {
	unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < height && j < width) {
		int R, G, B;
		NV12toRGB32Kernel(Y, UV, &R, &G, &B, i, j, pitchNV12);
		RGB[j * 3 + i * pitchRGB + 0/*R*/] = (T) R;
		if (swapRB)
			RGB[j * 3 + i * pitchRGB + 0] = (T) B;
		if (normalization)
			RGB[j * 3 + i * pitchRGB + 0] /= 255;

		RGB[j * 3 + i * pitchRGB + 1/*G*/] = (T) G;
		if (normalization)
			RGB[j * 3 + i * pitchRGB + 1] /= 255;

		RGB[j * 3 + i * pitchRGB + 2/*B*/] = (T) B;
		if (swapRB)
			RGB[j * 3 + i * pitchRGB + 2] = (T) R;
		if (normalization)
			RGB[j * 3 + i * pitchRGB + 2] /= 255;

	}
}

template< class T >
__global__ void NV12ToY800(unsigned char* Y, T* Yf, int width, int height, int pitchNV12, bool normalization) {
	unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < height && j < width) {
		Yf[j + i * width] = Y[j + i * pitchNV12];
		if (normalization)
			Yf[j + i * width] /= 255;
	}
}

//semi-planar 420 to merged 422
//u0 y0 v0 y1 | u1 y2 v1 y3 | u2 y4 v2 y5 | u3 y6 v3 y7
template< class T >
__global__ void NV12ToUYVY(unsigned char* Y, unsigned char* UV, T* YDest, T* UVDest, int width, int height, int pitchNV12, bool normalization) {
	unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < height && j < width) {
		YDest[j + i * width] = Y[j + i * pitchNV12];
		if (normalization)
			YDest[j + i * width] /= 255;
		
		//max UV for NV12 - j/2 i/2
		//       for UYVY - j/2 i
		int UVRow = i / 2;
		int UVCol = j;
		int value = UV[UVCol + UVRow * width];
		if (i % 2 != 0) {
			int point1 = i;
			int point2 = i + 1;
			point2 = min(point2, height / 2 - 1);
			int point3 = i - 1;
			point3 = max(point3, 0);
			int point4 = i + 2;
			point4 = min(point4, height / 2 - 1);
			value = ((9 * (UV[point1] + UV[point2]) - (UV[point3] + UV[point4]) + 8) >> 4);
			value = min(value, 255);
			value = max(value, 0);
		}
		UVDest[UVCol + (UVRow * 2) * width] = value;
	}
}


template <class T>
int colorConversionKernel(AVFrame* src, AVFrame* dst, ColorOptions color, int maxThreadsPerBlock, cudaStream_t* stream) {
	/*
	src in GPU nv12, dst in CPU rgb (packed)
	*/
	int width = src->width;
	int height = src->height;

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

	T* destination = nullptr;
	cudaError err = cudaSuccess;
	err = cudaMalloc(&destination, dst->channels * width * height * sizeof(T));

	//depends on fact of resize
	int pitchNV12 = src->linesize[0] ? src->linesize[0] : width;
	bool swapRB = false;
	switch (color.dstFourCC) {
		case BGR24:
			swapRB = true;
			if (color.planesPos == Planes::PLANAR) {
				int pitchRGB = width;
				NV12ToRGB32KernelPlanar<T> << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], destination, width, height, pitchNV12, pitchRGB, swapRB, color.normalization);
			}
			else {
				int pitchRGB = dst->channels * width;
				NV12ToRGB32KernelMerged<T> << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], destination, width, height, pitchNV12, pitchRGB, swapRB, color.normalization);
			}
		break;
		case RGB24:
			if (color.planesPos == Planes::PLANAR) {
				int pitchRGB = width;
				NV12ToRGB32KernelPlanar<T> << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], destination, width, height, pitchNV12, pitchRGB, swapRB, color.normalization);
			}
			else {
				int pitchRGB = dst->channels * width;
				NV12ToRGB32KernelMerged<T> << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], destination, width, height, pitchNV12, pitchRGB, swapRB, color.normalization);
			}
		break;
		case Y800:
			NV12ToY800 << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], destination, width, height, pitchNV12, color.normalization);
		break;
		case UVYV:
			NV12ToUYVY << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], destination, width, height, pitchNV12, color.normalization);
		break;
		default:
			err = cudaErrorMissingConfiguration;
	}

	dst->opaque = destination;

	return err;
}

template
int colorConversionKernel<unsigned char>(AVFrame* src, AVFrame* dst, ColorOptions color, int maxThreadsPerBlock, cudaStream_t* stream);

template
int colorConversionKernel<float>(AVFrame* src, AVFrame* dst, ColorOptions color, int maxThreadsPerBlock, cudaStream_t* stream);