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

__device__ unsigned char calculateUYVYChromaVertical(unsigned char* UV, int i, int j, int width, int height) {
	int UVRow = i / 2;
	int UVCol = j;
	int index = UVCol + UVRow * width;
	int value = UV[index];
	if (UVRow % 2 != 0) {
		int point1 = UVRow;
		int point2 = UVRow + 1;
		point2 = min(point2, height / 2 - 1);
		int point3 = UVRow - 1;
		point3 = max(point3, 0);
		int point4 = UVRow + 2;
		point4 = min(point4, height / 2 - 1);
		value = ((9 * (UV[point1 * width + UVCol] + UV[point2 * width + UVCol]) 
					- (UV[point3 * width + UVCol] + UV[point4 * width + UVCol]) + 8) >> 4);
		value = min(value, 255);
		value = max(value, 0);
	}
	
	return value;
}

template <class T>
__device__ T calculateYUV444ChromaHorizontal(T* src, int index, int shift, int width, int height) {
	int point1 = index - 3 + shift;
	int point2 = index + 1 + shift;
	int point3 = index - 7 + shift;
	if (point3 < 0)
		point3 = point1;
	int point4 = index + 5 + shift;
	if (point4 > width * height * 2 - 1)
		point4 = point2;
	T value = ((9 * (src[point1] + src[point2]) - (src[point3] + src[point4]) + 8) / 16);
	value = min(value, (T) 255);
	value = max(value, (T) 0);
	return value;
}

template< class T >
__global__ void UYVYToYUV444(T* src, T* dst, int width, int height, bool normalization) {
	unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < height && j < width) {
		int index = j + i * width;
		int srcIndex = index * 2 + 1;
		dst[index] = src[srcIndex];
		if (index % 2 == 0) {
			dst[width * height + index] = src[srcIndex - 1];
			if (normalization)
				dst[width * height + index] /= 255;
			dst[2 * width * height + index] = src[srcIndex + 1];
			if (normalization)
				dst[2 * width * height + index] /= 255;
		}
		else {
			dst[width * height + index] = calculateYUV444ChromaHorizontal(src, srcIndex, 0, width, height);
			if (normalization)
				dst[width * height + index] /= 255;
			dst[2 * width * height + index] = calculateYUV444ChromaHorizontal(src, srcIndex, 2, width, height);
			if (normalization)
				dst[2 * width * height + index] /= 255;
		}
	}
}

//semi-planar 420 to merged 422
//u0 y0 v0 y1 | u1 y2 v1 y3 | u2 y4 v2 y5 | u3 y6 v3 y7
template< class T >
__global__ void NV12ToUYVY(unsigned char* Y, unsigned char* UV, T* dest, int width, int height, int pitchNV12, bool normalization) {
	unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < height && j < width) {
		int index = j + i * width;
		if (index % 2 == 0) {
			int indexDest = index * 2;
			//max UV for NV12 - j/2 i/2
			//       for UYVY - j/2 i

			unsigned char UValue = calculateUYVYChromaVertical(UV, i, j, pitchNV12, height);
			dest[indexDest] = UValue;
			if (normalization)
				dest[indexDest] /= 255;
			dest[indexDest + 1] = Y[index];
			if (normalization)
				dest[indexDest + 1] /= 255;
			unsigned char VValue = calculateUYVYChromaVertical(UV, i, j + 1, pitchNV12, height);
			dest[indexDest + 2] = VValue;
			if (normalization)
				dest[indexDest + 2] /= 255;
		}
		else {
			int indexDest = index * 2 + 1;
			dest[indexDest] = Y[index];
			if (normalization)
				dest[indexDest] /= 255;
		}
	}
}

template< class T >
__global__ void NV12MergeBuffers(unsigned char* Y, unsigned char* UV, T* dest, int width, int height, int pitchNV12, bool normalization) {
	unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < height && j < width) {
		int index = j + i * width;
		int indexNV12 = j + i * pitchNV12;
		dest[index] = Y[indexNV12];
		if (normalization)
			dest[index] /= 255;
		if (i % 2 == 0 && j % 2 == 0) {
			int indexUV = (int) (i / 2) * width + j;
			int indexUVNV12 = (int) (i / 2) * pitchNV12 + j;
			dest[width * height + indexUV] = UV[indexUVNV12];
			if (normalization)
				dest[width * height + indexUV] /= 255;
			dest[width * height + indexUV + 1] = UV[indexUVNV12 + 1];
			if (normalization)
				dest[width * height + indexUV + 1] /= 255;
		}
	}
}

template <class T>
int colorConversionKernel(AVFrame* src, AVFrame* dst, ColorOptions color, int maxThreadsPerBlock, cudaStream_t* stream) {
	float channels = 3;
	switch (color.dstFourCC) {
	case Y800:
		channels = 1;
		break;
	case UYVY:
		channels = 2;
		break;
	case NV12:
		channels = 1.5;
	}
	
	/*
	src in GPU nv12, dst in CPU rgb (packed)
	*/
	int width = src->width;
	int height = src->height;

	//need to execute for width and height
	dim3 threadsPerBlock(64, maxThreadsPerBlock / 64);

	//blocks for merged format
	int blockX = std::ceil(channels * width / (float)threadsPerBlock.x);
	int blockY = std::ceil(dst->height / (float)threadsPerBlock.y);
	
	//blocks for planar format
	if (color.planesPos == Planes::PLANAR) {
		blockX = std::ceil(width / (float)threadsPerBlock.x);
		blockY = std::ceil(channels * dst->height / (float)threadsPerBlock.y);
	}

	dim3 numBlocks(blockX, blockY);

	T* destination = nullptr;
	cudaError err = cudaSuccess;
	err = cudaMalloc(&destination, channels * width * height * sizeof(T));

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
				int pitchRGB = channels * width;
				NV12ToRGB32KernelMerged<T> << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], destination, width, height, pitchNV12, pitchRGB, swapRB, color.normalization);
			}
		break;
		case RGB24:
			if (color.planesPos == Planes::PLANAR) {
				int pitchRGB = width;
				NV12ToRGB32KernelPlanar<T> << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], destination, width, height, pitchNV12, pitchRGB, swapRB, color.normalization);
			}
			else {
				int pitchRGB = channels * width;
				NV12ToRGB32KernelMerged<T> << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], destination, width, height, pitchNV12, pitchRGB, swapRB, color.normalization);
			}
		break;
		case Y800:
			NV12ToY800 << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], destination, width, height, pitchNV12, color.normalization);
		break;
		case UYVY:
			NV12ToUYVY << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], destination, width, height, pitchNV12, color.normalization);
		break;
		case YUV444: 
		{
			NV12ToUYVY << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], destination, width, height, pitchNV12, color.normalization);
			T* destinationYUV444 = nullptr;
			err = cudaMalloc(&destinationYUV444, channels * width * height * sizeof(T));
			//It's more convinient to work with width*height than with any other sizes
			UYVYToYUV444 << <numBlocks, threadsPerBlock, 0, *stream >> > (destination, destinationYUV444, width, height, color.normalization);
			cudaFree(destination);
			destination = destinationYUV444;

		}
		break;
		case NV12:
			NV12MergeBuffers << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], destination, width, height, pitchNV12, color.normalization);
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