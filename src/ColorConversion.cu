#include <libavutil/frame.h>
#include "cuda.h"
#include "VideoProcessor.h"



__device__ void NV12toRGB32Kernel(unsigned char* Y, unsigned char* UV, unsigned char* R, unsigned char* G, unsigned char* B, int i, int j, int pitchNV12) {
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

__global__ void NV12ToRGB32KernelPlanar(unsigned char* Y, unsigned char* UV, float* RGB, int width, int height, int pitchNV12, bool normalization, bool swapRB) {
	unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < height && j < width) {
		unsigned char* R = (unsigned char*) &(RGB[j + i * width + 0 * (width * height) /*R*/]);
		if (swapRB)
			R = (unsigned char*) &(RGB[j + i * width + 2 * (width * height) /*B*/]);
		unsigned char* G = (unsigned char*) &(RGB[j + i * width + 1 * (width * height) /*G*/]);
		unsigned char* B = (unsigned char*) &(RGB[j + i * width + 2 * (width * height) /*B*/]);
		if (swapRB)
			B = (unsigned char*) &(RGB[j + i * width + 0 * (width * height) /*R*/]);
		NV12toRGB32Kernel(Y, UV, R, G, B, i, j, pitchNV12);
		if (normalization) {
			*R = (float)*R / 255;
			*G = (float)*G / 255;
			*B = (float)*B / 255;
		}
	}
}

__global__ void NV12ToRGB32KernelMerged(unsigned char* Y, unsigned char* UV, float* RGB, int width, int height, int pitchNV12, bool normalization, bool swapRB) {
	unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < height && j < width) {
		unsigned char* R = (unsigned char*) &(RGB[j * 3 + i * width + 0/*R*/]);
		if (swapRB)
			R = (unsigned char*) &(RGB[j * 3 + i * width + 2/*B*/]);
		unsigned char* G = (unsigned char*) &(RGB[j * 3 + i * width + 1 /*G*/]);
		unsigned char* B = (unsigned char*) &(RGB[j * 3 + i * width + 2/*B*/]);
		if (swapRB)
			B = (unsigned char*) &(RGB[j * 3 + i * width + 0/*R*/]);
		NV12toRGB32Kernel(Y, UV, R, G, B, i, j, pitchNV12);
		if (normalization) {
			*R = (float)*R / 255;
			*G = (float)*G / 255;
			*B = (float)*B / 255;
		}
	}
}

__global__ void NV12ToY800N(unsigned char* Y, float* dst, int width, int height) {
	unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < height && j < width) {
		*dst = (float)(Y[j + i*width]) / 255;
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
	int pitchWidth = src->linesize[0] ? src->linesize[0] : width;
	bool swapRB = false;
	switch (color.dstFourCC) {
		case BGR24:
			dst->format = AV_PIX_FMT_BGR24;
			dst->channels = 3;
			swapRB = true;
			err = cudaMalloc(&destination, dst->channels * width * height * sizeof(float));
			if (color.planesPos == Planes::PLANAR) {
				NV12ToRGB32KernelPlanar << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], destination, width, height, pitchWidth, color.normalization, swapRB);
			}
			else {
				NV12ToRGB32KernelMerged << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], destination, width, height, pitchWidth, color.normalization, swapRB);
			}
		break;
		case RGB24:
			dst->format = AV_PIX_FMT_RGB24;
			dst->channels = 3;
			err = cudaMalloc(&destination, dst->channels * width * height * sizeof(float));
			if (color.planesPos == Planes::PLANAR) {
				NV12ToRGB32KernelPlanar << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], destination, width, height, pitchWidth, color.normalization, swapRB);
			}
			else {
				NV12ToRGB32KernelMerged << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], destination, width, height, pitchWidth, color.normalization, swapRB);
			}
		break;
		case Y800:
			dst->format = AV_PIX_FMT_GRAY8;
			dst->channels = 1;
			err = cudaMalloc(&destination, dst->width * dst->height * sizeof(float));
			if (color.normalization) {
				NV12ToY800N << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], destination, pitchWidth, height);
			}
			else {
				err = cudaMemcpy2D(destination, dst->width, dst->data[0], pitchWidth, dst->width, dst->height, cudaMemcpyDeviceToDevice);
			}
		break;
		default:
			err = cudaErrorMissingConfiguration;
	}

	//without resize
	dst->opaque = destination;
	return err;
}
