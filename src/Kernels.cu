#include <libavutil/frame.h>
#include "cuda.h"
#include "VideoProcessor.h"

__global__ void change_gpu(unsigned char* Y, unsigned char* UV, unsigned char* RGB, int width, int height, int pitchNV12, int pitchRGB) {
	/*
		R = 1.164(Y - 16) + 1.596(V - 128)
		B = 1.164(Y - 16)                   + 2.018(U - 128)
		G = 1.164(Y - 16) - 0.813(V - 128)  - 0.391(U - 128)
	*/
	/*
	in case of NV12 we have Y component for every pixel and UV for every 2x2 Y
	*/
	unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < height && j < width) {
		int UVRow = i / 2;
		int UVCol = j % 2 == 0 ? j : j - 1;
		int UIndex = UVRow * pitchNV12 /*pitch?*/ + UVCol;
		int VIndex = UVRow * pitchNV12 /*pitch?*/ + UVCol + 1;
		unsigned char U = UV[UIndex];
		unsigned char V = UV[VIndex];
		int indexNV12 = j + i * pitchNV12; /*indexNV12 and indexRGB with/without pitch*/
		unsigned char YVal = Y[indexNV12];
		int RVal = 1.164f*(YVal - 16) + 1.596f*(V - 128);
		if (RVal > 255)
			RVal = 255;
		if (RVal < 0)
			RVal = 0;
		int BVal = 1.164f*(YVal - 16) + 2.018f*(U - 128);
		if (BVal > 255)
			BVal = 255;
		if (BVal < 0)
			BVal = 0;
		int GVal = 1.164f*(YVal - 16) - 0.813f*(V - 128) - 0.391f*(U - 128);
		if (GVal > 255)
			GVal = 255;
		if (GVal < 0)
			GVal = 0;
		RGB[j * 3 + i * pitchRGB + 0/*R*/] = (unsigned char)RVal;
		RGB[j * 3 + i * pitchRGB + 1 /*G*/] = (unsigned char)GVal;
		RGB[j * 3 + i * pitchRGB + 2/*B*/] = (unsigned char)BVal;
	}
}

/*
should use CUStream?
*/
int NV12ToRGB24(AVFrame* src, AVFrame* dst, int maxThreadsPerBlock, cudaStream_t * stream) {
	/*
	src in GPU nv12, dst in CPU rgb (packed)
	*/
	int width = dst->width;
	int height = dst->height;
	unsigned char* RGB;
	clock_t tStart = clock();
	cudaError err = cudaMalloc(&RGB, 3 * width * height * sizeof(unsigned char));
	printf("Time taken for malloc: %f\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	//need to execute for width and height
	dim3 threadsPerBlock(64, maxThreadsPerBlock / 64);
	dim3 numBlocks(3 * width / threadsPerBlock.x, height / threadsPerBlock.y);
	change_gpu << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], RGB, width, height, src->linesize[0], 3 * width);
	//cudaDeviceSynchronize(); //needed when cudaMemcpy will be deleted
	dst->opaque = RGB;
	return err;
}

int NV12ToRGB24Dump(AVFrame* src, AVFrame* dst, int maxThreadsPerBlock, cudaStream_t * stream) {
	/*
	src in GPU nv12, dst in CPU rgb (packed)
	*/
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int width = dst->width;
	int height = dst->height;
	unsigned char* RGB;
	cudaError err = cudaMalloc(&RGB, 3 * width * height * sizeof(unsigned char));
	//need to execute for width and height
	dim3 threadsPerBlock(64, maxThreadsPerBlock / 64);
	dim3 numBlocks(3 * width / threadsPerBlock.x, height / threadsPerBlock.y);
	change_gpu << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], RGB, width, height, src->linesize[0], 3 * width);
	cudaMemcpy(dst->data[0], RGB, 3 * width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	dst->opaque = RGB;
	return err;
}
