#include <libavutil/frame.h>
#include "cuda.h"

__global__ void kernel(int *a)
{
	*a = *a * 5;
	*a = *a + 2;
}

void kernel_wrapper(int *a)
{
	int* d_a;

	cudaMalloc(&d_a, sizeof(int));

	cudaMemcpy(d_a, a, sizeof(int), cudaMemcpyHostToDevice);

	// Perform SAXPY on 1M elements
	kernel <<<1,1>> > (d_a);

	cudaMemcpy(a, d_a, sizeof(int), cudaMemcpyDeviceToHost);
}


void change(unsigned char* Y, unsigned char* UV, unsigned char* RGB, int width, int height, int pitchNV12, int pitchRGB) {
	/*
		R = 1.164(Y - 16) + 1.596(V - 128)
		B = 1.164(Y - 16)                   + 2.018(U - 128)
		G = 1.164(Y - 16) - 0.813(V - 128)  - 0.391(U - 128)
	*/
	/*
	in case of NV12 we have Y component for every pixel and UV for every 2x2 Y
	*/
	for (int i = 0; i < height; i++) {
		for (int j = 0, jRGB = 0; j < width; j++, jRGB += 3) {
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
			RGB[jRGB + i * pitchRGB + 0/*R*/] = (unsigned char)RVal;
			RGB[jRGB + i * pitchRGB + 1 /*G*/] = (unsigned char)GVal;
			RGB[jRGB + i * pitchRGB + 2/*B*/] = (unsigned char)BVal;
		}
	}
}

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


void change_pixels(AVFrame* src, AVFrame* dst, CUstream stream) {
	/*
	src in GPU nv12, dst in CPU rgb (packed)
	*/
	int width = dst->width;
	int height = dst->height;
	//change(src->data[0], src->data[1], dst->data[0], width, height, src->linesize[0], dst->linesize[0]);
	unsigned char* RGB;
	cudaError err = cudaMalloc(&RGB, dst->linesize[0] * dst->height * sizeof(unsigned char));
	//need to execute for width and height
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(dst->linesize[0] / threadsPerBlock.x, height / threadsPerBlock.y);
	change_gpu << <numBlocks, threadsPerBlock, 0, stream >> > (src->data[0], src->data[1], RGB, width, height, src->linesize[0], dst->linesize[0]);
	err = cudaMemcpy(dst->data[0], RGB, dst->linesize[0] * dst->height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize(); //needed when cudaMemcpy will be deleted
}