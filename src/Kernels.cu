#include <libavutil/frame.h>
#include "cuda.h"
#include "VideoProcessor.h"

__global__ void NV12ToRGB32Kernel(unsigned char* Y, unsigned char* UV, unsigned char* RGB, int width, int height, int pitchNV12, int pitchRGB) {
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

__global__ void NV12ToBGR32Kernel(unsigned char* Y, unsigned char* UV, unsigned char* BGR, int width, int height, int pitchNV12, int pitchRGB) {
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
		BGR[j * 3 + i * pitchRGB + 0/*B*/] = (unsigned char)BVal;
		BGR[j * 3 + i * pitchRGB + 1 /*G*/] = (unsigned char)GVal;
		BGR[j * 3 + i * pitchRGB + 2/*R*/] = (unsigned char)RVal;
	}
}

__global__ void resizeNV12NearestKernel(unsigned char* inputY, unsigned char* inputUV, unsigned char* outputY, unsigned char* outputUV,
	                             int srcWidth, int srcHeight, int srcLinesizeY, int srcLinesizeUV, int dstWidth, int dstHeight, float xRatio, float yRatio) {
	
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; //coordinate of pixel (y) in destination image
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; //coordinate of pixel (x) in destination image
	
	if (i < dstHeight && j < dstWidth) {
		int y = (int)(yRatio * i); //it's coordinate of pixel in source image
		int x = (int)(xRatio * j); //it's coordinate of pixel in source image
		int index = y * srcLinesizeY + x; //index in source image
		outputY[i * dstWidth + j] = inputY[index];
		//we should take chroma for every 2 luma, also height of data[1] is twice less than data[0]
		//there are no difference between x_ratio for Y and UV also as for y_ratio because (src_height / 2) / (dst_height / 2) = src_height / dst_height
		if (j % 2 == 0 && i < dstHeight / 2) {
			index = y * srcLinesizeUV + x; //index in source image
			int indexU, indexV;
			if (index % 2 == 0) {
				indexU = index;
				indexV = index + 1;
			}
			else {
				indexU = index - 1;
				indexV = index;
			}
			outputUV[i * dstWidth + j] = inputUV[indexU];
			outputUV[i * dstWidth + j + 1] = inputUV[indexV];
		}
	}
}

__device__ int calculateBillinearInterpolation(unsigned char* data, int startIndex, int xDiff, int yDiff, int linesize, int weightX, int weightY) {
	// range is 0 to 255 thus bitwise AND with 0xff
	int A = data[startIndex] & 0xff;
	int B = data[startIndex + xDiff] & 0xff;
	int C = data[startIndex + linesize] & 0xff;
	int D = data[startIndex + linesize + yDiff] & 0xff;

	// value = A(1-w)(1-h) + B(w)(1-h) + C(h)(1-w) + Dwh
	int value = (int)(
		A * (1 - weightX) * (1 - weightY) +
		B * (weightX) * (1 - weightY) +
		C * (weightY) * (1 - weightX) +
		D * (weightX  *      weightY)
		);
	return value;
}

__global__ void resizeNV12BilinearKernel(unsigned char* inputY, unsigned char* inputUV, unsigned char* outputY, unsigned char* outputUV,
	int srcWidth, int srcHeight, int srcLinesizeY, int srcLinesizeUV, int dstWidth, int dstHeight, float xRatio, float yRatio) {

	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; //coordinate of pixel (y) in destination image
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; //coordinate of pixel (x) in destination image

	if (i < dstHeight && j < dstWidth) {
		int y = (int)(yRatio * i); //it's coordinate of pixel in source image
		int x = (int)(xRatio * j); //it's coordinate of pixel in source image
		float xFloat = (xRatio * j) - x;
		float yFloat = (yRatio * i) - y;
		int index = y * srcLinesizeY + x; //index in source image
		outputY[i * dstWidth + j] = calculateBillinearInterpolation(inputY, index, 1, 1, srcLinesizeY, xFloat, yFloat);
		//we should take chroma for every 2 luma, also height of data[1] is twice less than data[0]
		//there are no difference between x_ratio for Y and UV also as for y_ratio because (src_height / 2) / (dst_height / 2) = src_height / dst_height
		if (j % 2 == 0 && i < dstHeight / 2) {
			index = y * srcLinesizeUV + x; //index in source image
			int indexU, indexV;
			if (index % 2 == 0) {
				indexU = index;
				indexV = index + 1;
			}
			else {
				indexU = index - 1;
				indexV = index;
			}
			outputUV[i * dstWidth + j] = calculateBillinearInterpolation(inputUV, indexU, 2, 2, srcLinesizeUV, xFloat, yFloat);
			outputUV[i * dstWidth + j + 1] = calculateBillinearInterpolation(inputUV, indexV, 2, 2, srcLinesizeUV, xFloat, yFloat);
		}
	}
}

int resizeNV12Nearest(AVFrame* src, AVFrame* dst, int maxThreadsPerBlock, cudaStream_t * stream) {
	unsigned char* outputY = nullptr;
	unsigned char* outputUV = nullptr;
	cudaError err = cudaMalloc(&outputY, dst->width * dst->height * sizeof(unsigned char)); //in resize we don't change color format
	err = cudaMalloc(&outputUV, dst->width * (dst->height / 2) * sizeof(unsigned char));
	//need to execute for width and height
	dim3 threadsPerBlock(64, maxThreadsPerBlock / 64);
	int blockX = std::ceil(dst->width / (float)threadsPerBlock.x);
	int blockY = std::ceil(dst->height / (float)threadsPerBlock.y);
	dim3 numBlocks(blockX, blockY);
	float xRatio = ((float)(src->width - 1)) / dst->width; //if not -1 we should examine 2x2 square with top-left corner in the last pixel of src, so it's impossible
	float yRatio = ((float)(src->height - 1)) / dst->height;
	resizeNV12NearestKernel << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], outputY, outputUV,
		                                                              src->width, src->height, src->linesize[0], src->linesize[1], 
		                                                              dst->width, dst->height, xRatio, yRatio);
	dst->data[0] = outputY;
	dst->data[1] = outputUV;
	return err;
}

int resizeNV12Bilinear(AVFrame* src, AVFrame* dst, int maxThreadsPerBlock, cudaStream_t * stream) {
	unsigned char* outputY = nullptr;
	unsigned char* outputUV = nullptr;
	cudaError err = cudaMalloc(&outputY, dst->width * dst->height * sizeof(unsigned char)); //in resize we don't change color format
	err = cudaMalloc(&outputUV, dst->width * (dst->height / 2) * sizeof(unsigned char));
	//need to execute for width and height
	dim3 threadsPerBlock(64, maxThreadsPerBlock / 64);
	int blockX = std::ceil(dst->width / (float)threadsPerBlock.x);
	int blockY = std::ceil(dst->height / (float)threadsPerBlock.y);
	dim3 numBlocks(blockX, blockY);
	float xRatio = ((float)(src->width - 1)) / dst->width; //if not -1 we should examine 2x2 square with top-left corner in the last pixel of src, so it's impossible
	float yRatio = ((float)(src->height - 1)) / dst->height;
	resizeNV12BilinearKernel << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], outputY, outputUV,
		src->width, src->height, src->linesize[0], src->linesize[1],
		dst->width, dst->height, xRatio, yRatio);
	dst->data[0] = outputY;
	dst->data[1] = outputUV;
	return err;
}

int NV12ToRGB24(AVFrame* src, AVFrame* dst, int maxThreadsPerBlock, cudaStream_t * stream) {
	/*
	src in GPU nv12, dst in CPU rgb (packed)
	*/
	int width = src->width;
	int height = src->height;
	unsigned char* RGB = nullptr;
	clock_t tStart = clock();
	cudaError err = cudaMalloc(&RGB, dst->channels * width * height * sizeof(unsigned char));
	//need to execute for width and height
	dim3 threadsPerBlock(64, maxThreadsPerBlock / 64);
	int blockX = std::ceil(dst->channels * width / (float)threadsPerBlock.x);
	int blockY = std::ceil(dst->height / (float)threadsPerBlock.y);
	dim3 numBlocks(blockX, blockY);
	if (src->linesize[0])
		NV12ToRGB32Kernel << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], RGB, width, height, src->linesize[0], dst->channels * width);
	else
		NV12ToRGB32Kernel << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], RGB, width, height, width, dst->channels * width);
	dst->opaque = RGB;
	return err;
}

int NV12ToBGR24(AVFrame* src, AVFrame* dst, int maxThreadsPerBlock, cudaStream_t * stream) {
	/*
	src in GPU nv12, dst in CPU rgb (packed)
	*/
	int width = src->width;
	int height = src->height;
	unsigned char* BGR = nullptr;
	cudaError err = cudaMalloc(&BGR, dst->channels * width * height * sizeof(unsigned char));
	//need to execute for width and height
	dim3 threadsPerBlock(64, maxThreadsPerBlock / 64);
	int blockX = std::ceil(dst->channels * width / (float)threadsPerBlock.x);
	int blockY = std::ceil(dst->height / (float)threadsPerBlock.y);
	dim3 numBlocks(blockX, blockY);
	if (src->linesize[0])
		NV12ToBGR32Kernel << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], BGR, width, height, src->linesize[0], dst->channels * width);
	else
		NV12ToBGR32Kernel << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], BGR, width, height, width, dst->channels * width);
	dst->opaque = BGR;
	return err;
}