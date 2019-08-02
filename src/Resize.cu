#include <libavutil/frame.h>
#include "cuda.h"
#include "VideoProcessor.h"

__device__ int calculateIntegerAreaInterpolation(unsigned char* data, int startIndex, float scaleX, float scaleY, int linesize, int stride, float* patternX, float* patternY) {
	float colorSum = 0;
	int rScaleX = round(scaleX);
	int rScaleY = round(scaleY);
	float divide = 0;
	for (int i = 0; i < rScaleY + 1; i++) {
		for (int j = 0; j < rScaleX + 1; j++) {
			int index = startIndex + j * stride + i * linesize;
			float weightX = patternX[j];
			float weightY = patternY[i];
			float weight = weightX * weightY;
			divide += weight;
			colorSum += (float)data[index] * weight;
		}
	}

	colorSum /= divide;

	return colorSum;
}

__global__ void resizeNV12IntegerDownscaleAreaKernel(unsigned char* inputY, unsigned char* inputUV, unsigned char* outputY, unsigned char* outputUV,
	int srcWidth, int srcHeight, int srcLinesizeY, int srcLinesizeUV, int dstWidth, int dstHeight, float xRatio, float yRatio, 
	float** patternX, int patternXSize, float** patternY, int patternYSize) {
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; //coordinate of pixel (y) in destination image
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; //coordinate of pixel (x) in destination image
	if (i < dstHeight && j < dstWidth) {
		int y = (int)(yRatio * i); //it's coordinate of pixel in source image
		int x = (int)(xRatio * j); //it's coordinate of pixel in source image
		int index = y * srcLinesizeY + x; //index in source image
		int patternIndexX = index % patternXSize;
		int patternIndexY = index % patternYSize;
		float* rowPatternX = patternX[patternIndexX];
		float* rowPatternY = patternY[patternIndexY];
		outputY[i * dstWidth + j] = calculateIntegerAreaInterpolation(inputY, index, xRatio, yRatio, srcLinesizeY, 1, rowPatternX, rowPatternY);
		//we should take chroma for every 2 luma, also height of data[1] is twice less than data[0]
		//there are no difference between x_ratio for Y and UV also as for y_ratio because (src_height / 2) / (dst_height / 2) = src_height / dst_height
		if (i < dstHeight / 2 && j < dstWidth / 2) {
			index = y * srcLinesizeUV + x * 2; //index in source image
			int indexU, indexV;
			indexU = index;
			indexV = index + 1;
			outputUV[i * dstWidth + 2 * j] = calculateIntegerAreaInterpolation(inputUV, indexU, xRatio, yRatio, srcLinesizeUV, 2, rowPatternX, rowPatternY);
			outputUV[i * dstWidth + 2 * j + 1] = calculateIntegerAreaInterpolation(inputUV, indexV, xRatio, yRatio, srcLinesizeUV, 2, rowPatternX, rowPatternY);
			/*
			0 0 -> 0 -> 0 1 -> 0 1
			0 1 -> 2 -> 2 3 -> 2 3
			0 2 -> 6 -> 6 7 -> 4 5
			*/
		}
	}
}

__global__ void resizeNV12UpscaleAreaKernel(unsigned char* inputY, unsigned char* inputUV, unsigned char* outputY, unsigned char* outputUV,
	int srcWidth, int srcHeight, int srcLinesizeY, int srcLinesizeUV, int dstWidth, int dstHeight, float xRatio, float yRatio) {

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

__device__ int calculateBillinearInterpolation(unsigned char* data, int startIndex, int xDiff, int yDiff, int linesize, float weightX, float weightY) {
	int A = data[startIndex];
	int B = data[startIndex + xDiff];
	int C = data[startIndex + linesize];
	int D = data[startIndex + linesize + yDiff];

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

void generateResizePattern(float scale, std::vector<std::vector<float> >& pattern) {
	int currentID = 0;
	float rest = 0;

	while (currentID * scale == 0 || (currentID * scale - (int)(currentID * scale) > std::numeric_limits<float>::epsilon())) {
		float dynScale = scale;
		pattern.push_back(std::vector<float>());
		if (rest) {
			pattern[currentID].push_back(rest);
			dynScale -= rest;
		}
		while (dynScale - 1 > 0) {
			pattern[currentID].push_back(1);
			dynScale--;
		}
		//push rest to pattern
		pattern[currentID].push_back(dynScale);
		rest = 1 - dynScale;

		while (pattern[currentID].size() < round(scale) + 1)
			pattern[currentID].push_back(0);

		currentID += 1;
	}
}


float** copy2DArray(std::vector<std::vector<float> > pattern, float ratio) {
	float** patternCUDA;
	cudaError err = cudaMalloc((void **)&patternCUDA, sizeof(float*) * pattern.size());
	float** tempPatternCUDA = (float**)malloc(sizeof(float*) * pattern.size());
	for (int i = 0; i < pattern.size(); i++) {
		err = cudaMalloc((void**)&tempPatternCUDA[i], sizeof(float) * (round(ratio) + 1));
		err = cudaMemcpy(tempPatternCUDA[i], pattern[i].data(), sizeof(float) * (round(ratio) + 1), cudaMemcpyHostToDevice);
	}

	err = cudaMemcpy(patternCUDA, tempPatternCUDA, sizeof(float*) * pattern.size(), cudaMemcpyHostToDevice);
	return patternCUDA;
}

cudaError free2DArray(float** pattern, int size, float ratio) {
	cudaError err;
	err = cudaFree(pattern);
	return err;
}

int resizeKernel(AVFrame* src, AVFrame* dst, ResizeType resize, int maxThreadsPerBlock, cudaStream_t * stream) {
	unsigned char* outputY = nullptr;
	unsigned char* outputUV = nullptr;
	cudaError err = cudaMalloc(&outputY, dst->width * dst->height * sizeof(unsigned char)); //in resize we don't change color format
	err = cudaMalloc(&outputUV, dst->width * (dst->height / 2) * sizeof(unsigned char));
	//need to execute for width and height
	dim3 threadsPerBlock(64, maxThreadsPerBlock / 64);
	int blockX = std::ceil(dst->width / (float)threadsPerBlock.x);
	int blockY = std::ceil(dst->height / (float)threadsPerBlock.y);
	dim3 numBlocks(blockX, blockY);
	float xRatio = (float)(src->width) / dst->width; //if not -1 we should examine 2x2 square with top-left corner in the last pixel of src, so it's impossible
	float yRatio = (float)(src->height) / dst->height;
	switch (resize) {
		case ResizeType::BILINEAR:
			resizeNV12BilinearKernel << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], outputY, outputUV,
				src->width, src->height, src->linesize[0], src->linesize[1],
				dst->width, dst->height, xRatio, yRatio);
		break;
		case ResizeType::NEAREST:
			resizeNV12NearestKernel << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], outputY, outputUV,
				src->width, src->height, src->linesize[0], src->linesize[1],
				dst->width, dst->height, xRatio, yRatio);
		break;
		case ResizeType::AREA:
			std::vector<std::vector<float> > patternX;
			std::vector<std::vector<float> > patternY;
			generateResizePattern(xRatio, patternX);
			generateResizePattern(yRatio, patternY);
			
			float** patternXCUDA = copy2DArray(patternX, xRatio);
			float** patternYCUDA = copy2DArray(patternY, yRatio);

			//Here we should decide which AREA algorithm to use
			resizeNV12IntegerDownscaleAreaKernel<< <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], outputY, 
				outputUV, src->width, src->height, src->linesize[0], src->linesize[1], dst->width, 	dst->height, xRatio, yRatio, 
				patternXCUDA, patternX.size(), patternYCUDA, patternY.size());
			
			free2DArray(patternXCUDA, patternX.size(), xRatio);
			free2DArray(patternYCUDA, patternY.size(), yRatio);

		break;
	}
	dst->data[0] = outputY;
	dst->data[1] = outputUV;
	return err;
}
