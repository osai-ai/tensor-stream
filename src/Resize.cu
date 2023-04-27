#include <libavutil/frame.h>
#include "cuda.h"
#include "VideoProcessor.h"

__device__ int calculateBillinearInterpolation(unsigned char* data, float x, float y, int xDiff, int yDiff, int linesize, int width, int height, float weightX, float weightY) {
	int startIndex = x + y * linesize;
	if (x + xDiff >= width)
		xDiff = 0;
	if (y + yDiff >= height)
		linesize = 0;
	int A = data[startIndex];
	int B = data[startIndex + xDiff];
	int C = data[startIndex + linesize * yDiff];
	int D = data[startIndex + linesize * yDiff + xDiff];

	//the most precise one
	int value = (int)(
		A * (1 - weightX) * (1 - weightY) +
		B * (weightX) * (1 - weightY) +
		C * (weightY) * (1 - weightX) +
		D * (weightX  *      weightY)
		);
	
	return value;
}

__device__ int calculateBicubicSplineInterpolation(unsigned char* data, int x, int y, int xDiff, int yDiff, int linesize, int width, int height, double weightX, double weightY) {
	int startIndex = x + y * linesize;
	int xDiffTop = xDiff;
	int yDiffTop = yDiff;

	if (x + xDiff >= width)
		xDiff = 0;
	if (x + xDiff * 2 >= width)
		xDiff = 0;
	if (x - xDiffTop < 0)
		xDiffTop = 0;
	if (y + yDiff >= height)
		yDiff = 0;
	if (y + yDiff * 2 >= height)
		yDiff = 0;
	if (y - yDiffTop < 0)
		yDiffTop = 0;

	double a = -0.75;
	double a0, a1, a2, a3;
	a0 = (a * weightX - 2 * a * pow(weightX, 2) + a * pow(weightX, 3)) * data[startIndex - xDiffTop - linesize * yDiffTop];
	a1 = (1 - (a + 3) * pow(weightX, 2) + (a + 2) * pow(weightX, 3)) * data[startIndex - linesize * yDiffTop];
	a2 = (-a * weightX + (2 * a + 3) * pow(weightX, 2) - (a + 2) * pow(weightX, 3)) * data[startIndex + xDiff - linesize * yDiffTop];
	a3 = (a * pow(weightX, 2) - a * pow(weightX, 3)) * data[startIndex + 2 * xDiff - linesize * yDiffTop];
	int b0 = round(a0 + a1 + a2 + a3);
	b0 = min(b0, 255);
	b0 = max(b0, 0);

	a0 = (a * weightX - 2 * a * pow(weightX, 2) + a * pow(weightX, 3)) * data[startIndex - xDiffTop];
	a1 = (1 - (a + 3) * pow(weightX, 2) + (a + 2) * pow(weightX, 3)) * data[startIndex];
	a2 = (-a * weightX + (2 * a + 3) * pow(weightX, 2) - (a + 2) * pow(weightX, 3)) * data[startIndex + xDiff];
	a3 = (a * pow(weightX, 2) - a * pow(weightX, 3)) * data[startIndex + 2 * xDiff];
	int b1 = round(a0 + a1 + a2 + a3);

	b1 = min(b1, 255);
	b1 = max(b1, 0);

	a0 = (a * weightX - 2 * a * pow(weightX, 2) + a * pow(weightX, 3)) * data[startIndex - xDiffTop + linesize * yDiff];
	a1 = (1 - (a + 3) * pow(weightX, 2) + (a + 2) * pow(weightX, 3)) * data[startIndex + linesize * yDiff];
	a2 = (-a * weightX + (2 * a + 3) * pow(weightX, 2) - (a + 2) * pow(weightX, 3)) * data[startIndex + xDiff + linesize * yDiff];
	a3 = (a * pow(weightX, 2) - a * pow(weightX, 3)) * data[startIndex + 2 * xDiff + linesize * yDiff];
	int b2 = round(a0 + a1 + a2 + a3);

	b2 = min(b2, 255);
	b2 = max(b2, 0);

	a0 = (a * weightX - 2 * a * pow(weightX, 2) + a * pow(weightX, 3)) * data[startIndex - xDiffTop + 2 * linesize * yDiff];
	a1 = (1 - (a + 3) * pow(weightX, 2) + (a + 2) * pow(weightX, 3)) * data[startIndex + 2 * linesize * yDiff];
	a2 = (-a * weightX + (2 * a + 3) * pow(weightX, 2) - (a + 2) * pow(weightX, 3)) * data[startIndex + xDiff + 2 * linesize * yDiff];
	a3 = (a * pow(weightX, 2) - a * pow(weightX, 3)) * data[startIndex + 2 * xDiff + 2 * linesize * yDiff];
	int b3 = round(a0 + a1 + a2 + a3);

	b3 = min(b3, 255);
	b3 = max(b3, 0);

	a0 = (a * weightY - 2 * a * pow(weightY, 2) + a * pow(weightY, 3)) * b0;
	a1 = (1 - (a + 3) * pow(weightY, 2) + (a + 2) * pow(weightY, 3)) * b1;
	a2 = (-a * weightY + (2 * a + 3) * pow(weightY, 2) - (a + 2) * pow(weightY, 3)) * b2;
	a3 = (a * pow(weightY, 2) - a * pow(weightY, 3)) * b3;
	int value = round(a0 + a1 + a2 + a3);
	value = min(value, 255);
	value = max(value, 0);

	return value;
}

__device__ int calculateBicubicPolynomInterpolation(unsigned char* data, float x, float y, int xDiff, int yDiff, int linesize, int width, int height, float weightX, float weightY) {
	int startIndex = x + y * linesize;
	x = weightX;
	y = weightY;

	if (x + xDiff >= width)
		xDiff = 0;
	if (x - xDiff < 0)
		xDiff = 0;
	if (y + yDiff >= height)
		yDiff = 0;
	if (y - yDiff < 0)
		yDiff = 0;

	double value;
	double p1  = data[startIndex                           ]; //f(y, x) = f1(0, 0)
	double b1 = (double)(x - 1) * (x - 2) * (x + 1) * (y - 1) * (y - 2) * (y + 1) / 4;
	value += p1 * b1;
	p1  = data[startIndex + xDiff                           ]; //f(y, x) = f2(0, 1)
	b1 = (double)-x * (x + 1) * (x - 2) * (y - 1) * (y - 2) * (y + 1) / 4;
	value += p1 * b1;
	p1  = data[startIndex              + linesize * yDiff   ]; //f(y, x) = f3(1, 0)
	b1 = (double)-y * (x - 1) * (x - 2) * (x + 1) * (y + 1) * (y - 2) / 4;
	value += p1 * b1;
	p1  = data[startIndex + xDiff      + linesize * yDiff   ]; //f(y, x) = f4(1, 1)
	b1 = (double)x * y * (x + 1) * (x - 2) * (y + 1) * (y - 2) / 4;
	value += p1 * b1;
	p1  = data[startIndex - xDiff                           ]; //f(y, x) = f5(0, -1)
	b1 = (double)-x * (x - 1) * (x - 2) * (y - 1) * (y - 2) * (y + 1) / 12;
	value += p1 * b1;
	p1  = data[startIndex             - linesize * yDiff    ]; //f(y, x) = f6(-1, 0)
	b1 = (double)-y * (x - 1) * (x - 2) * (x + 1) * (y - 1) * (y - 2) / 12;
	value += p1 * b1;
	p1  = data[startIndex - xDiff     + linesize * yDiff    ]; //f(y, x) = f7(1, -1)
	b1 = (double)x * y * (x - 1) * (x - 2) * (y + 1) * (y - 2) / 12;
	value += p1 * b1;
	p1  = data[startIndex + xDiff     - linesize * yDiff    ]; //f(y, x) = f8(-1, 1)
	b1 = (double)x * y * (x + 1) * (x - 2) * (y - 1) * (y - 2) / 12;
	value += p1 * b1;
	p1  = data[startIndex + 2 * xDiff                       ]; //f(y, x) = f9(0, 2)
	b1 = (double)x * (x - 1) * (x + 1) * (y - 1) * (y - 2) * (y + 1) / 12;
	value += p1 * b1;
	p1 = data[startIndex              + 2 * linesize * yDiff]; //f(y, x) = f10(2, 0)
	b1 = (double)y * (x - 1) * (x - 2) * (x + 1) * (y - 1) * (y + 1) / 12;
	value += p1 * b1;
	p1 = data[startIndex - xDiff      - linesize * yDiff    ]; //f(y, x) = f11(-1, -1)
	b1 = (double)x * y * (x - 1) * (x - 2) * (y - 1) * (y - 2) / 36;
	value += p1 * b1;
	p1 = data[startIndex + 2 * xDiff  + linesize * yDiff    ]; //f(y, x) = f12(1, 2)
	b1 = (double)-x * y * (x - 1) * (x + 1) * (y + 1) * (y - 2) / 12;
	value += p1 * b1;
	p1 = data[startIndex + xDiff      + 2 * linesize * yDiff]; //f(y, x) = f13(2, 1)
	b1 = (double)-x * y * (x + 1) * (x - 2) * (y - 1) * (y + 1) / 12;
	value += p1 * b1;
	p1 = data[startIndex + 2 * xDiff  - linesize * yDiff    ]; //f(y, x) = f14(-1, 2)
	b1 = (double)-x * y * (x - 1) * (x + 1) * (y - 1) * (y - 2) / 36;
	value += p1 * b1;
	p1 = data[startIndex - xDiff     + 2 * linesize * yDiff]; //f(y, x) = f15(2, -1)
	b1 = (double)-x * y * (x - 1) * (x - 2) * (y - 1) * (y + 1) / 36;
	value += p1 * b1;
	p1 = data[startIndex + 2 * xDiff + 2 * linesize * yDiff]; //f(y, x) = f16(2, 2)
	b1 = (double)  x * y * (x - 1) * (x + 1) * (y - 1) * (y + 1) / 36;
	value += p1 * b1;

	return value;
}

__device__ int calculateAreaInterpolation(unsigned char* data, int startIndex, float scaleX, float scaleY, int linesize, int stride, float* patternX, float* patternY) {
	float colorSum = 0;
	int rScaleX = ceilf(scaleX);
	int rScaleY = ceilf(scaleY);
	float divide = 0;
	for (int i = 0; i < rScaleY; i++) {
		for (int j = 0; j < rScaleX; j++) {
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

__global__ void resizeNV12DownscaleAreaKernel(unsigned char* inputY, unsigned char* inputUV, unsigned char* outputY, unsigned char* outputUV,
	int srcWidth, int srcHeight, int srcLinesizeY, int srcLinesizeUV, int dstWidth, int dstHeight, float xRatio, float yRatio, 
	float** patternX, int patternXSize, float** patternY, int patternYSize) {
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; //coordinate of pixel (y) in destination image
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; //coordinate of pixel (x) in destination image
	if (i < dstHeight && j < dstWidth) {
		float yF = (int)(yRatio * i); //it's coordinate of pixel in source image
		float xF = (int)(xRatio * j); //it's coordinate of pixel in source image
		//bit to bit with above approach
		//float yF = (float)((i + 0.5f) * yRatio - 0.5f); //it's coordinate of pixel in source image
		//float xF = (float)((j + 0.5f) * xRatio - 0.5f); //it's coordinate of pixel in source image

		int x = floorf(xF);
		int y = floorf(yF);
		
		int index = y * srcLinesizeY + x; //index in source image
		int patternIndexX = j % patternXSize;
		int patternIndexY = i % patternYSize;
		float* rowPatternX = patternX[patternIndexX];
		float* rowPatternY = patternY[patternIndexY];
		outputY[i * dstWidth + j] = calculateAreaInterpolation(inputY, index, xRatio, yRatio, srcLinesizeY, 1, rowPatternX, rowPatternY);
		//we should take chroma for every 2 luma, also height of data[1] is twice less than data[0]
		//there are no difference between x_ratio for Y and UV also as for y_ratio because (src_height / 2) / (dst_height / 2) = src_height / dst_height
		if (i < dstHeight / 2 && j < dstWidth / 2) {
			index = y * srcLinesizeUV + x * 2; //index in source image
			int indexU, indexV;
			indexU = index;
			indexV = index + 1;
			outputUV[i * dstWidth + 2 * j] = calculateAreaInterpolation(inputUV, indexU, xRatio, yRatio, srcLinesizeUV, 2, rowPatternX, rowPatternY);
			outputUV[i * dstWidth + 2 * j + 1] = calculateAreaInterpolation(inputUV, indexV, xRatio, yRatio, srcLinesizeUV, 2, rowPatternX, rowPatternY);
		}
	}
}

__global__ void resizeNV12UpscaleAreaKernel(unsigned char* inputY, unsigned char* inputUV, unsigned char* outputY, unsigned char* outputUV,
	int srcWidth, int srcHeight, int srcLinesizeY, int srcLinesizeUV, int dstWidth, int dstHeight, float xRatio, float yRatio) {
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; //coordinate of pixel (y) in destination image
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; //coordinate of pixel (x) in destination image

	if (i < dstHeight && j < dstWidth) {
		int x = floorf(xRatio * j); //it's coordinate of pixel in source image
		float xFloat = (j + 1) - (x + 1) / xRatio;
		if (xFloat <= 0)
			xFloat = 0;
		else
			xFloat = xFloat - floorf(xFloat);

		int y = floorf(yRatio * i); //it's coordinate of pixel in source image
		float yFloat = (i + 1) - (y + 1) / yRatio;
		if (yFloat <= 0)
			yFloat = 0;
		else
			yFloat = yFloat - floorf(yFloat);

		outputY[i * dstWidth + j] = calculateBillinearInterpolation(inputY, x, y, 1, 1, srcLinesizeY, srcWidth, srcHeight, xFloat, yFloat);
		if (i < dstHeight / 2 && j < dstWidth / 2) {
			outputUV[i * dstWidth + 2 * j] = calculateBillinearInterpolation(inputUV, 2 * x, y, 2, 1, srcLinesizeUV, srcWidth, srcHeight / 2, xFloat, yFloat);
			outputUV[i * dstWidth + 2 * j + 1] = calculateBillinearInterpolation(inputUV, 2 * x + 1, y, 2, 1, srcLinesizeUV, srcWidth, srcHeight / 2, xFloat, yFloat);
		}
	}
}

__global__ void resizeNV12NearestKernel(unsigned char* inputY, unsigned char* inputUV, unsigned char* outputY, unsigned char* outputUV,
	int srcWidth, int srcHeight, int srcLinesizeY, int srcLinesizeUV, int dstWidth, int dstHeight, float xRatio, float yRatio) {

	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; //coordinate of pixel (y) in destination image
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; //coordinate of pixel (x) in destination image

	if (i < dstHeight && j < dstWidth) {
		int y = (int)(yRatio * i); //it's coordinate of pixel in source image
		int x = (int)(xRatio * j); //it's coordinate of pixel in source image
		/*
		Bit to bit with not biased approach
		float yF = (float)((i + 0.5f) * yRatio - 0.5f); //it's coordinate of pixel in source image
		float xF = (float)((j + 0.5f) * xRatio - 0.5f); //it's coordinate of pixel in source image
		int x = floor(xF);
		int y = floor(yF);
		*/
		int index = y * srcLinesizeY + x; //index in source image
		outputY[i * dstWidth + j] = inputY[index];
		//we should take chroma for every 2 luma, also height of data[1] is twice less than data[0]
		//there are no difference between x_ratio for Y and UV also as for y_ratio because (src_height / 2) / (dst_height / 2) = src_height / dst_height
		if (i < dstHeight / 2 && j < dstWidth / 2) {
			outputUV[i * dstWidth + 2 * j] = inputUV[y * srcLinesizeUV + 2 * x];
			outputUV[i * dstWidth + 2 * j + 1] = inputUV[y * srcLinesizeUV + 2 * x + 1];
		}
	}
}

__global__ void resizeNV12BilinearKernel(unsigned char* inputY, unsigned char* inputUV, unsigned char* outputY, unsigned char* outputUV,
	int srcWidth, int srcHeight, int srcLinesizeY, int srcLinesizeUV, int dstWidth, int dstHeight, float xRatio, float yRatio) {

	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; //coordinate of pixel (y) in destination image
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; //coordinate of pixel (x) in destination image

	if (i < dstHeight && j < dstWidth) {
		float yF = (float)((i + 0.5f) * yRatio - 0.5f); //it's coordinate of pixel in source image
		float xF = (float)((j + 0.5f) * xRatio - 0.5f); //it's coordinate of pixel in source image
		int x = floorf(xF);
		int y = floorf(yF);
		float weightX = xF - x;
		float weightY = yF - y;
		
		//need to avoid empty lines at the top and left corners
		if (x < 0) {
			x = 0;
			weightX = 0;
		}

		if (y < 0) {
			y = 0;
			weightY = 0;
		}

		if (x > srcWidth - 1) {
			x = srcWidth - 1;
			weightX = 0;
		}

		if (y > srcHeight - 1) {
			y = srcHeight - 1;
			weightY = 0;
		}

		outputY[i * dstWidth + j] = calculateBillinearInterpolation(inputY, x, y, 1, 1, srcLinesizeY, srcWidth, srcHeight, weightX, weightY);
		//we should take chroma for every 2 luma, also height of data[1] is twice less than data[0]
		//there are no difference between x_ratio for Y and UV also as for y_ratio because (src_height / 2) / (dst_height / 2) = src_height / dst_height
		if (i < dstHeight / 2 && j < dstWidth / 2) {
			outputUV[i * dstWidth + 2 * j] = calculateBillinearInterpolation(inputUV, 2 * x, y, 2, 1, srcLinesizeUV, srcWidth, srcHeight / 2, weightX, weightY);
			outputUV[i * dstWidth + 2 * j + 1] = calculateBillinearInterpolation(inputUV, 2 * x + 1, y, 2, 1, srcLinesizeUV, srcWidth, srcHeight / 2, weightX, weightY);
		}
	}
}

__global__ void resizeNV12BicubicKernel(unsigned char* inputY, unsigned char* inputUV, unsigned char* outputY, unsigned char* outputUV,
	int srcWidth, int srcHeight, int srcLinesizeY, int srcLinesizeUV, int dstWidth, int dstHeight, float xRatio, float yRatio) {

	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; //coordinate of pixel (y) in destination image
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; //coordinate of pixel (x) in destination image

	if (i < dstHeight && j < dstWidth) {
		double yF = (double)((i + 0.5f) * yRatio - 0.5f); //it's coordinate of pixel in source image
		double xF = (double)((j + 0.5f) * xRatio - 0.5f); //it's coordinate of pixel in source image
		int x = floor(xF);
		int y = floor(yF);
		double weightX = xF - x;
		double weightY = yF - y;

		//need to avoid empty lines at the top and left corners
		if (x < 0) {
			x = 0;
			weightX = 0;
		}

		if (y < 0) {
			y = 0;
			weightY = 0;
		}

		if (x > srcWidth - 1) {
			x = srcWidth - 1;
			weightX = 0;
		}

		if (y > srcHeight - 1) {
			y = srcHeight - 1;
			weightY = 0;
		}

		outputY[i * dstWidth + j] = calculateBicubicSplineInterpolation(inputY, x, y, 1, 1, srcLinesizeY, srcWidth, srcHeight, weightX, weightY);
		//we should take chroma for every 2 luma, also height of data[1] is twice less than data[0]
		//there are no difference between x_ratio for Y and UV also as for y_ratio because (src_height / 2) / (dst_height / 2) = src_height / dst_height
		if (i < dstHeight / 2 && j < dstWidth / 2) {
			outputUV[i * dstWidth + 2 * j] = calculateBicubicSplineInterpolation(inputUV, 2 * x, y, 2, 1, srcLinesizeUV, srcWidth, srcHeight / 2, weightX, weightY);
			outputUV[i * dstWidth + 2 * j + 1] = calculateBicubicSplineInterpolation(inputUV, 2 * x + 1, y, 2, 1, srcLinesizeUV, srcWidth, srcHeight / 2, weightX, weightY);
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

		if (dynScale > std::numeric_limits<float>::epsilon()) {
			//push rest to pattern
			pattern[currentID].push_back(dynScale);
			rest = 1 - dynScale;
		}

		while (pattern[currentID].size() < ceil(scale))
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

int resizeKernel(AVFrame* src, AVFrame* dst, bool crop, ResizeOptions resize, int maxThreadsPerBlock, cudaStream_t * stream) {
	unsigned char* outputY = nullptr;
	unsigned char* outputUV = nullptr;
	cudaError err = cudaMalloc(&outputY, resize.width * resize.height * sizeof(unsigned char)); //in resize we don't change color format
	err = cudaMalloc(&outputUV, resize.width * (resize.height / 2) * sizeof(unsigned char));
	//need to execute for width and height
	dim3 threadsPerBlock(64, maxThreadsPerBlock / 64);
	int blockX = std::ceil(resize.width / (float)threadsPerBlock.x);
	int blockY = std::ceil(resize.height / (float)threadsPerBlock.y);
	dim3 numBlocks(blockX, blockY);
	float xRatio = (float)(src->width) / resize.width; //if not -1 we should examine 2x2 square with top-left corner in the last pixel of src, so it's impossible
	float yRatio = (float)(src->height) / resize.height;
	int pitchY = src->linesize[0] ? src->linesize[0] : src->width;
	int pitchUV = src->linesize[1] ? src->linesize[1] : src->width;
	switch (resize.type) {
	case ResizeType::BILINEAR:
		resizeNV12BilinearKernel << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], outputY, outputUV,
			src->width, src->height, pitchY, pitchUV,
			resize.width, resize.height, xRatio, yRatio);
		break;
	case ResizeType::NEAREST:
		resizeNV12NearestKernel << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], outputY, outputUV,
			src->width, src->height, pitchY, pitchUV,
			resize.width, resize.height, xRatio, yRatio);
		break;
	case ResizeType::AREA:
		//The smart "area" algorithm is used only in case of downscaling
		if (xRatio > 1 && yRatio > 1) {
			std::vector<std::vector<float> > patternX;
			std::vector<std::vector<float> > patternY;
			generateResizePattern(xRatio, patternX);
			generateResizePattern(yRatio, patternY);

			float** patternXCUDA = copy2DArray(patternX, xRatio);
			float** patternYCUDA = copy2DArray(patternY, yRatio);

			//Here we should decide which AREA algorithm to use
			resizeNV12DownscaleAreaKernel << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], outputY,
				outputUV, src->width, src->height, pitchY, pitchUV, resize.width, resize.height, xRatio, yRatio,
				patternXCUDA, patternX.size(), patternYCUDA, patternY.size());

			free2DArray(patternXCUDA, patternX.size(), xRatio);
			free2DArray(patternYCUDA, patternY.size(), yRatio);
		}
		//otherwise bilinear algorithm with some weight adjustments is used
		else {
			resizeNV12UpscaleAreaKernel << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], outputY,
				outputUV, src->width, src->height, pitchY, pitchUV, resize.width, resize.height, xRatio, yRatio);
		}
		break;
	case ResizeType::BICUBIC:
		resizeNV12BicubicKernel << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], outputY, outputUV,
			src->width, src->height, pitchY, pitchUV,
			resize.width, resize.height, xRatio, yRatio);
		break;
	}

	if (crop) {
		err = cudaFree(dst->data[0]);
		err = cudaFree(dst->data[1]);
	}

	dst->data[0] = outputY;
	dst->data[1] = outputUV;
	return err;
}
