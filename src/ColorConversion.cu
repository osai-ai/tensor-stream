#include <libavutil/frame.h>
#include "cuda.h"
#include "VideoProcessor.h"
#include <iostream>

__device__ void NV12toRGB24Kernel(unsigned char* Y, unsigned char* UV, int* R, int* G, int* B, int i, int j, int pitchNV12) {
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
	float YVal = max(0.f, Y[indexNV12] - 16.f) * 1.163999557f;

	float RVal = 1.5959997177f * (V - 128) + 0.5f;
	*R = YVal + RVal;
	*R = min(*R, 255);
	*R = max(*R, 0);

	float BVal = 2.017999649f  * (U - 128) + 0.5f;
	*B = YVal + BVal;
	*B = min(*B, 255);
	*B = max(*B, 0);

	float GVal = -0.812999725f  * (V - 128) - 0.390999794f * (U - 128) + 0.5f;
	*G = YVal + GVal;
	*G = min(*G, 255);
	*G = max(*G, 0);
}

template< class T >
__global__ void NV12ToRGB24KernelPlanar(unsigned char* Y, unsigned char* UV, T* RGB, int width, int height, int pitchNV12, int pitchRGB, bool swapRB, bool normalization) {
	unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < height && j < width) {
		int R, G, B;
		NV12toRGB24Kernel(Y, UV, &R, &G, &B, i, j, pitchNV12);
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
__global__ void NV12ToRGB24KernelMerged(unsigned char* Y, unsigned char* UV, T* RGB, int width, int height, int pitchNV12, int pitchRGB, bool swapRB, bool normalization) {
	unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < height && j < width) {
		int R, G, B;
		NV12toRGB24Kernel(Y, UV, &R, &G, &B, i, j, pitchNV12);
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
		if (normalization)
			dst[index] /= 255;
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
		int indexSrc = j + i * pitchNV12;
		if (index % 2 == 0) {
			int indexDest = index * 2;
			//max UV for NV12 - j/2 i/2
			//       for UYVY - j/2 i

			unsigned char UValue = calculateUYVYChromaVertical(UV, i, j, pitchNV12, height);
			dest[indexDest] = UValue;
			if (normalization)
				dest[indexDest] /= 255;
			dest[indexDest + 1] = Y[indexSrc];
			if (normalization)
				dest[indexDest + 1] /= 255;
			unsigned char VValue = calculateUYVYChromaVertical(UV, i, j + 1, pitchNV12, height);
			dest[indexDest + 2] = VValue;
			if (normalization)
				dest[indexDest + 2] /= 255;
		}
		else {
			int indexDest = index * 2 + 1;
			dest[indexDest] = Y[indexSrc];
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

template< class T >
__global__ void RGBMergedToHSVMerged(T* RGB, T* dest, int width, int height) {
	unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < height && j < width) {
		int index = j * 3 + i * width * 3;
		T R = RGB[index];
		T G = RGB[index + 1];
		T B = RGB[index + 2];
		
		T* H = &dest[index];
		T* S = &dest[index + 1];
		T* V = &dest[index + 2];
		
		T minVal = min(min(R, G), B);
		T maxVal = max(max(R, G), B);
		T delta = maxVal - minVal;

		*V = maxVal;

		*S = 0;
		if (maxVal != 0) {
			*S = 1 - minVal / maxVal;
		}

		if (maxVal == minVal) {
			*H = 0;
			return;
		}
		else if (R == maxVal && G >= B)
			*H = 60 * (G - B) / delta;
		else if (R == maxVal && G < B)
			*H = 60 * (G - B) / delta + 360;
		else if (G == maxVal)
			*H = 60 * (B - R) / delta + 120;
		else if (B == maxVal)
			*H = 60 * (R - G) / delta + 240;
		if (*H < 0)
			*H += 360;
		
		*H /= 360;
	}
}

//width is half of real width, the same for height
__global__ void YUV420PToNV12(uint8_t* srcU, uint8_t* srcV, uint8_t* destUV, int width, int height) {
	unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < height && j < width) {
		int srcIndex = j + i * width;
		int dstIndex = j * 2 + i * width * 2;
		destUV[dstIndex] = srcU[srcIndex];
		destUV[dstIndex + 1] = srcV[srcIndex];
	}
}

int convertSWToHW(AVFrame* src, AVFrame* dst, int maxThreadsPerBlock, cudaStream_t* stream) {
	cudaError err = cudaSuccess;
	int width = src->width;
	int height = src->height;
	//first of all we have to copy data from YUV420P to NV12 in case of SW decoder and create AVFrame structure similar to output of CUDA decoding
	uint8_t *Y, *U, *V;
	err = cudaMalloc(&Y, width * height * sizeof(uint8_t));
	err = cudaMalloc(&U, width / 2 * height / 2 * sizeof(uint8_t));
	err = cudaMalloc(&V, width / 2 * height / 2 * sizeof(uint8_t));

	err = cudaMemcpy2D(Y, width, (void*)src->data[0], src->linesize[0], width, height, cudaMemcpyHostToDevice);
	err = cudaMemcpy2D(U, width / 2, (void*)src->data[1], src->linesize[1], width / 2, height / 2, cudaMemcpyHostToDevice);
	err = cudaMemcpy2D(V, width / 2, (void*)src->data[2], src->linesize[2], width / 2, height / 2, cudaMemcpyHostToDevice);
	
	uint8_t *UV;
	err = cudaMalloc(&UV, width * height / 2 * sizeof(uint8_t));

	//need to execute for width and height
	dim3 threadsPerBlock(64, maxThreadsPerBlock / 64);

	//blocks for merged format
	int blockX = std::ceil(width / 2 / (float)threadsPerBlock.x);
	int blockY = std::ceil(src->height / 2 / (float)threadsPerBlock.y);

	dim3 numBlocks(blockX, blockY);

	YUV420PToNV12<< <numBlocks, threadsPerBlock, 0, *stream >> > (U, V, UV, width / 2, height / 2);

	cudaFree(U);
	cudaFree(V);
	dst->data[0] = Y;
	dst->data[1] = UV;
	dst->width = width;
	dst->height = height;
	dst->format = src->format;

	return VREADER_OK;
}

template <class T>
int colorConversionKernel(AVFrame* src, AVFrame* dst, ColorOptions color, int maxThreadsPerBlock, cudaStream_t* stream) {
	float channels = channelsByFourCC(color.dstFourCC);
	/*
	src in GPU nv12, dst in CPU rgb (packed)
	*/
	int width = src->width;
	int height = src->height;

	//need to execute for width and height
	dim3 threadsPerBlock(64, maxThreadsPerBlock / 64);

	//blocks for merged format
	int blockX = std::ceil(width / (float)threadsPerBlock.x);
	int blockY = std::ceil(dst->height / (float)threadsPerBlock.y);
	
	dim3 numBlocks(blockX, blockY);

	void* destination = nullptr;
	cudaError err = cudaSuccess;
	//depends on fact of resize
	int pitchNV12 = src->linesize[0] ? src->linesize[0] : width;
	bool swapRB = false;
	switch (color.dstFourCC) {
		case BGR24:
			swapRB = true;
			err = cudaMalloc(&destination, channels * width * height * sizeof(T));

			if (color.planesPos == Planes::PLANAR) {
				int pitchRGB = width;
				NV12ToRGB24KernelPlanar<T> << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], (T*) destination, width, height, pitchNV12, pitchRGB, swapRB, color.normalization);
			}
			else {
				int pitchRGB = channels * width;
				NV12ToRGB24KernelMerged<T> << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], (T*) destination, width, height, pitchNV12, pitchRGB, swapRB, color.normalization);
			}
		break;
		case RGB24:
			err = cudaMalloc(&destination, channels * width * height * sizeof(T));

			if (color.planesPos == Planes::PLANAR) {
				int pitchRGB = width;
				NV12ToRGB24KernelPlanar<T> << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], (T*) destination, width, height, pitchNV12, pitchRGB, swapRB, color.normalization);
			}
			else {
				int pitchRGB = channels * width;
				NV12ToRGB24KernelMerged<T> << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], (T*) destination, width, height, pitchNV12, pitchRGB, swapRB, color.normalization);
				cudaStreamSynchronize(*stream);

			}
		break;
		case Y800:
			err = cudaMalloc(&destination, channels * width * height * sizeof(T));

			NV12ToY800 << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], (T*) destination, width, height, pitchNV12, color.normalization);
		break;
		case UYVY:
			err = cudaMalloc(&destination, channels * width * height * sizeof(T));

			NV12ToUYVY << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], (T*) destination, width, height, pitchNV12, color.normalization);
		break;
		case YUV444: 
		{
			err = cudaMalloc(&destination, channels * width * height * sizeof(T));

			NV12ToUYVY << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], (T*) destination, width, height, pitchNV12, /*normalization*/false);
			T* destinationYUV444 = nullptr;
			err = cudaMalloc(&destinationYUV444, channels * width * height * sizeof(T));
			//It's more convinient to work with width*height than with any other sizes
			UYVYToYUV444 << <numBlocks, threadsPerBlock, 0, *stream >> > ((T*) destination, (T*) destinationYUV444, width, height, color.normalization);
			cudaFree(destination);
			destination = destinationYUV444;

		}
		break;
		case NV12:
			err = cudaMalloc(&destination, channels * width * height * sizeof(T));

			NV12MergeBuffers << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], (T*) destination, width, height, pitchNV12, color.normalization);
		break;
		case HSV:
		{
			err = cudaMalloc(&destination, channels * width * height * sizeof(float));

			int pitchRGB = channels * width;
			NV12ToRGB24KernelMerged<float> << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], (float*) destination, 
																						width, height, pitchNV12, pitchRGB, /*swapRB*/ false, /*normalization*/ true);
			float* destinationHSV = nullptr;
			err = cudaMalloc(&destinationHSV, channels * width * height * sizeof(float));
			RGBMergedToHSVMerged<float> << <numBlocks, threadsPerBlock, 0, *stream >> > ((float*) destination, (float*) destinationHSV, width, height);
			cudaFree(destination);
			destination = destinationHSV;
		}
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