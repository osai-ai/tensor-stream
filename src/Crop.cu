#include "cuda.h"
#include "VideoProcessor.h"

__global__ void cropKernel(unsigned char* inputY, unsigned char* inputUV, unsigned char* outputY, unsigned char* outputUV,
	int srcPitchY, int srcPitchUV, int topLeftX, int topLeftY, int botRightX, int botRightY) {
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; //coordinate of pixel (y) in destination image
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; //coordinate of pixel (x) in destination image
	if (j < botRightX - topLeftX && i < botRightY - topLeftY) {
		int UVRow = i / 2;
		int UVCol = j % 2 == 0 ? j : j - 1;
		int UIndexSrc = (topLeftY / 2 + UVRow) * srcPitchUV /*pitch?*/ + (UVCol + topLeftX);
		int VIndexSrc = (topLeftY / 2 + UVRow) * srcPitchUV /*pitch?*/ + (UVCol + topLeftX + 1);

		int UIndexDst = UVRow * (botRightX - topLeftX) /*pitch?*/ + UVCol;
		int VIndexDst = UVRow * (botRightX - topLeftX) /*pitch?*/ + UVCol + 1;

		outputY[j + i * (botRightX - topLeftX)] = inputY[(topLeftX + j) + (topLeftY + i) * srcPitchY];
		outputUV[UIndexDst] = inputUV[UIndexSrc];
		outputUV[VIndexDst] = inputUV[VIndexSrc];
	}
}

int cropHost(AVFrame* src, AVFrame* dst, CropOptions crop, int maxThreadsPerBlock, cudaStream_t * stream) {
	cudaError err;
	int cropWidth = std::get<0>(crop.rightBottomCorner) - std::get<0>(crop.leftTopCorner);
	int cropHeight = std::get<1>(crop.rightBottomCorner) - std::get<1>(crop.leftTopCorner);
	unsigned char* outputY = nullptr;
	unsigned char* outputUV = nullptr;
	err = cudaMalloc(&outputY, cropWidth * cropHeight * sizeof(unsigned char));
	err = cudaMalloc(&outputUV, cropWidth * (cropHeight / 2) * sizeof(unsigned char));
	//need to execute for width and height
	dim3 threadsPerBlock(64, maxThreadsPerBlock / 64);
	int blockX = std::ceil(cropWidth / (float)threadsPerBlock.x);
	int blockY = std::ceil(cropHeight / (float)threadsPerBlock.y);
	dim3 numBlocks(blockX, blockY);

	int pitchY = src->linesize[0] ? src->linesize[0] : src->width;
	int pitchUV = src->linesize[1] ? src->linesize[1] : src->width;

	cropKernel << <numBlocks, threadsPerBlock, 0, *stream >> > (src->data[0], src->data[1], outputY, outputUV,
		pitchY, pitchUV, std::get<0>(crop.leftTopCorner), std::get<1>(crop.leftTopCorner),
											std::get<0>(crop.rightBottomCorner), std::get<1>(crop.rightBottomCorner));

	dst->data[0] = outputY;
	dst->data[1] = outputUV;

	return err;
}