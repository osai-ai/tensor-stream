#include <libavutil/frame.h>

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


__global__ void change(unsigned char* Y, unsigned char* UV, unsigned char* RGB, int width, int height) {
	/*
		R = 1.164(Y - 16) + 1.596(V - 128)
		B = 1.164(Y - 16)                   + 2.018(U - 128)
		G = 1.164(Y - 16) - 0.813(V - 128)  - 0.391(U - 128)
	*/
	/*
	in case of NV12 we have Y component for every pixel and UV for every 2x2 Y
	*/
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int UVRow = i / 2;
			int UVCol = j % 2 == 0 ? j : j - 1;
			int UIndex = UVRow * width + UVCol;
			int VIndex = UVRow * width + UIndex + 1;
			unsigned char U = UV[UIndex];
			unsigned char V = UV[VIndex];
			int index = j + i * width;
			unsigned char YVal = Y[index];
			unsigned char RVal = 1.164 * (YVal - 16) + 1.596*(V - 128);
			unsigned char GVal = 1.164 * (YVal - 16) + 2.018*(U - 128);
			unsigned char BVal = 1.164 * (YVal - 16) - 0.813*(V - 128) - 0.391*(U - 128);
			RGB[index + 0 /*R*/] = RVal;
			RGB[index + 1 /*G*/] = GVal;
			RGB[index + 2 /*B*/] = BVal;
		}
	}
}

void change_pixels(AVFrame* src, AVFrame* dst) {
	/*
	src in GPU nv12, dst in CPU rgb (packed)
	*/
	int width = dst->width;
	int height = dst->height;
	unsigned char* RGB;
	cudaMalloc(&RGB, width * height * sizeof(unsigned char));
	cudaMemcpy(RGB, dst->data[0], width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
	change << <1, 1 >> > (src->data[0], src->data[1], RGB, width, height);
	cudaMemcpy(dst->data[0], RGB, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}