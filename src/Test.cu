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


__global__ void change(unsigned char* Y, unsigned char* UV) {
	for (int i = 0; i < 1000; i++) {
		unsigned char y = Y[i];
		unsigned char U = UV[i];
		Y[i] = 255;
		UV[i] = 255;
	}
}

void change_pixels(unsigned char* Y, unsigned char* UV) {
	change << <1, 1 >> > (Y, UV);
}