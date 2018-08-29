#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
// basic file operations
#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef _DEBUG
#undef _DEBUG
#include <torch/torch.h>
#include <THC/THC.h>
#include <ATen/ATen.h>
#define _DEBUG
#else
#include <torch/torch.h>
#include <THC/THC.h>
#include <ATen/ATen.h>
#endif

at::Tensor load() {
	CUdeviceptr data_done;
	cudaError_t err2 = cudaMalloc(reinterpret_cast<void**>(&data_done), 16 * sizeof(float));
	auto f = torch::CUDA(at::kFloat).tensorFromBlob(reinterpret_cast<void*>(data_done), { 16 });
	return f;
}

PYBIND11_MODULE(Parser, m) {
	m.def("load", &load, "load data");
}