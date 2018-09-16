/*
The main class which provide API for python and manages all calls to components. Consumers should work with this class.
*/
#include "Common.h"
#include "stdio.h"

void printContext() {
	CUcontext test_cu;
	auto cu_err = cuCtxGetCurrent(&test_cu);
	printf("Context %x\n", test_cu);
}
