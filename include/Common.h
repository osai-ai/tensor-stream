#pragma once
#include "cuda.h"
#ifdef TRACER
#include "nvToolsExt.h"
#endif
enum {
	OK = 0,
	REPEAT = 1
};


#define CHECK_STATUS(status) if (status != 0) return status;

static int maxConsumers = 5;

void printMemory();