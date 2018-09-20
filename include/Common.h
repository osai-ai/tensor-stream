#pragma once
#include "cuda.h"
//#define TRACER
#ifdef TRACER
#include "nvToolsExt.h"
#include "Windows.h"
#endif

#define TIMINGS

enum {
	OK = 0,
	REPEAT = 1
};


#define CHECK_STATUS(status) if (status != 0) return status;

static int maxConsumers = 5;

void printMemory();