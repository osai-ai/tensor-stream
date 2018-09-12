#pragma once
#include <windows.h>
#include <psapi.h>

enum {
	OK = 0
};


#define CHECK_STATUS(status) if (status != 0) return status;

static int maxConsumers = 5;

void printMemory();