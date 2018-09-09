#pragma once
#include <windows.h>
#include <psapi.h>

enum {
	OK = 0
};


#define CHECK_STATUS(status) if (status != 0) return status;



void printMemory();