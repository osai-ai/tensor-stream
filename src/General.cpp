/*
The main class which provide API for python and manages all calls to components. Consumers should work with this class.
*/
#include "Common.h"
#include "stdio.h"

void printMemory() {
	PROCESS_MEMORY_COUNTERS pmc;
	BOOL result = GetProcessMemoryInfo(GetCurrentProcess(),
		&pmc,
		sizeof(pmc));
	printf("\tWorkingSetSize: 0x%08X - %u\n", pmc.WorkingSetSize,
		pmc.WorkingSetSize / 1024);
	printf("\tQuotaPeakPagedPoolUsage: 0x%08X - %u\n",
		pmc.QuotaPeakPagedPoolUsage, pmc.QuotaPeakPagedPoolUsage / 1024);
	printf("\tQuotaPagedPoolUsage: 0x%08X - %u\n", pmc.QuotaPagedPoolUsage,
		pmc.QuotaPagedPoolUsage / 1024);
	printf("\tQuotaPeakNonPagedPoolUsage: 0x%08X - %u\n",
		pmc.QuotaPeakNonPagedPoolUsage, pmc.QuotaPeakNonPagedPoolUsage / 1024);
	printf("\tQuotaNonPagedPoolUsage: 0x%08X - %u\n", pmc.QuotaNonPagedPoolUsage,
		pmc.QuotaNonPagedPoolUsage / 1024);
	printf("\tPagefileUsage: 0x%08X - %u\n", pmc.PagefileUsage,
		pmc.PagefileUsage / 1024);
	printf("\tPeakPagefileUsage: 0x%08X - %u\n", pmc.PeakPagefileUsage,
		pmc.PeakPagefileUsage / 1024);
	printf("\tcb: 0x%08X - %u\n", pmc.cb, pmc.cb / 1024);
	printf("\n\n");
}

void printContext() {
	CUcontext test_cu;
	auto cu_err = cuCtxGetCurrent(&test_cu);
	printf("Context %x\n", test_cu);
}
