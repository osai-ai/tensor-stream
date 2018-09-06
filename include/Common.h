#pragma once

enum {
	OK = 0
};


#define CHECK_STATUS(status) if (status != 0) return status;
