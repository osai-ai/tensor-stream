#pragma once
#include "cuda.h"
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#ifdef _WIN32
#include "nvToolsExt.h"
#include "Windows.h"
#endif

#define TIMINGS

enum {
	OK = 0,
	REPEAT = 1
};

#define CHECK_STATUS(status) if (status != 0) return status;

enum LogsLevel {
	NONE,
	LOW, //start and end with frame indexes
	MEDIUM, //LOW + times
	HIGH, //whole pipeline with times
	DETAILED //HIGH + notices if latency is greater
};


static std::shared_ptr<FILE> logsName = nullptr;
static LogsLevel logsLevel = NONE;
static std::mutex logsMutex;

#define START_LOG(messageIn) \
	{ \
		clock_t start; \
		{ \
			std::unique_lock<std::mutex> locker(logsMutex); \
			if (logsLevel && logsName) \
			{ \
				std::string finalMessage = messageIn + std::string(" +\n"); \
				fwrite(finalMessage.c_str(), finalMessage.length(), 1, logsName.get()); \
				fflush(logsName.get()); \
				start = clock(); \
			} \
		} \

#define END_LOG(messageOut) \
		{ \
			std::unique_lock<std::mutex> locker(logsMutex); \
			if (logsLevel && logsName) { \
				std::string time = std::to_string(clock() - start); \
				std::string finalMessage = messageOut + std::string(" -\ntime: ") + time + std::string("ms\n"); \
				fwrite(finalMessage.c_str(), finalMessage.length(), 1, logsName.get()); \
				fflush(logsName.get()); \
			} \
		} \
	}
	
const int maxConsumers = 5;

template <class T>
T findFree(std::string consumerName, std::vector<std::pair<std::string, T> >& entities) {
	for (auto& item : entities) {
		if (item.first == consumerName) {
			return item.second;
		}
		else if (item.first == "empty") {
			item.first = consumerName;
			return item.second;
		}
	}
	return nullptr;
}